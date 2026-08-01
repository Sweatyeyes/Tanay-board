#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Распознаёт табло со скриншота и складывает результат в board.json.
Запускается на GitHub Actions (Ubuntu), поэтому кириллица здесь безопасна.

Использование:
    python recognize.py board.png out/board.json out/debug.png
"""

import json
import os
import re
import sys
import difflib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import pytesseract

# Положение таблицы не задаётся числами: окно RMS съезжает между снимками,
# поэтому панели, верх таблицы и шаг строки ищутся заново каждый раз.

# границы колонок внутри строки - долей от ширины панели
COL_NUM  = (0.006, 0.093)
COL_NAME = (0.086, 0.759)
COL_CAT  = (0.762, 0.991)

# известные категории - результат распознавания подгоняется к ближайшей.
# AFF бывает уровней 1-7 плюс дефисные "AFF 8-1" и "AFF 8-2".
KNOWN_CATS = [
    "Спортивный", "ФВ",
    "AFF 1", "AFF 2", "AFF 3", "AFF 4", "AFF 5", "AFF 6", "AFF 7",
    "AFF 8-1", "AFF 8-2",
    "ХК ТМ3 90", "ХК ТМ4 90",
    "ТМ4000 90", "ТМ4000 100", "ТМ4000 120",
    "ТМ3000 90", "ТМ3000 100", "ТМ3000 120",
]

# латинские двойники кириллицы + частые ошибки распознавания цифр:
# S - это криво прочитанная 3, $ - это 9
LAT2CYR = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У", "S": "3", "$": "9",
})

# буквы, неотличимые от цифр на этом шрифте, приводим к цифрам.
# Применяется и к распознанному, и к эталонам (см. CAT_CANON ниже),
# поэтому сравнение остаётся честным.
CYR2DIGIT = str.maketrans({
    "О": "0", "З": "3", "Б": "6", "В": "8", "Ч": "4", "І": "1",
})


def canon_cat(text):
    """Сжимает категорию до канонического вида для сравнения."""
    t = (text or "").upper().translate(LAT2CYR).translate(CYR2DIGIT)
    return re.sub(r"[^А-ЯЁ0-9]", "", t)


# те же категории в каноническом виде - для сравнения с распознанным,
# у которого пробелы и похожие буквы гуляют как попало.
# AFF-категории сюда не входят: у них отдельный разбор по буквам.
CAT_CANON = {canon_cat(c): c for c in KNOWN_CATS if not c.startswith("AFF")}
CAT_CANON["ФЕ"] = "ФВ"   # Е вместо В - частая ошибка на этом шрифте

TESS_RU = "--oem 1 --psm 7 -l rus"
TESS_MIX = "--oem 1 --psm 7 -l rus+eng"
TESS_MIX8 = "--oem 1 --psm 8 -l rus+eng"

SAMPLES = []   # образцы ячеек для отладочной картинки


def prep(img, invert=True, scale=4, threshold=110, margin=16,
         hline_thr=0.85, smooth=False):
    """Готовит картинку к распознаванию.

    Фон определяется автоматически (самый частый цвет), текстом считается всё,
    что от него заметно отличается. Так одинаково хорошо работают и белые буквы
    на чёрном, и красные или голубые на оранжевом.

    Дальше: обрезка по основной полосе текста (чтобы выбросить линии-разделители),
    увеличение и белые поля - Tesseract заметно точнее, когда текст не у края.

    hline_thr - порог стирания горизонтальных линий (доля ширины кадра).
    Обычно 0.85, но у надписей в рамке (например "Работа до ...") рамка
    занимает меньше половины ширины кадра и порог 0.85 её не берёт -
    тогда ocr() повторяет попытку с порогом пониже.

    smooth - сгладить ступеньки после увеличения. Жирный блочный шрифт после
    резкого увеличения иногда читается хуже, чем слегка размытый; какой вариант
    лучше - решается по уверенности Tesseract (см. ocr_best).
    """
    rgb = np.array(img.convert("RGB")).astype(int)
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    bg = colors[counts.argmax()]
    dist = np.abs(rgb - bg).sum(axis=2)
    b = np.where(dist > 90, 0, 255).astype("uint8")   # текст чёрный, фон белый

    # Сначала стираем сплошные рамки: вертикальные линии по краям панели и
    # горизонтальные разделители строк. Иначе они склеиваются с текстом.
    h, w = b.shape
    dark = b < 128
    col_has = dark.sum(axis=0)
    for x in range(w):
        if col_has[x] > h * 0.85:
            b[:, x] = 255
    dark = b < 128
    row_has = dark.sum(axis=1)
    for y in range(h):
        if row_has[y] > w * hline_thr:
            b[y, :] = 255

    # Оставляем только основную полосу с текстом. Строки-разделители и обрывки
    # линий сверху/снизу идут отдельными группами - их отбрасываем.
    dark = b < 128
    row_has = dark.sum(axis=1)
    groups, start = [], None
    for y in range(h):
        if row_has[y] > 0 and start is None:
            start = y
        elif row_has[y] == 0 and start is not None:
            groups.append((start, y)); start = None
    if start is not None:
        groups.append((start, h))

    if groups:
        # Выбираем группу с наибольшим количеством чернил, отбрасывая сплошные
        # заливки: линии-разделители и края соседних панелей.
        def ink(g):
            seg = row_has[g[0]:g[1]]
            solid = (seg > w * 0.85).mean()
            if solid > 0.5:
                return -1
            return int(seg.sum())
        best = max(groups, key=ink)
        b = b[best[0]:best[1], :]

    dark = b < 128
    cols = [x for x in range(b.shape[1]) if dark[:, x].any()]
    if cols:
        b = b[:, max(0, min(cols) - 1):min(b.shape[1], max(cols) + 2)]

    if b.size == 0 or b.shape[0] < 3:
        return Image.new("L", (10, 10), 255)

    out = Image.fromarray(b)
    out = out.resize((out.width * scale, out.height * scale), Image.NEAREST)

    canvas = Image.new("L", (out.width + margin * 2, out.height + margin * 2), 255)
    canvas.paste(out, (margin, margin))
    if smooth:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=scale * 0.5))
    return canvas


def ocr(img, cfg=TESS_RU, invert=True, scale=4):
    """Распознаёт одну строку текста.

    Если первая попытка дала пустоту, пробует ещё раз с агрессивным
    стиранием горизонтальных линий - выручает надписи в рамке.
    """
    if img.width < 3 or img.height < 3:
        return ""
    try:
        text = pytesseract.image_to_string(
            prep(img, invert, scale), config=cfg).strip()
        if not text:
            text = pytesseract.image_to_string(
                prep(img, invert, scale, hline_thr=0.4), config=cfg).strip()
        return text
    except Exception as e:
        sys.stderr.write("ocr error: %s\n" % e)
        return ""


def ocr_with_conf(img, cfg):
    """Распознаёт и возвращает (текст, средняя уверенность)."""
    try:
        data = pytesseract.image_to_data(
            img, config=cfg, output_type=pytesseract.Output.DICT)
    except Exception as e:
        sys.stderr.write("ocr error: %s\n" % e)
        return "", -1.0
    words, confs = [], []
    for txt, c in zip(data.get("text", []), data.get("conf", [])):
        t = (txt or "").strip()
        try:
            c = float(c)
        except (TypeError, ValueError):
            c = -1.0
        if t and c >= 0:
            words.append(t)
            confs.append(c)
    if not words:
        return "", -1.0
    return " ".join(words), sum(confs) / len(confs)


def ocr_best(img, cfg=TESS_RU):
    """Распознаёт двумя вариантами препроцессинга, берёт более уверенный.

    Резкое увеличение оставляет ступеньки на жирном шрифте, из-за них
    Tesseract путает похожие буквы (е-а, и-ы). Сглаживание убирает ступеньки,
    а увеличение x6 против x4 в замерах на живых снимках дало ещё несколько
    процентов точности. Между двумя сглаженными вариантами решает
    уверенность самого Tesseract.
    """
    if img.width < 3 or img.height < 3:
        return ""
    best_text, best_conf = "", -1.0
    for scale in (6, 4):
        text, conf = ocr_with_conf(prep(img, scale=scale, smooth=True), cfg)
        if conf > best_conf and text:
            best_text, best_conf = text, conf
    if not best_text:
        best_text = ocr(img, cfg=cfg)
    return best_text


# Имена и отчества - закрытые словари, распознанное подгоняется к ближайшему.
# Фамилии не трогаем: их список открытый, а уведомления всё равно сверяют
# фамилию с допуском на ошибки распознавания.
MALE_NAMES = [
    "Александр", "Алексей", "Альберт", "Анатолий", "Андрей", "Антон",
    "Аркадий", "Арсений", "Артём", "Артем", "Артур", "Богдан", "Борис",
    "Вадим", "Валентин", "Валерий", "Василий", "Виктор", "Виталий",
    "Владимир", "Владислав", "Всеволод", "Вячеслав", "Геннадий", "Георгий",
    "Герман", "Глеб", "Григорий", "Давид", "Дамир", "Даниил", "Данил",
    "Данила", "Денис", "Дмитрий", "Евгений", "Егор", "Иван", "Игорь",
    "Ильдар", "Илья", "Камиль", "Кирилл", "Константин", "Лев", "Леонид",
    "Максим", "Марат", "Марк", "Матвей", "Михаил", "Никита", "Николай",
    "Олег", "Павел", "Пётр", "Петр", "Равиль", "Рамиль", "Ренат", "Ринат",
    "Роберт", "Родион", "Роман", "Руслан", "Рустам", "Савелий", "Семён",
    "Семен", "Сергей", "Станислав", "Степан", "Тарас", "Тимофей", "Тимур",
    "Фёдор", "Федор", "Эдуард", "Эмиль", "Юрий", "Ян", "Ярослав",
]

FEMALE_NAMES = [
    "Аделина", "Александра", "Алина", "Алла", "Анастасия", "Анна", "Арина",
    "Валентина", "Валерия", "Варвара", "Вера", "Вероника", "Виктория",
    "Галина", "Дарья", "Диана", "Евгения", "Екатерина", "Елена", "Елизавета",
    "Жанна", "Инна", "Ирина", "Карина", "Кристина", "Ксения", "Лариса",
    "Лидия", "Любовь", "Людмила", "Маргарита", "Марина", "Мария", "Милана",
    "Надежда", "Наталья", "Наталия", "Нина", "Оксана", "Ольга", "Полина",
    "Светлана", "Софья", "София", "Тамара", "Татьяна", "Ульяна", "Эльвира",
    "Юлия", "Яна",
]

FIRST_NAMES = MALE_NAMES + FEMALE_NAMES

# Стафф аэродрома: выделяются на странице другим цветом. Заодно это
# словарь фамилий: распознанное подгоняется к нему, поэтому у стаффа
# фамилии всегда выходят без опечаток.
STAFF = [
    "Башмаков Иван", "Воробьев Вячеслав", "Калинин Александр",
    "Ковган Сергей", "Костюков Роман", "Моисеенко Алексей",
    "Памятных Ринат", "Панченко Станислав", "Пахомов Антон",
    "Пометелин Дмитрий", "Сорока Виктор", "Толстов Анатолий",
    "Трофимов Сергей", "Тырышкин Ярослав", "Шнуров Максим",
    "Ерофеев Алексей", "Капралов Евгений", "Колпаков Дмитрий",
    "Мышкевич Сергей", "Павлов Александр", "Титков Сергей",
    "Трофимова Ирина", "Ярмонов Сергей",
]


def match_staff(name):
    """Сверяет "Фамилия Имя" со списком стаффа с допуском на ошибки OCR.

    Имя к этому моменту уже подогнано к словарю, поэтому требуем его
    точного совпадения, а фамилию сравниваем фуззи (порог 0.7 пропускает
    "Башызков" -> "Башмаков", но не склеивает разных людей с одинаковым
    именем). Возвращает (каноническое имя, True) или (как было, False).
    """
    parts = (name or "").split()
    if len(parts) != 2:
        return name, False
    surname, first = parts
    best, best_r = None, 0.0
    for s in STAFF:
        s_sur, s_first = s.split()
        if s_first != first:
            continue
        r = difflib.SequenceMatcher(None, surname.lower(), s_sur.lower()).ratio()
        if r > best_r:
            best, best_r = s, r
    if best and best_r >= 0.7:
        return best, True
    return name, False


def guess_gender(surname, patronymic=""):
    """Пол по отчеству (надёжнее), иначе по окончанию фамилии.

    None - не определить (Сорока, Памятных без отчества).
    """
    p = (patronymic or "").lower().rstrip(".…")
    if p.endswith(("ич", "ыч")) or (len(p) > 5 and p.endswith("ч")):
        return "m"
    if p.endswith(("вна", "чна", "шна")):
        return "f"
    s = (surname or "").lower()
    if s.endswith(("ова", "ева", "ёва", "ина", "ына", "ая", "яя", "ская")):
        return "f"
    if s.endswith(("ов", "ев", "ёв", "ин", "ын", "ий", "ый", "ский")):
        return "m"
    return None


def fix_first_name(token, surname, patronymic=""):
    """Подгоняет имя к словарю с учётом пола (по отчеству или фамилии).

    Ловит случаи вида "Гольцов Евгения": распозналось женское имя, но
    человек мужского пола - предпочитаем близкий мужской вариант (Евгений).
    Или "Памятных Нинат Сергеевич": фамилия пол не выдаёт, а отчество - да,
    и вместо ближайшей "Нины" побеждает "Ринат".
    """
    t = (token or "").strip()
    if not t or "." in t or "…" in t or len(t) < 3:
        return token
    cands = difflib.get_close_matches(t, FIRST_NAMES, n=3, cutoff=0.66)
    if not cands:
        return token
    g = guess_gender(surname, patronymic)
    if g:
        pool = MALE_NAMES if g == "m" else FEMALE_NAMES
        best_r = difflib.SequenceMatcher(None, t, cands[0]).ratio()
        for c in cands:
            if c in pool:
                r = difflib.SequenceMatcher(None, t, c).ratio()
                if r >= best_r - 0.15:
                    return c
    return cands[0]

# "AFF" при распознавании кириллицей превращается в "АРЕ", "ВЕЕ" и т.п.:
# буква F не входит в алфавит и заменяется похожей по начертанию.
# "АФФ" на табло - легальная отдельная пометка, Ф в набор не входит.
AFF_A = "AАДВ"
AFF_F = "FРЕГТ"


def looks_like_aff(tok):
    t = (tok or "").upper()
    return (len(t) == 3 and t[0] in AFF_A
            and t[1] in AFF_F and t[2] in AFF_F)


def normalize_name(name):
    """Подгоняет распознанную строку к каноническому виду.

    - "КВОРУМ DZ [n]": латинское "DZ" с -l rus превращается в "02"/"ОХ",
      а "К" - в "И"; цифра номера может склеиться с "DZ" ("021" = "DZ 1").
    - "(Вып.)" - служебная пометка выполненного взлёта.
    - Обычная строка: оставляем только фамилию и имя. Отчество и пометку
      категории после ФИО (AFF, АФФ, A-D...) на страницу не выводим.
      Имя подгоняется к словарю.
    """
    t = (name or "").strip()
    if not t:
        return t
    parts = t.split()

    ratio = difflib.SequenceMatcher(None, parts[0].upper(), "КВОРУМ").ratio()
    if ratio >= 0.7:
        rest = "".join(parts[1:])
        # отрезаем само "DZ" в любом прочтении, остаток - номер (бывает 10+)
        rest = re.sub(r"^[DOОdoо0][Zz2Хх7]", "", rest)
        m = re.search(r"([1-9][0-9]?)$", rest)
        return "КВОРУМ DZ" + (" " + m.group(1) if m else "")

    core = re.sub(r"[^\w]", "", t, flags=re.UNICODE).lower()
    if difflib.SequenceMatcher(None, core, "вып").ratio() >= 0.7:
        return "(Вып.)"

    if len(parts) >= 2:
        patronymic = parts[2] if len(parts) >= 3 else ""
        parts[1] = fix_first_name(parts[1], parts[0], patronymic)
    return " ".join(parts[:2])


# восстановление цифр, прочитанных как буквы: "АРЕТ" - это "AFF 7"
DIGIT_FIX = {"О": "0", "о": "0", "O": "0", "o": "0", "З": "3", "з": "3",
             "Б": "6", "б": "6", "В": "8", "в": "8", "Т": "7",
             "І": "1", "I": "1", "l": "1", "S": "5", "s": "5"}


def normalize_category(text):
    """Приводит распознанную категорию к каноническому виду.

    AFF-категории разбираются по буквам: цифры восстанавливаются из похожих
    букв, а не подгоняются фуззи-матчем - иначе "AFF ТГ" превращался
    в "AFF 8" вместо "AFF 7". Дефисные уровни всегда начинаются с 8
    (бывают 8-1 и 8-2), а первая цифра дефисного часто читается криво.

    Остальное сжимается до канонического вида (без пробелов, латиница
    приведена к кириллице) и подгоняется к списку известных - так
    "АК TMS 30" находит "ХК ТМ3 90", а "ХК TM4 90" не гнётся в "ТМ4000 90".
    """
    t = (text or "").strip()
    if not t:
        return ""
    if "?" in t:
        return "???"
    parts = t.split()
    first = parts[0]
    aff = None
    if looks_like_aff(first):
        aff = "".join(parts[1:])
    elif len(first) > 3 and looks_like_aff(first[:3]):
        # цифра прилипла к буквам: "АРЕТ" = "AFF 7"
        aff = first[3:] + "".join(parts[1:])
    if aff is not None:
        digits = "".join(DIGIT_FIX.get(c, c) for c in aff)
        digits = "".join(c for c in digits if c.isdigit() or c == "-")
        if "-" in digits:
            digits = "8-" + digits.split("-")[-1]
        return "AFF " + digits if digits else "AFF"
    c = canon_cat(t)
    if c in CAT_CANON:
        return CAT_CANON[c]
    # коротким обрывкам ("ТО90") нужен строгий порог, иначе они цепляются
    # к первой попавшейся категории
    if len(c) >= 2:
        cutoff = 0.55 if len(c) >= 6 else 0.75
        m = difflib.get_close_matches(c, list(CAT_CANON), n=1, cutoff=cutoff)
        if m:
            return CAT_CANON[m[0]]
    return t


def ocr_category(img):
    """Распознаёт ячейку категории четырьмя способами, возвращает сырые.

    Разные конфигурации ошибаются по-разному: psm 8 читает целиком там,
    где psm 7 возвращает пустоту ("Ты 3000 90"), чистый rus не
    галлюцинирует латиницей на "Спортивный", а сглаженные варианты
    надёжнее на мелких цифрах. Состав ансамбля подобран замером на живых
    снимках (47/49 против 36/49 у одиночного прогона). Итог выбирается
    голосованием в pick_category.
    """
    if img.width < 3 or img.height < 3:
        return []
    variants = [
        (prep(img), TESS_MIX8),
        (prep(img, smooth=True), TESS_MIX),
        (prep(img), TESS_RU),
        (prep(img, scale=6, smooth=True), TESS_MIX),
    ]
    raws = []
    for p, cfg in variants:
        try:
            raws.append(pytesseract.image_to_string(p, config=cfg).strip())
        except Exception as e:
            sys.stderr.write("ocr error: %s\n" % e)
            raws.append("")
    if not any(raws):
        try:
            raws.append(pytesseract.image_to_string(
                prep(img, hline_thr=0.4), config=TESS_MIX8).strip())
        except Exception:
            pass
    return raws


def pick_category(raws, is_service):
    """Выбирает категорию голосованием нескольких прогонов распознавания.

    Сначала побеждает категория из списка известных, набравшая больше
    голосов (при равенстве - от более надёжной конфигурации). Потом
    "???", AFF-образные, служебные строки, и только затем сырой текст.
    """
    norms = [normalize_category(r) for r in raws]
    votes = [n for n in norms if n in KNOWN_CATS]
    if votes:
        best, best_c = None, 0
        for n in votes:
            c = votes.count(n)
            if c > best_c:
                best, best_c = n, c
        return best
    for src in (norms, raws):
        if any("?" in (x or "") for x in src):
            return "???"
    for n in norms:
        if n.startswith("AFF"):
            return n
    # у служебных строк на табло всегда "???"
    if is_service:
        return "???"
    for n in norms:
        if n:
            return n
    return ""


def not_background(arr):
    """Маска "это не фон табло".

    Фон берём из нижней части экрана - там всегда пустое оранжевое поле.
    По всей картинке так делать нельзя: чёрных пикселей в панелях больше,
    чем оранжевых, и фоном ошибочно окажется чёрный.
    """
    h = arr.shape[0]
    sample = arr[int(h * 0.60):int(h * 0.90), :, :]
    flat = sample.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    bg = colors[counts.argmax()]
    return np.abs(arr - bg).sum(axis=2) > 90


def find_table(arr):
    """Находит панели по горизонтали и границы таблицы по вертикали.

    Окно RMS может съезжать, поэтому ничего не задаём числами - ищем каждый раз.
    Возвращает (список панелей, верх, низ) или (None, None, None).
    """
    mask = not_background(arr)
    h, w = mask.shape

    # 1. Панели по столбцам: у панели почти вся высота столбца - не фон
    colsum = mask.sum(axis=0)
    thr_col = h * 0.35
    panels, inside, s = [], False, 0
    for x in range(w):
        if colsum[x] > thr_col and not inside:
            inside, s = True, x
        elif colsum[x] <= thr_col and inside:
            inside = False
            if x - s > 100:
                panels.append((s, x))
    if inside and w - s > 100:
        panels.append((s, w))
    if not panels:
        return None, None, None

    # 2. Верх и низ меряем по самой панели: её строки заполнены целиком
    x0, x1 = panels[0]
    inner = mask[:, x0 + 3:x1 - 3]
    need = inner.shape[1] * 0.9
    rows = [y for y in range(h) if inner[y].sum() >= need]
    if not rows:
        return panels, None, None

    groups, start = [], None
    prev = None
    for y in rows:
        if start is None:
            start = y
        elif y - prev > 2:
            groups.append((start, prev + 1)); start = y
        prev = y
    groups.append((start, prev + 1))
    top, bottom = max(groups, key=lambda g: g[1] - g[0])
    return panels, top, bottom


def fit_grid(arr, x0, x1, top, bottom, default=17.85):
    """Подгоняет сетку строк по линиям-разделителям.

    Разделители идут ровно по границам строк, но часть из них теряется
    (там, где строка подсвечена). Поэтому не меряем соседние промежутки,
    а подгоняем прямую по всем найденным линиям сразу - так не копится
    ошибка к нижним строкам.

    Возвращает (начало первой строки, шаг, число строк).
    """
    a = arr[:, x0 + 3:x1 - 3]
    w = a.shape[1]
    bright = (a.sum(axis=2) > 330).sum(axis=1)
    seps = [y for y in range(top, bottom) if bright[y] > w * 0.7]

    clean, prev = [], None
    for y in seps:
        if prev is None or y - prev > 3:
            clean.append(y)
        prev = y

    pitch = default
    origin = float(top)
    if len(clean) >= 4:
        s = np.array(clean, dtype=float)
        k = np.round((s - s[0]) / default)
        kk = k - k.mean()
        denom = (kk ** 2).sum()
        if denom > 0:
            p = float((kk * (s - s.mean())).sum() / denom)
            if 15 < p < 21:
                pitch = p
                intercept = float(s.mean() - pitch * k.mean())
                # сдвигаем сетку к верху панели
                k_top = round((top - intercept) / pitch)
                origin = intercept + pitch * k_top
                if origin > top + pitch * 0.5:
                    origin -= pitch
                if origin < top - pitch * 0.5:
                    origin += pitch

    n = int(round((bottom - origin) / pitch))
    n = max(18, min(30, n))
    return origin, pitch, n


def classify_row(cell):
    """Определяет тип строки: занята / свободна (зелёная) / пустая."""
    a = np.array(cell).astype(int)
    if a.size == 0:
        return "empty"
    green = ((a[:, :, 1] > 190) & (a[:, :, 0] < 210) & (a[:, :, 2] < 210)).mean()
    white = (a.sum(axis=2) > 600).mean()
    if green > 0.30:
        return "free"
    if white > 0.005:
        return "filled"
    return "empty"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "board.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else "board.json"
    dbg = sys.argv[3] if len(sys.argv) > 3 else None

    im = Image.open(src).convert("RGB")
    arr = np.array(im).astype(int)
    W, H = im.size

    result = {
        "updated": None,
        "source_size": [W, H],
        "clock": "",
        "work_until": "",
        "message": "",
        "loads": [],
        "status": "ok",
    }

    from datetime import datetime
    result["updated"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    panels, top, bottom = find_table(arr)

    # Если панелей нет - скорее всего RMS показывает заглушку "Подключение..."
    if not panels or top is None:
        result["status"] = "no_board"
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print("панели не найдены - похоже, табло сейчас не показывается")
        return

    origin, pitch, n_rows = fit_grid(arr, panels[0][0], panels[0][1], top, bottom)
    result["grid"] = {"top": top, "bottom": bottom, "origin": round(origin, 2),
                      "pitch": round(pitch, 3), "rows": n_rows,
                      "panels": [list(p) for p in panels]}

    # шапка находится над таблицей, её положение тоже плавает вместе с окном
    hy0 = max(0, top - 58)
    hy1 = max(1, top - 26)
    result["clock"] = ocr(im.crop((0, hy0, 420, hy1)), cfg=TESS_MIX)
    result["work_until"] = ocr(im.crop((max(0, W - 420), hy0, W, hy1)), cfg=TESS_MIX)

    # ---- сообщение внизу (под таблицей) ----
    msg_crop = im.crop((0, min(H - 1, bottom + 90), W, min(H, bottom + 190)))
    msg_arr = np.array(msg_crop).astype(int)
    if (msg_arr.sum(axis=2) < 200).mean() > 0.002:
        result["message"] = ocr(msg_crop, cfg=TESS_RU, invert=False, scale=2)

    draw = ImageDraw.Draw(im) if dbg else None

    # ---- панели ----
    for pi, (x0, x1) in enumerate(panels):
        pw = x1 - x0
        title = ocr(im.crop((max(0, x0 - 10), max(0, top - 26), x1 + 10, top - 3)),
                    cfg=TESS_MIX)

        load = {"index": pi + 1, "title": title, "capacity": n_rows,
                "rows": [], "free_from": None}

        for r in range(n_rows):
            ry0 = int(round(origin + r * pitch))
            ry1 = int(round(origin + (r + 1) * pitch))
            # для определения типа строки отступаем от рамок панели
            cell = im.crop((x0 + 4, ry0 + 2, x1 - 4, ry1 - 1))
            kind = classify_row(cell)

            # для распознавания отступаем от границ строки:
            # там проходят линии-разделители, они мешают распознаванию
            ty0, ty1 = ry0 + 1, ry1 - 1

            if draw:
                color = {"filled": (255, 0, 0), "free": (0, 255, 0),
                         "empty": (80, 80, 80)}[kind]
                draw.rectangle([x0, ry0, x1 - 1, ry1 - 1], outline=color)

            if kind == "free" and load["free_from"] is None:
                load["free_from"] = r + 1
            if kind != "filled":
                continue

            # границы колонок задаём долей ширины панели - она чуть плавает
            nx0 = x0 + int(pw * COL_NAME[0])
            nx1 = x0 + int(pw * COL_NAME[1])
            cx0 = x0 + int(pw * COL_CAT[0])
            cx1 = x0 + int(pw * COL_CAT[1])

            name_cell = im.crop((nx0, ty0, nx1, ty1))
            name = normalize_name(ocr_best(name_cell).replace("|", "").strip())
            is_service = name.startswith("КВОРУМ") or name == "(Вып.)"
            is_staff = False
            if not is_service:
                name, is_staff = match_staff(name)

            cat_cell = im.crop((cx0, ty0, cx1, ty1))
            cat_raws = ocr_category(cat_cell)
            cat = pick_category(cat_raws, is_service)
            cat_raw = next((r for r in cat_raws if r), "")

            # копим образцы того, что видит распознаватель
            if len(SAMPLES) < 12:
                SAMPLES.append((name_cell.copy(), prep(name_cell)))
            load["rows"].append({
                "n": r + 1,
                "name": name,
                "staff": is_staff,
                "cat": cat,
                "cat_raw": cat_raw,
            })

        if load["rows"] or load["free_from"]:
            result["loads"].append(load)

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    if dbg:
        im.save(dbg)
        # отдельная картинка: слева исходная ячейка (увеличенная),
        # справа то, что реально уходит в распознаватель
        if SAMPLES:
            pad = 6
            wid = max(max(a.width * 4, b.width) for a, b in SAMPLES)
            hei = sum(a.height * 4 + b.height + pad * 3 for a, b in SAMPLES)
            sheet = Image.new("L", (wid, hei), 160)
            y = 0
            for raw, ready in SAMPLES:
                big = raw.convert("L").resize((raw.width * 4, raw.height * 4), Image.NEAREST)
                sheet.paste(big, (0, y)); y += big.height + pad
                sheet.paste(ready, (0, y)); y += ready.height + pad * 2
            sheet.save(os.path.join(os.path.dirname(dbg) or ".", "cells.png"))

    total = sum(len(l["rows"]) for l in result["loads"])
    print("панелей: %d, распознано строк: %d" % (len(panels), total))
    print("сообщение: %r" % result["message"])


if __name__ == "__main__":
    main()

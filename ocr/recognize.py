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

# известные категории - результат распознавания подгоняется к ближайшей
KNOWN_CATS = [
    "Спортивный", "ФВ",
    "AFF 1", "AFF 2", "AFF 3", "AFF 4", "AFF 5", "AFF 6", "AFF 7", "AFF 8",
    "TM4000 90", "TM4000 100", "TM4000 120",
]

TESS_RU = "--oem 1 --psm 7 -l rus"
TESS_MIX = "--oem 1 --psm 7 -l rus+eng"

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
    Tesseract путает похожие буквы (е-а, и-ы). Сглаженный вариант иногда
    читается лучше, иногда хуже - решает уверенность самого Tesseract.
    """
    if img.width < 3 or img.height < 3:
        return ""
    best_text, best_conf = "", -1.0
    for smooth in (False, True):
        text, conf = ocr_with_conf(prep(img, smooth=smooth), cfg)
        if conf > best_conf and text:
            best_text, best_conf = text, conf
    if not best_text:
        best_text = ocr(img, cfg=cfg)
    return best_text


# Имена и отчества - закрытые словари, распознанное подгоняется к ближайшему.
# Фамилии не трогаем: их список открытый, а уведомления всё равно сверяют
# фамилию с допуском на ошибки распознавания.
FIRST_NAMES = [
    "Александр", "Алексей", "Анатолий", "Андрей", "Антон", "Аркадий",
    "Арсений", "Артём", "Артем", "Борис", "Вадим", "Валентин", "Валерий",
    "Василий", "Виктор", "Виталий", "Владимир", "Владислав", "Вячеслав",
    "Геннадий", "Георгий", "Глеб", "Григорий", "Даниил", "Данил", "Денис",
    "Дмитрий", "Евгений", "Егор", "Иван", "Игорь", "Илья", "Кирилл",
    "Константин", "Лев", "Леонид", "Максим", "Марк", "Матвей", "Михаил",
    "Никита", "Николай", "Олег", "Павел", "Пётр", "Петр", "Роман", "Руслан",
    "Семён", "Семен", "Сергей", "Станислав", "Степан", "Тимофей", "Тимур",
    "Фёдор", "Федор", "Эдуард", "Юрий", "Ярослав",
    "Александра", "Алина", "Алла", "Анастасия", "Анна", "Валентина",
    "Валерия", "Вера", "Вероника", "Виктория", "Галина", "Дарья", "Диана",
    "Евгения", "Екатерина", "Елена", "Елизавета", "Жанна", "Инна", "Ирина",
    "Карина", "Кристина", "Ксения", "Лариса", "Лидия", "Любовь", "Людмила",
    "Маргарита", "Марина", "Мария", "Надежда", "Наталья", "Наталия", "Нина",
    "Оксана", "Ольга", "Полина", "Светлана", "Софья", "София", "Татьяна",
    "Юлия", "Яна",
]

PATRONYMICS = [
    "Александрович", "Алексеевич", "Анатольевич", "Андреевич", "Антонович",
    "Аркадьевич", "Арсеньевич", "Артёмович", "Борисович", "Вадимович",
    "Валентинович", "Валерьевич", "Васильевич", "Викторович", "Витальевич",
    "Владимирович", "Владиславович", "Вячеславович", "Геннадьевич",
    "Георгиевич", "Глебович", "Григорьевич", "Даниилович", "Данилович",
    "Денисович", "Дмитриевич", "Евгеньевич", "Егорович", "Иванович",
    "Игоревич", "Ильич", "Кириллович", "Константинович", "Львович",
    "Леонидович", "Максимович", "Маркович", "Матвеевич", "Михайлович",
    "Никитич", "Никитович", "Николаевич", "Олегович", "Павлович",
    "Петрович", "Романович", "Русланович", "Семёнович", "Семенович",
    "Сергеевич", "Станиславович", "Степанович", "Тимофеевич", "Тимурович",
    "Фёдорович", "Федорович", "Эдуардович", "Юрьевич", "Ярославович",
    "Александровна", "Алексеевна", "Анатольевна", "Андреевна", "Антоновна",
    "Борисовна", "Валентиновна", "Валерьевна", "Васильевна", "Викторовна",
    "Витальевна", "Владимировна", "Владиславовна", "Вячеславовна",
    "Геннадьевна", "Георгиевна", "Григорьевна", "Денисовна", "Дмитриевна",
    "Евгеньевна", "Ивановна", "Игоревна", "Ильинична", "Кирилловна",
    "Константиновна", "Леонидовна", "Львовна", "Максимовна", "Михайловна",
    "Николаевна", "Олеговна", "Павловна", "Петровна", "Романовна",
    "Сергеевна", "Станиславовна", "Степановна", "Фёдоровна", "Федоровна",
    "Эдуардовна", "Юрьевна", "Ярославовна",
]


def fix_token(token, vocab, cutoff=0.66):
    """Подгоняет слово к ближайшему из словаря.

    Обрезанные многоточием слова ("Александро...") не трогаем - на экране
    нет их полного варианта, и подгонять не к чему.
    """
    t = (token or "").strip()
    if not t or "." in t or "…" in t or len(t) < 3:
        return token
    m = difflib.get_close_matches(t, vocab, n=1, cutoff=cutoff)
    return m[0] if m else token


# "AFF" при распознавании кириллицей превращается в "АРЕ", "АГТ" и т.п.:
# буква F не входит в алфавит и заменяется похожей по начертанию.
# "АФФ" на табло - легальная отдельная пометка, Ф в набор не входит.
AFF_A = "AАД"
AFF_F = "FРЕГТ"


def looks_like_aff(tok):
    t = (tok or "").upper()
    return (len(t) == 3 and t[0] in AFF_A
            and t[1] in AFF_F and t[2] in AFF_F)


def normalize_name(name):
    """Подгоняет распознанную строку к каноническому виду.

    - "КВОРУМ DZ [n]": латинское "DZ" с -l rus превращается в "02"/"ОХ",
      а "К" - в "И". Первое слово похоже на КВОРУМ - строка переписывается.
    - "(Вып.)" - служебная пометка выполненного взлёта.
    - Имя и отчество подгоняются к словарям.
    - Хвост-пометка после ФИО: "АРЕ" - это AFF, одиночная "О"/"Р" - это D.
    """
    t = (name or "").strip()
    if not t:
        return t
    parts = t.split()

    ratio = difflib.SequenceMatcher(None, parts[0].upper(), "КВОРУМ").ratio()
    if ratio >= 0.7:
        num = ""
        rest = parts[1:]
        if rest and re.fullmatch(r"[1-9]", rest[-1]):
            num = " " + rest[-1]
        return "КВОРУМ DZ" + num

    core = re.sub(r"[^\w]", "", t, flags=re.UNICODE).lower()
    if difflib.SequenceMatcher(None, core, "вып").ratio() >= 0.7:
        return "(Вып.)"

    if len(parts) >= 2:
        parts[1] = fix_token(parts[1], FIRST_NAMES)
    if len(parts) >= 3:
        parts[2] = fix_token(parts[2], PATRONYMICS)

    if len(parts) >= 4:
        last = parts[-1]
        if looks_like_aff(last) and last.upper() != "AFF":
            parts[-1] = "AFF"
        elif last in ("О", "O", "0", "Р", "P"):
            parts[-1] = "D"
    return " ".join(parts)


# восстановление цифр, прочитанных как буквы: "АРЕТ" - это "AFF 7"
DIGIT_FIX = {"О": "0", "о": "0", "O": "0", "o": "0", "З": "3", "з": "3",
             "Б": "6", "б": "6", "В": "8", "в": "8", "Т": "7",
             "І": "1", "I": "1", "l": "1", "S": "5", "s": "5"}


def normalize_category(text):
    """Приводит распознанную категорию к каноническому виду.

    AFF-категории разбираются по буквам: цифры восстанавливаются из похожих
    букв, а не подгоняются фуззи-матчем - иначе "AFF ТГ" превращался
    в "AFF 8" вместо "AFF 7". Остальное подгоняется к списку известных.
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
        return "AFF " + digits if digits else "AFF"
    m = difflib.get_close_matches(t, KNOWN_CATS, n=1, cutoff=0.6)
    return m[0] if m else t


def pick_category(raw_rus, raw_mix, is_service):
    """Выбирает категорию из двух прогонов распознавания.

    rus+eng на кириллических категориях иногда галлюцинирует латиницей
    ("COMBA" вместо "Спортивный"), поэтому категория распознаётся дважды:
    чистым rus и rus+eng. Побеждает результат, совпавший со списком
    известных категорий, затем AFF-образный (у rus+eng цифры надёжнее).
    """
    for raw in (raw_rus, raw_mix):
        if "?" in (raw or ""):
            return "???"
    n_rus = normalize_category(raw_rus)
    n_mix = normalize_category(raw_mix)
    for n in (n_rus, n_mix):
        if n in KNOWN_CATS:
            return n
    for n in (n_mix, n_rus):
        if n.startswith("AFF"):
            return n
    # у служебных строк на табло всегда "???"
    if is_service:
        return "???"
    return n_mix or n_rus


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

            cat_cell = im.crop((cx0, ty0, cx1, ty1))
            cat_raw_rus = ocr(cat_cell, cfg=TESS_RU)
            cat_raw_mix = ocr(cat_cell, cfg=TESS_MIX)
            cat = pick_category(cat_raw_rus, cat_raw_mix, is_service)

            # копим образцы того, что видит распознаватель
            if len(SAMPLES) < 12:
                SAMPLES.append((name_cell.copy(), prep(name_cell)))
            load["rows"].append({
                "n": r + 1,
                "name": name,
                "cat": cat,
                "cat_raw": cat_raw_mix or cat_raw_rus,
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

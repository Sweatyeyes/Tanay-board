#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Распознаёт табло со скриншота и складывает результат в board.json.
Запускается на GitHub Actions (Ubuntu), поэтому кириллица здесь безопасна.

Использование:
    python recognize.py board.png out/board.json out/debug.png
"""

import hashlib
import json
import os
import re
import sys
import time
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
    "Спортивный", "ФВ", "Совершенствование", "RW", "CP",
    # групповые прыжки: на табло пишут "SPL 7-way" и "SPL Tanay".
    # Без них строки уходили в сырой текст и один и тот же прыжок
    # показывался то как "SPL T-way", то как "SPL нау:".
    "SPL Tanay",
    "AFF 1", "AFF 2", "AFF 3", "AFF 4", "AFF 5", "AFF 6", "AFF 7",
    "AFF 8-1", "AFF 8-2",
]
# тандемы: ТМ3000/ТМ4000 и «хлопковые» ХК, вес 90-120
for _mark in ("ТМ3000", "ТМ4000"):
    for _w in ("90", "100", "110", "120"):
        KNOWN_CATS.append("%s %s" % (_mark, _w))
for _mark in ("ХК ТМ3", "ХК ТМ4"):
    for _w in ("90", "100", "110", "120"):
        KNOWN_CATS.append("%s %s" % (_mark, _w))
# групповая акробатика: размер группы бывает разный
for _n in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20):
    KNOWN_CATS.append("SPL %d-way" % _n)

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

# Борта аэродрома и число мест. Вместимость всё равно меряется по табло
# (сколько строк закрашено зелёным), таблица нужна как подстраховка, когда
# борт только что отправлен и зелёных строк уже нет.
# Ан-2 - уточнить: бывает 10 или 12, пока берём по табло.
AIRCRAFT = [
    ("Л-410", ["Л-410", "Л410", "L-410", "L410"], 18),
    ("Ан-2",  ["Ан-2", "Ан2", "AH-2", "AN-2"],    None),
    ("Ми-2",  ["Ми-2", "Ми2", "MU-2", "MN-2"],    7),
    ("Ми-8",  ["Ми-8", "Ми8", "MU-8", "MN-8"],    18),
]

TESS_RU = "--oem 1 --psm 7 -l rus"
TESS_MIX = "--oem 1 --psm 7 -l rus+eng"
TESS_MIX8 = "--oem 1 --psm 8 -l rus+eng"
# Отдельный проход только латиницей - для иностранных фамилий.
# Смешанный rus+eng тут не годится: модель всё равно тянет в кириллицу.
TESS_EN = "--oem 1 --psm 7 -l eng"

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
        return "", -1.0
    best_text, best_conf = "", -1.0
    for scale in (6, 4):
        text, conf = ocr_with_conf(prep(img, scale=scale, smooth=True), cfg)
        if conf > best_conf and text:
            best_text, best_conf = text, conf
        # уверенное чтение - второй прогон не нужен
        if best_conf >= 88:
            break
    if not best_text:
        best_text = ocr(img, cfg=cfg)
    return best_text, best_conf


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


VOWELS = "аеёиоуыэюя"


def fix_surname(s, first_name=""):
    """Чинит типовые ошибки распознавания в фамилии без словаря.

    Опирается на устройство русского языка, а не на список фамилий:
    - окончаний "-ое", "-еа", "-зв", "-ес" у фамилий не бывает - это
      битые "-ов"/"-ев" (Кривощекое, Астафьеа, Соловьзв, Тукачес);
    - "м" между двумя согласными не встречается - это буква "и",
      прочитанная как "м" (Вавмлов, Савмна, Кокормн);
    - "жс", "шы", "жы", "ии" в середине слова тоже артефакты (Ажсенов,
      Хорошылова, Шиикевич);
    - женское окончание фамилии при мужском имени - лишняя "а" на конце
      (Гордова Владислав).
    """
    if len(s) < 5 or "." in s or "…" in s:
        return s
    t = s
    for bad, good in (("евыч", "евич"), ("овыч", "ович"), ("фьга", "фьев"),
                      ("ьза", "ьев"), ("ое", "ов"), ("ее", "ев"),
                      ("еа", "ев"), ("зв", "ев"), ("ес", "ев")):
        if t.endswith(bad):
            t = t[:-len(bad)] + good
            break
    t = t.replace("жс", "кс").replace("пст", "лст")
    t = t.replace("шы", "ши").replace("жы", "жи")
    i = t.find("ии", 1)
    if i != -1 and i + 2 < len(t):
        t = t[:i + 1] + "н" + t[i + 2:]
    chars = list(t)
    for i in range(1, len(chars) - 1):
        if chars[i] == "м":
            l, r = chars[i - 1].lower(), chars[i + 1].lower()
            if l not in VOWELS + "ьъ" and r not in VOWELS + "ьъ":
                chars[i] = "и"
    t = "".join(chars)
    if first_name in MALE_NAMES and re.search(r"(ов|ев|ёв|ин|ын)а$", t):
        t = t[:-1]
    return t


def reconcile_names(loads, history=None):
    """Сверяет дубли и память прошлых прогонов.

    Один человек часто записан в несколько взлётов, а окно RMS дрейфует,
    поэтому ошибки распознавания в разных строках и в разных прогонах
    разные - правильное же написание всегда одно. Побеждает вариант
    с наибольшим весом: вхождения в текущем кадре весят по 2, накопленные
    прошлыми прогонами (history) - по 1 (с потолком 20, чтобы память
    не становилась неисправимой), совпадение со стаффом - решает сразу.
    При равенстве выбирается распознанный с большей уверенностью.
    """
    history = history or {}
    entries = []
    for li, l in enumerate(loads):
        for ri, r in enumerate(l["rows"]):
            n = r["name"]
            if not n or n.startswith("КВОРУМ") or n == "(Вып.)":
                continue
            parts = n.split()
            if len(parts) != 2:
                continue
            conf = 200.0 if r.get("staff") else float(r.get("_conf") or 0.0)
            entries.append([li, ri, parts[0], parts[1], conf])

    used = [False] * len(entries)
    for a in range(len(entries)):
        if used[a]:
            continue
        cluster = [entries[a]]
        used[a] = True
        for b in range(a + 1, len(entries)):
            if used[b] or entries[b][3] != entries[a][3]:
                continue
            r = difflib.SequenceMatcher(
                None, entries[b][2].lower(), entries[a][2].lower()).ratio()
            if r >= 0.75:
                cluster.append(entries[b])
                used[b] = True

        first = cluster[0][3]
        weights, confs = {}, {}
        for e in cluster:
            weights[e[2]] = weights.get(e[2], 0.0) + (1000.0 if e[4] >= 200 else 2.0)
            confs[e[2]] = max(confs.get(e[2], 0.0), e[4])
        for key, cnt in history.items():
            ksur, _, kfirst = key.partition(" ")
            if kfirst != first:
                continue
            r = difflib.SequenceMatcher(
                None, ksur.lower(), cluster[0][2].lower()).ratio()
            if r >= 0.75:
                weights[ksur] = weights.get(ksur, 0.0) + min(float(cnt), 20.0)
                confs.setdefault(ksur, 0.0)
        if len(weights) < 2:
            continue
        best = max(weights, key=lambda k: (weights[k], confs[k]))
        for e in cluster:
            if e[2] != best:
                loads[e[0]]["rows"][e[1]]["name"] = best + " " + e[3]


def update_history(history, loads):
    """Копит счётчик написаний между прогонами и потихоньку забывает.

    Затухание нужно, чтобы разовые ошибки и уехавшие люди со временем
    исчезали, а потолок - чтобы даже устойчивая ошибка статичного кадра
    не могла стать неисправимой.
    """
    for l in loads:
        for r in l["rows"]:
            n = r["name"]
            if not n or n.startswith("КВОРУМ") or n == "(Вып.)":
                continue
            if len(n.split()) == 2:
                history[n] = min(float(history.get(n, 0)) + 1.0, 50.0)
    out = {}
    for k, v in history.items():
        v = round(float(v) * 0.98, 2)
        if v >= 0.5:
            out[k] = v
    return out


def update_stats(path, entry, keep=400):
    """Копит замеры прогонов: частота смены кадра, задержки, время работы.

    Файл лежит в ветке ocr рядом с board.json и переживает прогоны.
    Хранится последние keep записей плюс сводка.
    """
    data = {"runs": []}
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            sys.stderr.write("stats load error: %s\n" % e)
    runs = data.get("runs", [])
    runs.append(entry)
    runs = runs[-keep:]

    # сводка: как часто реально меняется кадр и какова задержка
    changed = [r for r in runs if r.get("changed")]
    def gaps(items):
        ts = sorted(r["t"] for r in items)
        return [round(b - a, 1) for a, b in zip(ts, ts[1:]) if 0 < b - a < 3600]
    g_all, g_ch = gaps(runs), gaps(changed)
    def avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else None
    # Прогоны без табло на экране в замерах скорости не участвуют: там нечего
    # распознавать, и они занижали бы среднее. В счётчике прогонов и в
    # "обновлено" они есть - иначе сводка выглядит так, будто конвейер умер,
    # хотя он просто ждёт, когда табло вернётся на экран.
    work = [r for r in runs if r.get("board") is not False]
    lags = [r["lag"] for r in work if r.get("lag") is not None]
    durs = [r["sec"] for r in work if r.get("sec") is not None]

    data["runs"] = runs
    data["summary"] = {
        "прогонов": len(runs),
        "из них с новым кадром": len(changed),
        "интервал между прогонами, с": avg(g_all),
        "интервал между сменами кадра, с": avg(g_ch),
        "задержка снимок-распознавание, с": avg(lags),
        "время распознавания, с": avg(durs),
        "обновлено": entry.get("iso"),
    }
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    return data["summary"]


# "SPL 7-way" и "SPL Tanay" отличаются одним слогом, и общий нечёткий поиск
# их путает: "T-way" ближе к "Tanay", чем к "7-way". Поэтому разбираем хвост
# после SPL руками. Дефис есть только у групповых ("7-way", "T-way", "f-way"),
# а "Tanay" начинается с Т - этого хватает, чтобы развести все прочтения.
SPL_HEAD = re.compile(r"^\s*[S3$5ЅЗ]\s*[PР]\s*[LI1|]\s*(.*)$", re.I | re.S)


def normalize_spl(text):
    """Приводит строку вида "SPL ..." к канону. Не SPL - возвращает None."""
    m = SPL_HEAD.match(text or "")
    if not m:
        return None
    tail = m.group(1).strip()
    # цифра в хвосте - это размер группы и есть
    d = re.search(r"([1-9][0-9]?)", tail)
    if d and 1 <= int(d.group(1)) <= 20:
        return "SPL %s-way" % d.group(1)
    if "-" in tail:                      # "T-way", "f-way" - тоже группа
        return "SPL 7-way"
    # Tanay читается длинным словом на Т ("Тамау", "Tanay"), а короткие
    # "Тау", "Рау", "нау" - это искажённое "way"
    letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", tail)
    if len(letters) >= 4 and letters[0] in "TТtт":
        return "SPL Tanay"
    return "SPL 7-way"


def detect_aircraft(title):
    """Определяет борт по заголовку взлёта. Возвращает (имя, число мест).

    Точного вхождения мало: распознавание вставляет лишние цифры
    ("Л-4710" вместо "Л-410") и путает похожие знаки ("Л-4l0", "Л-41О").
    Поэтому после точной проверки идёт сравнение с допуском по каждому
    слову заголовка.
    """
    t = re.sub(r"[\s.]", "", (title or "").upper()).replace("O", "О")
    for name, variants, seats in AIRCRAFT:
        for v in variants:
            if re.sub(r"[\s.]", "", v.upper()) in t:
                return name, seats

    def canon(s):
        s = (s or "").upper().translate(LAT2CYR).translate(CYR2DIGIT)
        return re.sub(r"[^А-ЯЁ0-9]", "", s)

    best, best_r, best_seats = "", 0.0, None
    for word in re.split(r"[\s.]+", (title or "")):
        cw = canon(word)
        if len(cw) < 3:
            continue
        for name, variants, seats in AIRCRAFT:
            for v in variants:
                r = difflib.SequenceMatcher(None, cw, canon(v)).ratio()
                if r > best_r:
                    best, best_r, best_seats = name, r, seats
    return (best, best_seats) if best_r >= 0.7 else ("", None)


# Постоянная подпись внизу табло - это не сообщение диспетчера, а легенда.
# Без этой проверки она вылезала в приложении красной плашкой.
STATIC_MSG = re.compile(r"^\s*КАТЕГОРИ", re.I)

# Водяной знак Windows в правом нижнем углу экрана попадает в ту же полосу,
# где диспетчер пишет объявление. Читается он всегда криво ("AkTHeauwa
# Windows"), поэтому ловим не по точному тексту, а по похожести.
WATERMARK = ("активация windows", "activate windows",
             "чтобы активировать windows", "go to settings to activate")


def looks_watermark(text):
    t = re.sub(r"[^a-zа-яё ]", " ", (text or "").lower())
    t = " ".join(t.split())
    if not t:
        return False
    if "windows" in t or "windovs" in t:
        return True
    return any(difflib.SequenceMatcher(None, t, w).ratio() >= 0.6
               for w in WATERMARK)


def read_message(img):
    """Читает строку сообщения и русской, и латинской моделью.

    Диспетчер пишет и по-русски, и по-английски: "STANDBY UNTIL 16:30"
    под моделью rus превращалось в "ЭТАМОВУ ПИМТИЕ 16:30". Определить
    язык заранее нельзя, поэтому читаем обеими и берём ту, в которой
    Tesseract увереннее. Строка одна, лишний проход ничего не стоит.
    """
    best_text, best_conf = "", -1.0
    for cfg in (TESS_RU, TESS_EN):
        text, conf = ocr_with_conf(prep(img, scale=2, invert=False), cfg)
        if text and conf > best_conf:
            best_text, best_conf = text, conf
    return best_text


def clean_message(text):
    """Отсеивает статичную легенду и обрывки без смысла."""
    t = (text or "").strip()
    if not t or STATIC_MSG.match(t) or looks_watermark(t):
        return ""
    letters = sum(1 for c in t if c.isalpha())
    # у настоящего объявления букв заметно больше, чем мусорных знаков
    if letters < 6 or letters < len(t) * 0.6:
        return ""
    return t


def mostly_latin(text):
    """Строка написана латиницей, а не кириллицей."""
    lat = len(re.findall(r"[A-Za-z]", text or ""))
    cyr = len(re.findall(r"[А-Яа-яЁё]", text or ""))
    return lat >= 3 and lat > cyr


def looks_garbled(text):
    """Похоже, что кириллическая модель прочитала латинское имя.

    Признак: в слове больше одной заглавной буквы, но оно не целиком
    заглавными. У русской фамилии заглавная только первая; "Abdulbari"
    под моделью rus выходит то "АБЧиБай", то "АБЧИБай" - в обоих случаях
    заглавных несколько. Служебные строки целиком капсом ("КВОРУМ DZ")
    под это правило не попадают.
    """
    for w in (text or "").split():
        core = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", w)
        if len(core) < 4:
            continue
        upper = sum(1 for c in core if c.isupper())
        if upper >= 2 and upper < len(core):
            return True
    return False


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
    # мусорные символы внутри слов ("Сушко@") выбрасываем; точки и дефисы
    # оставляем - это обрезки и двойные фамилии
    parts = [re.sub(r"[^0-9A-Za-zА-Яа-яЁё.…-]", "", p) for p in t.split()]
    parts = [p for p in parts if p]
    if not parts:
        return ""

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

    # Латинское имя ("Abdulbari Qubaisi") к русским словарям не подгоняем:
    # там для него нет ни фамилий, ни имён, и любая подгонка только испортит.
    if mostly_latin(" ".join(parts)):
        return " ".join(parts[:2])

    if len(parts) >= 2:
        patronymic = parts[2] if len(parts) >= 3 else ""
        parts[1] = fix_first_name(parts[1], parts[0], patronymic)
        parts[0] = fix_surname(parts[0], parts[1])
    return " ".join(parts[:2])


# восстановление цифр, прочитанных как буквы: "АРЕТ" - это "AFF 7"
DIGIT_FIX = {"О": "0", "о": "0", "O": "0", "o": "0", "З": "3", "з": "3",
             "Б": "6", "б": "6", "В": "8", "в": "8", "Т": "7",
             "І": "1", "I": "1", "l": "1", "S": "5", "s": "5"}

# В числе готовности к тем же ошибкам добавляются палочки: семёрку на этом
# шрифте нередко читает как "/", единицу - как "|".
READY_DIGITS = dict(DIGIT_FIX)
READY_DIGITS.update({"/": "7", "\\": "7", "|": "1", "!": "1", "]": "1"})

READY_RE = re.compile(r"(готов\w*\s+)(\S{1,3})(\s*мин)", re.I)


# Номер взлёта в заголовке: "7 взлет Л-410". Иногда цифра читается как
# скобка или палочка ("{ взлет"), и тогда номер теряется целиком - а на нём
# держатся уведомления, книжка прыжков и память о свёрнутых списках.
LOAD_HEAD = re.compile(r"^(\s*)(\S{1,3})(\s*взл)", re.I)


def load_number(title):
    m = re.search(r"(\d+)\s*взл", title or "", re.I)
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 99 else None


def fix_load_numbers(loads):
    """Восстанавливает потерянный номер взлёта по соседним панелям.

    Панели на табло идут подряд слева направо, поэтому сосед задаёт номер
    надёжнее, чем угадывание по начертанию испорченного символа.
    """
    nums = [load_number(l.get("title")) for l in loads]
    known = [(i, n) for i, n in enumerate(nums) if n is not None]
    if not known or len(known) == len(nums):
        return
    for i, n in enumerate(nums):
        if n is not None:
            continue
        j, base = min(known, key=lambda kv: abs(kv[0] - i))
        guess = base + (i - j)
        if guess < 1 or guess > 99:
            continue
        t = loads[i].get("title") or ""
        new = LOAD_HEAD.sub(lambda m: m.group(1) + str(guess) + m.group(3), t, count=1)
        if new == t and re.match(r"\s*взл", t, re.I):
            new = "%d %s" % (guess, t.lstrip())     # номер пропал совсем
        if new != t:
            sys.stderr.write("панель %d: номер взлёта восстановлен по соседям: %r -> %r\n"
                             % (i + 1, t, new))
            loads[i]["title"] = new


def fix_ready_minutes(title):
    """Чинит число в "готовность N мин.".

    Одна испорченная цифра стоила дорого: из-за "готовность 2/ мин."
    приложение вообще переставало видеть готовность - разбор требовал
    подряд идущих цифр - и предупреждения за 15/10/5 минут не приходили.
    Правим только если после замены выходит правдоподобное число.
    """
    def sub(m):
        fixed = "".join(READY_DIGITS.get(c, c) for c in m.group(2))
        if fixed.isdigit() and 1 <= int(fixed) <= 99:
            return m.group(1) + fixed + m.group(3)
        return m.group(0)
    return READY_RE.sub(sub, title or "")


DEPARTED = "ОТПРАВЛЕН"


def fix_departed(title):
    """Восстанавливает пометку "ОТПРАВЛЕН!!!" в заголовке взлёта.

    Три восклицательных знака подряд сливаются в одну палку, и выходит
    то "ОТПРАВЛЕНИ!", то "ОТПРАВЛЕН111", то "ОТПРАВЛЕНШ". Само слово
    узнаём по началу и дописываем концовку целиком.

    Слово важное: по нему приложение понимает, что взлёт ушёл, - и
    подсвечивает его, и закрывает уведомления по этому борту.
    """
    out = []
    for w in (title or "").split():
        core = re.sub(r"[^А-ЯЁA-Z]", "", w.upper())
        if len(core) >= 8 and difflib.SequenceMatcher(
                None, core[:len(DEPARTED)], DEPARTED).ratio() >= 0.8:
            out.append("ОТПРАВЛЕН!!!")
        else:
            out.append(w)
    s = " ".join(out)
    # хвост мог отделиться в отдельное слово - убираем повтор
    return re.sub(r"(ОТПРАВЛЕН!!!)(\s*[!1lI|]+)+", r"\1", s)


def fix_work_until(text):
    """Чинит подпись в правом верхнем углу: "Работа до 20:14".

    Текст известен целиком, кроме времени, поэтому не подгоняем буквы,
    а собираем строку заново - но только если первое слово похоже на
    "Работа" и дальше действительно нашлось время.
    """
    t = (text or "").strip()
    m = re.search(r"(\d{1,2})[:.;](\d{2})", t)
    if not t or not m:
        return t
    words = re.sub(r"[^А-Яа-яЁё ]", " ", t[:m.start()]).split()
    if words and difflib.SequenceMatcher(
            None, words[0].lower(), "работа").ratio() >= 0.6:
        return "Работа до %s:%s" % (m.group(1), m.group(2))
    return t


def fix_takeoff_word(title):
    """Приводит к виду "взлет" слово, которое прочиталось криво.

    Оно короткое и стоит вторым в заголовке, поэтому ошибка в одной букве
    видна сразу: "2 аэлет АН-2", "4 взлёт". Правим только слова похожей
    длины - названия бортов и "готовность" под правило не попадают.
    """
    out = []
    for w in (title or "").split():
        core = re.sub(r"[^А-Яа-яЁё]", "", w).lower()
        if 4 <= len(core) <= 6 and difflib.SequenceMatcher(
                None, core, "взлет").ratio() >= 0.6:
            out.append(w.replace(core, "взлет") if core in w else "взлет")
        else:
            out.append(w)
    return " ".join(out)


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
    spl = normalize_spl(t)
    if spl:
        return spl
    c = canon_cat(t)
    if c in CAT_CANON:
        return CAT_CANON[c]
    # коротким обрывкам ("ТО90") нужен строгий порог, иначе они цепляются
    # к первой попавшейся категории
    if len(c) >= 2:
        cutoff = 0.55 if len(c) >= 6 else 0.75
        m = difflib.get_close_matches(c, list(CAT_CANON), n=1, cutoff=cutoff)
        if m:
            hit = CAT_CANON[m[0]]
            # у сильно испорченных "SPL" нечёткий поиск может выдумать размер
            # группы ("ЗРЕРмау" -> "SPL 9-way"). Цифру берём только если она
            # действительно есть в строке.
            if hit.startswith("SPL ") and not re.search(r"[1-9]", t):
                letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", t)[3:]
                return ("SPL Tanay" if len(letters) >= 4 and letters[0] in "TТtт"
                        else "SPL 7-way")
            return hit
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
    # готовим картинки лениво: часто хватает первых двух прогонов
    variants = [
        (lambda: prep(img), TESS_MIX8),
        (lambda: prep(img, smooth=True), TESS_MIX),
        (lambda: prep(img), TESS_RU),
        (lambda: prep(img, scale=6, smooth=True), TESS_MIX),
    ]
    raws = []
    for i, (mk, cfg) in enumerate(variants):
        try:
            raws.append(pytesseract.image_to_string(mk(), config=cfg).strip())
        except Exception as e:
            sys.stderr.write("ocr error: %s\n" % e)
            raws.append("")
        # ранний выход: два прогона сошлись на известной категории -
        # остальные ничего не изменят, а время экономят заметно
        if i >= 1:
            good = [normalize_category(x) for x in raws]
            good = [g for g in good if g in KNOWN_CATS or g == "???"]
            if len(good) >= 2 and good[0] == good[1]:
                break
    if not any(raws):
        try:
            raws.append(pytesseract.image_to_string(
                prep(img, hline_thr=0.4), config=TESS_MIX8).strip())
        except Exception:
            pass
    return raws


# короткие подписи для страницы: колонка узкая, длинные слова её распирают
SHORT_CATS = {
    "Спортивный": "Спорт",
    "Совершенствование": "Соверш.",
}


def short_category(cat):
    """Сокращает подпись для показа: ТМ4000 90 -> ТМ4 90, Спортивный -> Спорт.

    ФВ, RW, CP, AFF и "???" остаются как есть. У "ХК ТМ3 90" тысячи уже
    свёрнуты в исходнике, поэтому строка не меняется.
    """
    c = (cat or "").strip()
    if c in SHORT_CATS:
        return SHORT_CATS[c]
    # ТМ4000 90 -> ТМ4 90, в том числе с приставкой ХК
    m = re.match(r"^(ХК\s+)?ТМ(\d)000(\s+\d+)?$", c)
    if m:
        return "%sТМ%s%s" % (m.group(1) or "", m.group(2), m.group(3) or "")
    return c


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


def _common_color(block):
    flat = block.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    return colors[counts.argmax()]


def not_background(arr):
    """Маска "это не фон табло".

    Цвет фона берём с боковых полей - узких полосок вдоль левого и правого
    края экрана. Таблицы до края никогда не доходят, поэтому там оранжевый
    при любой раскладке табло.

    Раньше фон искали в нижней трети экрана. Пока таблиц было четыре, там
    было пустое оранжевое поле. Когда табло стало показывать восемь, туда
    попал второй ряд таблиц - и самым частым цветом мог оказаться чёрный.
    Тогда маска переворачивалась: фоном считались сами таблицы, и дальше
    не находилось ничего.

    Верх и низ картинки в расчёт не берём: там заголовок окна и панель
    задач Windows, они не имеют отношения к табло.
    """
    h, w = arr.shape[0], arr.shape[1]
    edge = max(3, int(w * 0.01))
    y0, y1 = int(h * 0.05), int(h * 0.88)
    ring = np.concatenate([
        arr[y0:y1, :edge, :].reshape(-1, 3),
        arr[y0:y1, w - edge:, :].reshape(-1, 3),
    ])
    bg = _common_color(ring)
    mask = np.abs(arr - bg).sum(axis=2) > 90

    # Подстраховка: если поля вдруг оказались нетипичными (другое
    # разрешение, окно не на весь экран), маска выйдет вырожденной -
    # почти всё фон или почти ничего. Тогда возвращаемся к старому способу.
    share = mask.mean()
    if share < 0.05 or share > 0.90:
        bg = _common_color(arr[int(h * 0.60):int(h * 0.90), :, :])
        mask = np.abs(arr - bg).sum(axis=2) > 90
    return mask


def _ridge(lum, axis, gap=3, thr=55):
    """Маска тонких светлых линий.

    Пиксель считается линией, если он заметно светлее того, что лежит в
    трёх пикселях по обе стороны от него поперёк линии. Ровная заливка
    любого цвета даёт ноль, обычный перепад "фон - панель" тоже: там
    светлее только с одной стороны. Остаются именно линии - рамки таблиц
    и разделители строк.
    """
    up = np.roll(lum, gap, axis=axis)
    dn = np.roll(lum, -gap, axis=axis)
    m = (lum - np.maximum(up, dn)) > thr
    if axis == 0:
        m[:gap, :] = False
        m[-gap:, :] = False
    else:
        m[:, :gap] = False
        m[:, -gap:] = False
    return m


def _longest_true(row):
    """Длина самой длинной непрерывной цепочки True в строке."""
    if not row.any():
        return 0
    edges = np.flatnonzero(np.diff(np.concatenate(
        ([0], row.astype(np.int8), [0]))))
    return int((edges[1::2] - edges[0::2]).max())


def _spans(col, fill=12, least=60):
    """Отрезки, занятые линией, с заклейкой мелких разрывов.

    Вертикальная рамка прерывается там, где в неё упирается разделитель
    строк, поэтому разрывы в десяток пикселей склеиваем.
    """
    ys = np.flatnonzero(col)
    if ys.size == 0:
        return []
    out, s, p = [], ys[0], ys[0]
    for y in ys[1:]:
        if y - p > fill:
            out.append((int(s), int(p)))
            s = y
        p = y
    out.append((int(s), int(p)))
    return [o for o in out if o[1] - o[0] >= least]


def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def find_bands(arr):
    """Находит таблицы: сколько бы их ни было и какого бы цвета ни было табло.

    Раньше панели искали как чёрные прямоугольники на оранжевом поле. Но
    табло умеет показывать и восемь таблиц вместо четырёх, и в другой
    раскраске - на чёрном фоне. Тогда "чёрный прямоугольник на другом
    фоне" перестаёт существовать как признак, и не находилось ничего.

    Поэтому опираемся на то, что есть у таблицы при любой раскраске: она
    расчерчена линиями. Горизонтальные разделители строк дают колонки -
    по ширине таблиц; вертикальные рамки дают ряды - по высоте.

    Возвращает список (панели, верх, низ), сверху вниз.
    """
    lum = arr.sum(axis=2)
    H, W = lum.shape
    hor = _ridge(lum, 0)
    ver = _ridge(lum, 1)

    # Боковые рамки таблиц: длинные вертикальные линии.
    cv = ver.sum(axis=0)
    lim = max(H * 0.18, cv.max() * 0.35)
    cand = []
    for x in range(W):
        if cv[x] <= lim:
            continue
        if cand and x - cand[-1][0] <= 4:
            if cv[x] > cand[-1][1]:
                cand[-1] = (x, cv[x])
        else:
            cand.append((x, cv[x]))
    # оставляем самые заметные линии: их и так немного, а перебор ниже
    # растёт как квадрат от их числа
    cand.sort(key=lambda c: -c[1])
    cand = sorted(c[0] for c in cand[:40])
    if len(cand) < 2:
        return []

    # Среди длинных линий есть и лишняя - черта после номера строки внутри
    # таблицы. Отсеиваем перебором: таблицы одной ширины и стоят с равным
    # шагом, поэтому подбираем ширину и промежуток так, чтобы получившаяся
    # решётка объяснила как можно больше найденных линий. Лишняя черта в
    # решётку не ложится и остаётся за бортом.
    near = np.zeros(W + 8, dtype=bool)
    for c in cand:
        near[max(0, c - 3):min(W, c + 4)] = True

    widths = sorted({b - a for i, a in enumerate(cand) for b in cand[i + 1:]
                     if b - a >= W // 12})
    best = None
    for w in widths:
        for x0 in cand:
            right = [x for x in cand if x > x0 + w + 2]
            g = (min(right) - (x0 + w)) if right else 14
            if not (2 <= g <= max(2, w // 3)):
                g = 14
            marks, x = [], x0
            while x - w - g >= 0:
                x -= w + g
            while x + w < W:
                marks += [x, x + w]
                x += w + g
            hits = sum(1 for m in marks if near[m])
            score = (hits, -w)
            if best is None or score > best[0]:
                best = (score, w, g, x0)
    if best is None:
        return []
    _, width, gap, anchor = best
    step = width + gap

    def snap(x):
        close = [b for b in cand if abs(b - x) <= 4]
        return min(close, key=lambda b: abs(b - x)) if close else x

    grid, x = [], anchor
    while x - step >= 0:
        x -= step
    while x + width < W:
        x0, x1 = snap(x), snap(x + width)
        if 0 <= x0 < x1 < W:
            grid.append((x0, x1))
        x += step
    if not grid:
        return []
    bx = cand

    # Ряды таблиц: по вертикали рамка идёт на всю высоту таблицы, а между
    # рядами обрывается. Разрывы от разделителей строк заклеиваем.
    seen = []
    for b in bx:
        seen.extend(_spans(ver[:, b]))
    if not seen:
        return []
    seen.sort()
    rows = [list(seen[0])]
    for s in seen[1:]:
        if _overlap(rows[-1], s) > (s[1] - s[0]) * 0.5:
            rows[-1][0] = min(rows[-1][0], s[0])
            rows[-1][1] = max(rows[-1][1], s[1])
        else:
            rows.append(list(s))
    rows = [r for r in rows if r[1] - r[0] >= 40]

    out = []
    for (top, bottom) in rows:
        band, panels = (top, bottom), []
        for (x0, x1) in grid:
            # Рамку считаем на месте, если она закрывает большую часть
            # высоты ряда. Целиком она бывает редко: её рвут разделители
            # строк и подсветка, поэтому складываем все куски.
            best = 0
            for xx in (x0, x1):
                cov = sum(_overlap(s, band)
                          for s in _spans(ver[:, xx], least=20))
                best = max(best, cov)
            if best > (bottom - top) * 0.4:
                panels.append((x0, x1))
        if not panels:
            continue
        # В ряду должна быть хоть одна расчерченная таблица - иначе это
        # не таблицы, а случайные длинные линии на экране.
        lines = sum(1 for y in range(top, min(bottom, H))
                    if _longest_true(hor[y, panels[0][0]:panels[-1][1]]) >= width * 0.5)
        if lines < 3:
            continue
        out.append((panels, top, bottom + 1))
    out.sort(key=lambda b: b[1])
    return out


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

    # Строк ровно столько, сколько целиком помещается в таблицу.
    #
    # Раньше здесь стояло "не меньше восемнадцати": в старой раскладке из
    # четырёх таблиц строк всегда хватало, и это ничего не портило. Когда
    # табло стало показывать восемь таблиц, нижний ряд стал вдвое ниже -
    # и лишние строки сетки уехали за нижний край, прямо на панель задач
    # Windows. Оттуда в списке и появлялся пассажир "Сеть".
    n = int((bottom - origin) / pitch)
    n = max(1, min(30, n))
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


# Кэш распознанных ячеек.
#
# Между соседними кадрами табло меняется на одну-две строки, а шрифт на нём
# пиксель в пиксель одинаковый. Значит одну и ту же строку незачем распознавать
# заново: ключ - хэш самих пикселей, значение - что из них вышло. Дороже всего
# именно tesseract, поэтому кэшируется только он; сверка фамилий по истории
# и отсев шума считаются каждый раз заново.
#
# Версию нужно поднимать при любой правке разбора - иначе старые записи
# продолжат отдавать результат по прежним правилам.
# 2 - уточнён разбор SPL: короткие "Тау", "Рау", "нау" - это искажённое
#     "way", а не "Tanay". Записи, сделанные по прежнему правилу, нужно
#     перечитать, поэтому версия поднята.
# 3 - расширено опознание латинских фамилий: ловим не только "АБЧиБай",
#     но и "АБЧИБай". Старые записи с кашей вместо имени надо перечитать.
CACHE_VER = 3
# Ветка ocr каждый раз пересоздаётся и уходит force-push'ем, истории коммитов
# у неё нет - но файл летит по сети на каждом прогоне, поэтому держим его
# небольшим: полутора тысяч строк хватает на несколько прыжковых дней.
CACHE_LIMIT = 1500
CELL_CACHE = {}


def cell_key(name_cell, cat_cell):
    h = hashlib.blake2b(digest_size=10)
    h.update(b"v%d|" % CACHE_VER)
    h.update(b"%dx%d|" % name_cell.size)
    h.update(name_cell.tobytes())
    h.update(b"|")
    h.update(cat_cell.tobytes())
    return h.hexdigest()


# Ниже этого порога распознаванию верить нельзя: именно так в кэш попали
# "Шумейючн Максим" и "ФИНАЛАХ АААЛЛАС" - и остались бы там навсегда.
# Неуверенные строки перечитываем каждый раз: сверка по истории вытянет их.
CACHE_MIN_CONF = 60.0


def cacheable(res):
    """В кэш идёт только уверенный разбор - сомнительный лучше пересчитать."""
    if res.get("service"):
        return True
    letters = re.sub(r"[^А-Яа-яЁёA-Za-z]", "", res.get("name") or "")
    try:
        conf = float(res.get("_conf", -1))
    except (TypeError, ValueError):
        conf = -1.0
    return (len(letters) >= 4 and res.get("cat_full") in KNOWN_CATS
            and conf >= CACHE_MIN_CONF)


def recognize_row(job):
    """Распознаёт одну строку: имя и категорию. Вызывается из нескольких потоков.

    Tesseract работает отдельным процессом, поэтому потоки дают настоящее
    ускорение - на четырёх ядрах раннера примерно вчетверо.
    """
    name_cell, cat_cell = job["name_cell"], job["cat_cell"]

    hit = CELL_CACHE.get(job.get("key"))
    if hit:
        job["result"] = {"n": job["n"], "name": hit[0], "staff": bool(hit[1]),
                         "cat": short_category(hit[2]), "cat_full": hit[2],
                         "cat_raw": hit[3], "_conf": hit[4],
                         "service": bool(hit[5])}
        job["cached"] = True
        return job

    name_txt, name_conf = ocr_best(name_cell)
    # Иностранная фамилия под русской моделью превращается в кашу
    # ("Abdulbari Qubaisi" -> "АБЧиБай Чиа"). Заметив это по заглавным
    # посреди слова, перечитываем ячейку только латиницей. Лишний проход
    # идёт редко, на общую скорость не влияет.
    if looks_garbled(name_txt):
        en_txt, en_conf = ocr_best(name_cell, cfg=TESS_EN)
        if en_txt and mostly_latin(en_txt) and en_conf >= name_conf - 8:
            name_txt, name_conf = en_txt, en_conf
    name = normalize_name(name_txt.replace("|", "").strip())
    is_service = name.startswith("КВОРУМ") or name == "(Вып.)"
    is_staff = False
    if not is_service:
        name, is_staff = match_staff(name)

    cat_raws = ocr_category(cat_cell)
    cat = pick_category(cat_raws, is_service)
    cat_raw = next((x for x in cat_raws if x), "")
    # спасательный проход: если ансамбль дал мусор, пробуем
    # увеличение x8 - оно вытаскивает совсем мелкие ячейки
    if cat not in KNOWN_CATS and cat != "???":
        for scale in (8, 6):
            try:
                r2 = pytesseract.image_to_string(
                    prep(cat_cell, scale=scale), config=TESS_MIX8).strip()
            except Exception:
                continue
            n2 = normalize_category(r2)
            if n2 in KNOWN_CATS:
                cat, cat_raw = n2, r2
                break

    job["result"] = {"n": job["n"], "name": name, "staff": is_staff,
                     "cat": short_category(cat), "cat_full": cat,
                     "cat_raw": cat_raw, "_conf": name_conf,
                     "service": is_service}
    return job


def main():
    """Обёртка. Если разбор упадёт, всё равно пишем board.json.

    Сборка публикует табло только когда скрипт завершился без ошибки.
    Значит любое падение здесь замораживает страницу на последнем удачном
    прогоне - снаружи это выглядит как "приложение перестало обновляться",
    и понять причину нельзя. Поэтому ошибку записываем в сам board.json.
    """
    try:
        run()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        dst = sys.argv[2] if len(sys.argv) > 2 else "board.json"
        from datetime import datetime
        try:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump({
                    "updated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "clock": "", "work_until": "", "message": "",
                    "loads": [], "status": "no_board",
                    "why": {"ошибка": "%s: %s" % (type(e).__name__, e)},
                }, f, ensure_ascii=False, indent=1)
        except Exception:
            pass


def run():
    t_start = time.time()
    src = sys.argv[1] if len(sys.argv) > 1 else "board.png"
    dst = sys.argv[2] if len(sys.argv) > 2 else "board.json"
    dbg = sys.argv[3] if len(sys.argv) > 3 else None
    hist_path = sys.argv[4] if len(sys.argv) > 4 else None
    stats_path = sys.argv[5] if len(sys.argv) > 5 else None

    # В history.json теперь два раздела: счётчики фамилий и кэш ячеек.
    # Старый формат (просто счётчики) читается как раньше.
    history = {}
    if hist_path and os.path.exists(hist_path):
        try:
            with open(hist_path, encoding="utf-8") as f:
                raw = json.load(f)
            names = raw.get("names") if isinstance(raw.get("names"), dict) else raw
            history = {str(k): float(v) for k, v in names.items()}
            cached = raw.get("cells") if isinstance(raw, dict) else None
            if isinstance(cached, dict) and raw.get("cache_ver") == CACHE_VER:
                CELL_CACHE.update(cached)
        except Exception as e:
            sys.stderr.write("history load error: %s\n" % e)

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

    # Разбор разметки не должен ронять весь прогон: если он сломается,
    # сборка молча перестанет публиковать табло, и снаружи это выглядит
    # как "приложение не обновляется". Лучше отдать пустой результат и
    # написать в нём, что именно упало.
    crash = None
    try:
        bands = find_bands(arr)
    except Exception as e:
        bands = []
        crash = "%s: %s" % (type(e).__name__, e)
        sys.stderr.write("find_bands упал: %s\n" % crash)

    # Если панелей нет - скорее всего RMS показывает заглушку "Подключение..."
    if not bands:
        result["status"] = "no_board"
        # Улика на будущее: по ней видно, была ли картинка пустой на самом
        # деле или разбор ошибся. "Занято" - какая доля экрана не фон.
        # Пустое табло даёт единицы процентов, живое - десятки.
        result["why"] = {}
        if crash:
            result["why"]["ошибка разбора"] = crash
        try:
            m = not_background(arr)
            result["why"]["занято"] = round(float(m.mean()), 3)
            result["why"]["ряды_с_таблицами"] = int((m.mean(axis=1) > 0.30).sum())
        except Exception:
            pass
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print("панели не найдены - похоже, табло сейчас не показывается")
        if stats_path:
            try:
                mt = os.path.getmtime(src)
            except OSError:
                mt = None
            now = time.time()
            update_stats(stats_path, {
                "t": round(now, 1), "iso": result["updated"],
                "digest": None, "changed": False, "board": False,
                "lag": round(now - mt, 1) if mt else None,
                "sec": round(now - t_start, 1), "rows": 0, "clock": "",
            })
        return

    top0 = bands[0][1]            # верх самого первого ряда - под ним шапка
    bottom_last = bands[-1][2]    # низ последнего ряда - под ним сообщение
    result["grid"] = {"bands": []}

    # шапка находится над таблицей, её положение тоже плавает вместе с окном
    hy0 = max(0, top0 - 58)
    hy1 = max(hy0 + 1, top0 - 26)
    result["clock"] = ocr(im.crop((0, hy0, 420, hy1)), cfg=TESS_MIX)
    result["work_until"] = fix_work_until(
        ocr(im.crop((max(0, W - 420), hy0, W, hy1)), cfg=TESS_MIX))

    # ---- сообщение внизу (под таблицей) ----
    msg_crop = im.crop((0, min(H - 1, bottom_last + 90), W, min(H, bottom_last + 190)))
    msg_arr = np.array(msg_crop).astype(int)
    # Есть ли там вообще надпись. Считаем не тёмные пиксели, а непохожие
    # на здешний фон: на оранжевом табло надпись тёмная, на чёрном светлая.
    if msg_arr.size:
        far = (np.abs(msg_arr - _common_color(msg_arr)).sum(axis=2) > 90).mean()
    else:
        far = 0.0
    if 0.002 < far < 0.5:
        result["message"] = clean_message(read_message(msg_crop))

    draw = ImageDraw.Draw(im) if dbg else None
    jobs = []          # что распознавать; сами прогоны идут ниже, параллельно

    # ---- ряды таблиц, в каждом - свои панели ----
    pi = -1
    for panels, top, bottom in bands:
      origin, pitch, n_rows = fit_grid(arr, panels[0][0], panels[0][1], top, bottom)
      result["grid"]["bands"].append(
          {"top": top, "bottom": bottom, "origin": round(origin, 2),
           "pitch": round(pitch, 3), "rows": n_rows,
           "panels": [list(p) for p in panels]})

      for (x0, x1) in panels:
        pi += 1
        pw = x1 - x0
        # заголовок взлёта над панелью; у верхнего ряда места может не хватать
        ty_a = max(0, top - 26)
        ty_b = max(ty_a + 1, top - 3)
        title = ocr(im.crop((max(0, x0 - 10), ty_a, x1 + 10, ty_b)), cfg=TESS_MIX)
        # на табло пишут "готов 5 мин." - на странице хотим "готовность"
        title = re.sub(r"\bготов\b", "готовность", title)
        title = fix_ready_minutes(title)
        title = fix_departed(title)
        title = fix_takeoff_word(title)

        # capacity считаем по самому табло: занятые + свободные строки.
        # Это вместимость борта (у Л-410 - 18), а не высота сетки: ниже
        # последней строки идёт чёрное поле, оно к местам отношения не имеет.
        craft, craft_seats = detect_aircraft(title)
        load = {"index": pi + 1, "title": title, "aircraft": craft,
                "capacity": 0, "rows": [], "free_from": None, "free": 0}
        seats = 0

        for r in range(n_rows):
            ry0 = int(round(origin + r * pitch))
            ry1 = int(round(origin + (r + 1) * pitch))
            # подстраховка: ниже таблицы читать нечего, там уже чужое
            if ry1 > bottom:
                break
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

            if kind in ("free", "filled"):
                seats = r + 1          # последняя непустая строка панели
            if kind == "free":
                load["free"] += 1
                if load["free_from"] is None:
                    load["free_from"] = r + 1
            if kind != "filled":
                continue

            # границы колонок задаём долей ширины панели - она чуть плавает
            nx0 = x0 + int(pw * COL_NAME[0])
            nx1 = x0 + int(pw * COL_NAME[1])
            cx0 = x0 + int(pw * COL_CAT[0])
            cx1 = x0 + int(pw * COL_CAT[1])

            name_cell = im.crop((nx0, ty0, nx1, ty1))
            cat_cell = im.crop((cx0, ty0, cx1, ty1))
            # сами прогоны распознавания делаются позже и параллельно
            jobs.append({"load": load, "n": r + 1,
                         "name_cell": name_cell, "cat_cell": cat_cell,
                         "key": cell_key(name_cell, cat_cell)})

        # мест на борту: меряем по табло, но у отправленного взлёта зелёных
        # строк уже нет - тогда берём из таблицы бортов
        load["capacity"] = seats or craft_seats or 0
        if craft_seats and seats and seats != craft_seats:
            sys.stderr.write("панель %d: на табло %d мест, у %s обычно %d\n"
                             % (pi + 1, seats, craft, craft_seats))
        if load["rows"] or load["free_from"] or jobs:
            result["loads"].append(load)

    # ---- само распознавание: параллельно по числу ядер ----
    fresh = {}
    if jobs:
        workers = min(8, max(2, (os.cpu_count() or 2)))
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(recognize_row, jobs))
        except Exception as e:
            sys.stderr.write("параллельный режим не вышел (%s), считаю по очереди\n" % e)
            for job in jobs:
                recognize_row(job)

        # снимок для кэша - пока в строках ещё есть служебные поля
        for job in jobs:
            res = job.get("result")
            if res and job.get("key") and cacheable(res):
                fresh[job["key"]] = [res["name"], 1 if res.get("staff") else 0,
                                     res.get("cat_full", ""), res.get("cat_raw", ""),
                                     res.get("_conf", 0), 1 if res.get("service") else 0]
        reused = sum(1 for j in jobs if j.get("cached"))
        sys.stderr.write("строк: %d, из кэша: %d\n" % (len(jobs), reused))

        for job in jobs:
            res = job.get("result")
            if not res:
                continue
            # шум на пустых строках (курсор, блики): пара букв вместо имени
            # и мусор вместо категории - такую строку выбрасываем
            letters = re.sub(r"[^А-Яа-яЁёA-Za-z]", "", res["name"])
            # сверяем с полной подписью: короткой в списке известных нет
            if not res["service"] and len(letters) < 4 and res.get("cat_full") not in KNOWN_CATS:
                continue
            res.pop("service", None)
            res.pop("cat_full", None)
            job["load"]["rows"].append(res)

        for l in result["loads"]:
            l["rows"].sort(key=lambda r0: r0["n"])

    # панели, где после отсева ничего не осталось, на страницу не выводим
    result["loads"] = [l for l in result["loads"] if l["rows"] or l["free_from"]]
    fix_load_numbers(result["loads"])

    reconcile_names(result["loads"], history)
    for l in result["loads"]:
        for r0 in l["rows"]:
            r0.pop("_conf", None)

    if hist_path:
        history = update_history(history, result["loads"])
        # свежие ячейки кладём первыми, хвост старых обрезаем по лимиту
        cells = dict(fresh)
        for k, v in CELL_CACHE.items():
            if len(cells) >= CACHE_LIMIT:
                break
            cells.setdefault(k, v)
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump({"names": history, "cache_ver": CACHE_VER, "cells": cells},
                      f, ensure_ascii=False, separators=(",", ":"))

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
    n_panels = sum(len(b[0]) for b in bands)
    dur = time.time() - t_start
    print("панелей: %d, распознано строк: %d, время: %.1f с"
          % (n_panels, total, dur))
    print("сообщение: %r" % result["message"])

    # ---- замеры: как часто меняется кадр и какова задержка ----
    if stats_path:
        shot_mtime = None
        try:
            shot_mtime = os.path.getmtime(src)
        except Exception:
            pass
        # "новый кадр" считаем по содержимому табло, а не по файлу.
        # Часы в шапке в расчёт не идут - они тикают сами по себе.
        # А вот заголовок взлёта берём целиком: "готовность 20 мин." -
        # это настоящая информация, её смена и есть изменение табло.
        digest = None
        try:
            import hashlib
            content = [
                [l.get("title"), l.get("free"),
                 [(r["n"], r["name"], r["cat"]) for r in l["rows"]]]
                for l in result["loads"]
            ]
            blob = json.dumps(content, ensure_ascii=False, sort_keys=True)
            digest = hashlib.md5(blob.encode("utf-8")).hexdigest()
        except Exception as e:
            sys.stderr.write("hash error: %s\n" % e)
        prev = {}
        if os.path.exists(stats_path):
            try:
                with open(stats_path, encoding="utf-8") as f:
                    prev = json.load(f)
            except Exception:
                prev = {}
        last_digest = (prev.get("runs") or [{}])[-1].get("digest")
        now = time.time()
        entry = {
            "t": round(now, 1),
            "iso": result["updated"],
            "digest": digest,
            "changed": bool(digest and digest != last_digest),
            "lag": round(now - shot_mtime, 1) if shot_mtime else None,
            "sec": round(dur, 1),
            "rows": total,
            "clock": result["clock"],
        }
        summary = update_stats(stats_path, entry)
        print("замеры:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

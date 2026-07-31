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
import sys
import difflib

import numpy as np
from PIL import Image, ImageDraw
import pytesseract

# ---------- геометрия (снята по реальному скриншоту 1366x768) ----------
PANEL_TOP = 75          # верх таблицы
PANEL_BOTTOM = 504      # низ таблицы
ROWS = 24               # строк в панели
ROW_H = (PANEL_BOTTOM - PANEL_TOP) / ROWS   # 17.875

TITLE_Y = (47, 70)      # заголовок взлёта над панелью (бывает красным или голубым)
HEADER_Y = (16, 48)     # шапка: дата/время слева, "Работа до" справа
MSG_Y = (600, 680)      # зона сообщения внизу

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


def prep(img, invert=True, scale=4, threshold=110, margin=16):
    """Готовит картинку к распознаванию.

    Фон определяется автоматически (самый частый цвет), текстом считается всё,
    что от него заметно отличается. Так одинаково хорошо работают и белые буквы
    на чёрном, и красные или голубые на оранжевом.

    Дальше: обрезка по основной полосе текста (чтобы выбросить линии-разделители),
    увеличение и белые поля - Tesseract заметно точнее, когда текст не у края.
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
        if row_has[y] > w * 0.85:
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
    return canvas


def ocr(img, cfg=TESS_RU, invert=True, scale=4):
    """Распознаёт одну строку текста."""
    if img.width < 3 or img.height < 3:
        return ""
    try:
        return pytesseract.image_to_string(prep(img, invert, scale), config=cfg).strip()
    except Exception as e:
        sys.stderr.write("ocr error: %s\n" % e)
        return ""


def snap_category(text):
    """Подгоняет распознанную категорию к ближайшей известной."""
    t = (text or "").strip()
    if not t:
        return ""
    m = difflib.get_close_matches(t, KNOWN_CATS, n=1, cutoff=0.6)
    return m[0] if m else t


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
            name = ocr(name_cell)
            cat_raw = ocr(im.crop((cx0, ty0, cx1, ty1)), cfg=TESS_MIX)

            # копим образцы того, что видит распознаватель
            if len(SAMPLES) < 12:
                SAMPLES.append((name_cell.copy(), prep(name_cell)))
            load["rows"].append({
                "n": r + 1,
                "name": name.replace("|", "").strip(),
                "cat": snap_category(cat_raw),
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

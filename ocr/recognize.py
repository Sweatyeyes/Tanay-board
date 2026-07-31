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

# отступы внутри строки, относительно левого края панели
COL_NUM  = (2, 30)
COL_NAME = (28, 246)
COL_CAT  = (247, 321)

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


def find_panels(arr):
    """Находит чёрные панели по вертикальным полосам тёмных пикселей."""
    dark = arr.sum(axis=2) < 180
    band = dark[PANEL_TOP:PANEL_BOTTOM, :]
    colsum = band.sum(axis=0)
    thr = (PANEL_BOTTOM - PANEL_TOP) * 0.5
    panels, inside, start = [], False, 0
    for x in range(arr.shape[1]):
        if colsum[x] > thr and not inside:
            inside, start = True, x
        elif colsum[x] <= thr and inside:
            inside = False
            if x - start > 100:
                panels.append((start, x))
    if inside and arr.shape[1] - start > 100:
        panels.append((start, arr.shape[1]))
    return panels


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

    panels = find_panels(arr)

    # Если панелей нет - скорее всего RMS показывает заглушку "Подключение..."
    if not panels:
        result["status"] = "no_board"
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print("панели не найдены - похоже, табло сейчас не показывается")
        return

    # ---- шапка ----
    result["clock"] = ocr(im.crop((0, HEADER_Y[0], 420, HEADER_Y[1])),
                          cfg=TESS_MIX, invert=False)
    right = ocr(im.crop((max(0, W - 420), HEADER_Y[0], W, HEADER_Y[1])),
                cfg=TESS_MIX, invert=False)
    result["work_until"] = right

    # ---- сообщение внизу ----
    msg_crop = im.crop((0, MSG_Y[0], W, min(H, MSG_Y[1])))
    msg_arr = np.array(msg_crop).astype(int)
    if (msg_arr.sum(axis=2) < 200).mean() > 0.002:
        result["message"] = ocr(msg_crop, cfg=TESS_RU, invert=False, scale=2)

    draw = ImageDraw.Draw(im) if dbg else None

    # ---- панели ----
    for pi, (x0, x1) in enumerate(panels):
        title = ocr(im.crop((max(0, x0 - 10), TITLE_Y[0], x1 + 10, TITLE_Y[1])),
                    cfg=TESS_MIX, invert=True)

        load = {"index": pi + 1, "title": title, "capacity": ROWS,
                "rows": [], "free_from": None}

        for r in range(ROWS):
            ry0 = int(round(PANEL_TOP + r * ROW_H))
            ry1 = int(round(PANEL_TOP + (r + 1) * ROW_H))
            cell = im.crop((x0, ry0, x1, ry1))
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

            name_cell = im.crop((x0 + COL_NAME[0], ty0, x0 + COL_NAME[1], ty1))
            name = ocr(name_cell)
            cat_raw = ocr(im.crop((x0 + COL_CAT[0], ty0, x0 + COL_CAT[1], ty1)),
                          cfg=TESS_MIX)

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сравнение движков распознавания на снимках табло.

Запускается вручную через workflow "OCR bench": там есть сеть, поэтому
EasyOCR и PaddleOCR можно поставить. Локально в песочнице их нет.

Считает две метрики по колонке с фамилиями:
  - строк точно: сколько строк совпало с эталоном символ в символ;
  - символов: посимвольная точность (частичное попадание тоже видно).

Использование:
    python ocr-test/bench.py [движок ...]
    движки: tesseract, easyocr, paddle   (по умолчанию все доступные)
"""

import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ocr"))
sys.path.insert(0, HERE)

import recognize as R          # геометрия таблицы берётся из рабочего кода

SNAPS = (4, 5, 6, 8)


def cells():
    """Ячейки с фамилиями и их эталонный текст."""
    out = []
    for snap in SNAPS:
        img_path = os.path.join(HERE, "source-%d.png" % snap)
        truth_path = os.path.join(HERE, "truth%d.json" % snap)
        if not (os.path.exists(img_path) and os.path.exists(truth_path)):
            continue
        truth = json.load(open(truth_path, encoding="utf-8"))["panels"]
        im = Image.open(img_path).convert("RGB")
        arr = np.array(im).astype(int)
        bands = R.find_bands(arr)
        if not bands:
            continue
        panels, top, bottom = bands[0]
        origin, pitch, n_rows = R.fit_grid(arr, panels[0][0], panels[0][1], top, bottom)
        for pi, pt in enumerate(truth):
            if pi >= len(panels):
                break
            x0, x1 = panels[pi]
            pw = x1 - x0
            nx0 = x0 + int(pw * R.COL_NAME[0])
            nx1 = x0 + int(pw * R.COL_NAME[1])
            for ri, want in enumerate(pt["names"]):
                ry0 = int(round(origin + ri * pitch))
                ry1 = int(round(origin + (ri + 1) * pitch))
                out.append((im.crop((nx0, ry0 + 1, nx1, ry1 - 1)), want))
    return out


def upscale(img, k=4):
    return img.resize((img.width * k, img.height * k), Image.LANCZOS)


# ---------------- движки ----------------

def engine_tesseract():
    import pytesseract
    R.load_font if False else None

    def run(img):
        txt, _ = R.ocr_best(img)
        return R.normalize_name(txt.replace("|", "").strip())
    return run


def engine_easyocr():
    import easyocr
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)

    def run(img):
        res = reader.readtext(np.array(upscale(img)), detail=0, paragraph=True)
        return " ".join(res).strip()
    return run


def engine_paddle():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="cyrillic", use_angle_cls=False, show_log=False)

    def run(img):
        res = ocr.ocr(np.array(upscale(img)), cls=False)
        if not res or not res[0]:
            return ""
        return " ".join(line[1][0] for line in res[0]).strip()
    return run


ENGINES = {"tesseract": engine_tesseract,
           "easyocr": engine_easyocr,
           "paddle": engine_paddle}


def score(run, data, limit=None):
    rows_ok = chars_ok = chars_total = 0
    items = data[:limit] if limit else data
    misses = []
    t0 = time.time()
    for img, want in items:
        try:
            got = run(img)
        except Exception as e:
            got = "ERR:%s" % str(e)[:40]
        # сравниваем столько слов, сколько в эталоне: у обычной строки два
        # (фамилия и имя), у служебной "КВОРУМ DZ 1" - три
        got2 = " ".join(got.split()[:len(want.split())])
        if got2 == want:
            rows_ok += 1
        elif len(misses) < 12:
            misses.append((want, got2))
        n = min(len(got2), len(want))
        chars_ok += sum(1 for i in range(n) if got2[i] == want[i])
        chars_total += max(len(got2), len(want))
    dt = time.time() - t0
    return {"rows": (rows_ok, len(items)), "chars": (chars_ok, chars_total),
            "sec": dt, "misses": misses}


def main():
    want_engines = sys.argv[1:] or list(ENGINES)
    data = cells()
    print("ячеек для проверки: %d" % len(data))
    if not data:
        print("нет снимков или эталонов - положите source-*.png и truth*.json рядом")
        return

    for name in want_engines:
        if name not in ENGINES:
            print("неизвестный движок: %s" % name)
            continue
        print("\n=== %s ===" % name)
        try:
            run = ENGINES[name]()
        except Exception as e:
            print("не завелся: %s" % e)
            continue
        s = score(run, data)
        ro, rt = s["rows"]
        co, ct = s["chars"]
        print("строк точно: %d/%d (%.0f%%)" % (ro, rt, 100.0 * ro / rt))
        print("символов:    %d/%d (%.1f%%)" % (co, ct, 100.0 * co / ct))
        print("время:       %.1f с (%.0f мс на строку)" % (s["sec"], s["sec"] / rt * 1000))
        if s["misses"]:
            print("промахи:")
            for want, got in s["misses"]:
                print("   %-28r -> %r" % (want, got))


if __name__ == "__main__":
    main()

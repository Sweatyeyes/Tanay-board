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
        if not os.path.exists(img_path):
            print("нет файла:", img_path)
            continue
        if not os.path.exists(truth_path):
            print("нет файла:", truth_path)
            continue
        truth = json.load(open(truth_path, encoding="utf-8"))["panels"]
        im = Image.open(img_path).convert("RGB")
        arr = np.array(im).astype(int)
        bands = R.find_bands(arr)
        if not bands:
            print("панели не найдены на", os.path.basename(img_path))
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
    # имя набора для кириллицы у разных версий разное - пробуем по очереди
    ocr, last = None, None
    for lang in ("ru", "cyrillic", "rus"):
        for kw in ({"lang": lang}, {"lang": lang, "use_angle_cls": False}):
            try:
                ocr = PaddleOCR(**kw)
                print("   paddle: язык %r" % lang)
                break
            except Exception as e:
                last = e
        if ocr: break
    if ocr is None:
        raise RuntimeError("ни один язык не подошёл: %s" % last)

    def run(img):
        a = np.array(upscale(img))
        try:
            res = ocr.predict(a)              # новый API (3.x)
            if isinstance(res, list) and res and isinstance(res[0], dict):
                return " ".join(res[0].get("rec_texts", [])).strip()
        except Exception:
            pass
        res = ocr.ocr(a)                      # старый API (2.x)
        if not res or not res[0]:
            return ""
        out = []
        for line in res[0]:
            try: out.append(line[1][0])
            except Exception: pass
        return " ".join(out).strip()
    return run


def _post(text):
    """Наша постобработка: чистка, словарь имён, список стаффа."""
    name = R.normalize_name(str(text).replace("|", "").strip())
    if not name.startswith("КВОРУМ") and name != "(Вып.)":
        name, _ = R.match_staff(name)
    return name


def engine_easy_post():
    """EasyOCR плюс наша постобработка - так работал бы рабочий конвейер."""
    raw = engine_easyocr()
    return lambda img: _post(raw(img))


def engine_duet():
    """Два движка вместе: где расходятся - берём ответ, похожий на фамилию.

    Ошибки у Tesseract и EasyOCR почти не пересекаются, поэтому если один
    из них дал знакомого стаффа или осмысленное имя - берём его.
    """
    t = engine_tesseract()
    e = engine_easyocr()

    def run(img):
        a = t(img)
        b = _post(e(img))
        if a == b:
            return a
        # предпочитаем вариант, совпавший со списком стаффа
        for cand in (a, b):
            parts = cand.split()
            if len(parts) >= 2 and R.match_staff(" ".join(parts[:2]))[1]:
                return cand
        # иначе - тот, где имя нашлось в словаре имён
        for cand in (b, a):
            parts = cand.split()
            if len(parts) >= 2 and parts[1] in R.FIRST_NAMES:
                return cand
        return b
    return run


ENGINES = {"tesseract": engine_tesseract,
           "easyocr": engine_easyocr,
           "easy_post": engine_easy_post,
           "duet": engine_duet,
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
    print("папка замера:", HERE)
    try:
        print("что в ней лежит:", sorted(os.listdir(HERE)))
    except Exception as e:
        print("не читается:", e)
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

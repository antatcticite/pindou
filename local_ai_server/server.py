import base64
import json
import os
import re
import sys
import time
from typing import Optional, Set, Tuple, Any, Dict, List
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".paddlex_cache"),
)


def _json_response(handler: BaseHTTPRequestHandler, code: int, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler):
    try:
        n = int(handler.headers.get("Content-Length", "0") or "0")
    except Exception:
        n = 0
    if n <= 0:
        return {}
    raw = handler.rfile.read(n)
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return {}


_CODE_RE = re.compile(r"^([A-Z])0*(\d{1,3})$")
_CODE_IN_TEXT_RE = re.compile(r"\b([A-Z]{1,3}\s*0*[0-9]{1,3})\b")


def _norm_code(s: str) -> str:
    s = (s or "").strip().upper()
    s = s.replace("θ", "0").replace("Θ", "0").replace("Ø", "0")
    s = re.sub(r"[^A-Z0-9]", "", s)
    s = re.sub(r"([A-Z]{1,3})O([0-9])", r"\g<1>0\g<2>", s)
    s = re.sub(r"([A-Z]{1,3})I([0-9])", r"\g<1>1\g<2>", s)
    s = re.sub(r"([A-Z]{1,3})L([0-9])", r"\g<1>1\g<2>", s)
    m = _CODE_RE.match(s)
    if not m:
        return s
    return m.group(1) + str(int(m.group(2)))


def _levenshtein_lte1(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return 2
    if la == lb:
        diff = 0
        for i in range(la):
            if a[i] != b[i]:
                diff += 1
                if diff > 1:
                    return 2
        return diff
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    i = j = 0
    diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        diff += 1
        if diff > 1:
            return 2
        j += 1
    return 1


def _closest_allowed(code: str, allowed: Optional[Set[str]]) -> Optional[str]:
    if not code:
        return None
    if not allowed:
        return code
    if code in allowed:
        return code
    prefix = re.sub(r"\d", "", code)
    best = None
    best_d = 2
    best_same = False
    for v in allowed:
        d = _levenshtein_lte1(code, v)
        if d >= best_d:
            continue
        same = re.sub(r"\d", "", v) == prefix
        if d < best_d or (not best_same and same):
            best_d = d
            best = v
            best_same = same
    return best if best_d == 1 else None


def _valid_or_close(code: str, allowed: Optional[Set[str]]) -> Optional[str]:
    code = _norm_code(code)
    if not code:
        return None
    if not allowed:
        return code
    if code in allowed:
        return code
    return _closest_allowed(code, allowed)


def _merge_legend_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    source_rank = {
        "paren": 4,
        "glued": 3,
        "same-line": 3,
        "two-row": 5,
        "box-direct": 3,
        "box-glued": 3,
        "box-same-row": 4,
        "box-below": 4,
        "box-crop-below": 6,
    }
    for item in items:
        code = _norm_code(str(item.get("code") or ""))
        try:
            count = int(item.get("count") or 0)
        except Exception:
            count = 0
        if not code or count <= 0:
            continue
        conf = float(item.get("conf") or 0.0)
        source = str(item.get("source") or "")
        prev = merged.get(code)
        prev_conf = float(prev.get("conf") or 0.0) if prev else -1.0
        prev_count = int(prev.get("count") or 0) if prev else 0
        rank = source_rank.get(source, 0)
        prev_rank = source_rank.get(str(prev.get("source") or ""), 0) if prev else -1
        if (
            prev is None
            or rank > prev_rank
            or (rank == prev_rank and conf > prev_conf + 0.04)
            or (rank == prev_rank and abs(conf - prev_conf) <= 0.04 and count > prev_count)
        ):
            merged[code] = {"code": code, "count": count, "conf": conf, "source": source}
    out = list(merged.values())
    out.sort(key=lambda it: (re.sub(r"\d", "", it["code"]), int(re.sub(r"\D", "", it["code"]) or 0), it["code"]))
    return out


def _drop_shadow_glued_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop short-code shadows from glued OCR, e.g. D1654 -> D1=654 and D16=54."""
    normalized = []
    for item in items:
        code = _norm_code(str(item.get("code") or ""))
        try:
            count = int(item.get("count") or 0)
        except Exception:
            count = 0
        m = re.match(r"^([A-Z]{1,3})(\d{1,3})$", code)
        if not m or count <= 0:
            normalized.append((item, code, "", "", count))
            continue
        normalized.append((item, code, m.group(1), m.group(2), count))

    drop_ids = set()
    glued_sources = {"glued", "box-glued"}
    for i, (short_item, short_code, short_letters, short_num, short_count) in enumerate(normalized):
        if not short_letters or not short_num:
            continue
        short_source = str(short_item.get("source") or "")
        for j, (long_item, long_code, long_letters, long_num, long_count) in enumerate(normalized):
            if i == j or not long_letters or not long_num:
                continue
            if short_letters != long_letters:
                continue
            if len(long_num) <= len(short_num) or not long_num.startswith(short_num):
                continue
            suffix = long_num[len(short_num):]
            if not suffix:
                continue
            if long_count < 20:
                continue
            if str(short_count) != suffix + str(long_count):
                continue
            if short_source in glued_sources or short_count >= 500:
                drop_ids.add(id(short_item))
                break
    if not drop_ids:
        return items
    return [it for it in items if id(it) not in drop_ids]


def _repair_low_count_long_code_items(items: List[Dict[str, Any]], allowed: Optional[Set[str]]) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        code = _norm_code(str(item.get("code") or ""))
        try:
            count = int(item.get("count") or 0)
        except Exception:
            count = 0
        m = re.match(r"^([A-Z]{1,3})(\d{2,3})$", code)
        if not m or count <= 0:
            out.append(item)
            continue
        letters, nums = m.group(1), m.group(2)
        short_code = _norm_code(letters + nums[:-1])
        if not short_code or (allowed and short_code not in allowed):
            out.append(item)
            continue
        fixed_count_s = nums[-1] + str(count)
        if fixed_count_s.startswith("0"):
            out.append(item)
            continue
        fixed_count = int(fixed_count_s)
        if count < 20 and 50 <= fixed_count <= 5000:
            fixed = dict(item)
            fixed["code"] = short_code
            fixed["count"] = fixed_count
            fixed["source"] = str(item.get("source") or "") + "-repair"
            out.append(fixed)
        else:
            out.append(item)
    return out


def _hex_to_rgb(v: str) -> Optional[Tuple[int, int, int]]:
    s = (v or "").strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", s or ""):
        return None
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _rgb_dist(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _sample_box_bg_rgb(img, box) -> Optional[Tuple[int, int, int]]:
    try:
        import numpy as np
    except Exception:
        return None
    try:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        w, h = img.size
        left = max(0, int(min(xs)))
        right = min(w, int(max(xs)))
        top = max(0, int(min(ys)))
        bottom = min(h, int(max(ys)))
        bw = max(1, right - left)
        bh = max(1, bottom - top)
        pad_x = int(max(4, bw * 0.35))
        pad_y = int(max(3, bh * 0.35))
        crop = img.crop((max(0, left - pad_x), max(0, top - pad_y), min(w, right + pad_x), min(h, bottom + pad_y)))
        arr = np.array(crop.convert("RGB")).reshape(-1, 3).astype("int16")
        if arr.size == 0:
            return None
        # Drop near-white and near-black text pixels; keep the colored label background.
        mx = arr.max(axis=1)
        mn = arr.min(axis=1)
        sat = mx - mn
        mask = (mx < 245) & (mn > 20) & (sat > 18)
        pts = arr[mask]
        if len(pts) < 12:
            pts = arr[(mx < 245) & (mn > 20)]
        if len(pts) == 0:
            return None
        med = np.median(pts, axis=0)
        return int(med[0]), int(med[1]), int(med[2])
    except Exception:
        return None


def _split_glued_code_count(
    token: str,
    allowed: Optional[Set[str]],
    color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
    bg_rgb: Optional[Tuple[int, int, int]] = None,
) -> Optional[Tuple[str, int]]:
    token = _norm_code(token)
    m = re.match(r"^([A-Z]{1,3})(\d{3,8})$", token)
    if not m:
        return None
    letters, digits = m.group(1), m.group(2)
    candidates: List[Dict[str, Any]] = []
    for k in range(1, min(3, len(digits) - 1) + 1):
        raw_code = _norm_code(letters + digits[:k])
        code = _valid_or_close(raw_code, allowed)
        if not code:
            continue
        try:
            count = int(digits[k:])
        except Exception:
            continue
        if count <= 0 or count > 5000:
            continue
        exact_bonus = 1 if (not allowed or raw_code in allowed) else 0
        leading_zero_penalty = 1 if digits[k:].startswith("0") else 0
        huge_count_penalty = 1 if count >= 2000 else 0
        normal_count_bonus = 1 if 20 <= count <= 1500 else 0
        color_d = None
        if bg_rgb is not None and color_map and code in color_map:
            color_d = _rgb_dist(bg_rgb, color_map[code])
        candidates.append({
            "leading_zero_penalty": leading_zero_penalty,
            "huge_count_penalty": huge_count_penalty,
            "normal_count_bonus": normal_count_bonus,
            "split_len": k,
            "count": count,
            "exact_bonus": exact_bonus,
            "code": code,
            "color_d": color_d,
        })
    if not candidates:
        return None

    by_code: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        prev = by_code.get(c["code"])
        if prev is None or c["count"] > prev["count"]:
            by_code[c["code"]] = c
    candidates = list(by_code.values())

    exact_candidates = [c for c in candidates if c["exact_bonus"]]
    if exact_candidates:
        candidates = exact_candidates

    shadow_ids = set()
    for i, short in enumerate(candidates):
        sm = re.match(r"^([A-Z]{1,3})(\d{1,3})$", short["code"])
        if not sm:
            continue
        for j, long in enumerate(candidates):
            if i == j:
                continue
            lm = re.match(r"^([A-Z]{1,3})(\d{1,3})$", long["code"])
            if not lm or sm.group(1) != lm.group(1):
                continue
            s_num, l_num = sm.group(2), lm.group(2)
            if len(l_num) <= len(s_num) or not l_num.startswith(s_num):
                continue
            suffix = l_num[len(s_num):]
            if str(short["count"]) == suffix + str(long["count"]) and int(long["count"]) >= 20:
                shadow_ids.add(i)
                break
    if shadow_ids and len(shadow_ids) < len(candidates):
        candidates = [c for i, c in enumerate(candidates) if i not in shadow_ids]

    color_candidates = [c for c in candidates if c["color_d"] is not None]
    if len(color_candidates) >= 2:
        color_candidates.sort(key=lambda x: x["color_d"])
        best_c = color_candidates[0]
        second_c = color_candidates[1]
        if best_c["color_d"] + 18 <= second_c["color_d"]:
            return best_c["code"], best_c["count"]

    candidates.sort(key=lambda x: (
        x["leading_zero_penalty"],
        x["huge_count_penalty"],
        -x["normal_count_bonus"],
        -x["exact_bonus"],
        x["split_len"] if x["normal_count_bonus"] else -x["split_len"],
        abs(x["count"] - 180),
        x["code"],
    ))
    best = candidates[0]
    return best["code"], best["count"]


def _color_correct_code(
    code: str,
    bg_rgb: Optional[Tuple[int, int, int]],
    allowed: Optional[Set[str]],
    color_map: Optional[Dict[str, Tuple[int, int, int]]],
) -> str:
    code = _norm_code(code)
    if not code or bg_rgb is None or not color_map:
        return code
    pool = list(allowed) if allowed else list(color_map.keys())
    scored = []
    for c in pool:
        rgb = color_map.get(c)
        if not rgb:
            continue
        scored.append((_rgb_dist(bg_rgb, rgb), c))
    if not scored:
        return code
    scored.sort(key=lambda x: x[0])
    best_d, best_code = scored[0]
    cur_d = _rgb_dist(bg_rgb, color_map[code]) if code in color_map else None
    if cur_d is None:
        return best_code if best_d <= 55 else code
    if cur_d <= 42:
        return code
    if best_code != code and best_d + 28 <= cur_d and best_d <= 65:
        return best_code
    return code


def _clean_legend_text(text: str) -> str:
    norm = (text or "").upper()
    norm = norm.replace("，", " ").replace(",", " ")
    norm = norm.replace("：", ":").replace("（", "(").replace("）", ")")
    norm = norm.replace("【", "(").replace("】", ")")
    norm = norm.replace("θ", "0").replace("Θ", "0").replace("Ø", "0")
    norm = re.sub(r"([A-Z]{1,3})O([0-9])", r"\g<1>0\g<2>", norm)
    norm = re.sub(r"([A-Z]{1,3})I([0-9])", r"\g<1>1\g<2>", norm)
    norm = re.sub(r"([A-Z]{1,3})L([0-9])", r"\g<1>1\g<2>", norm)
    return norm


def _extract_legend_counts(line: str) -> List[int]:
    line = re.sub(r"(?<=\d)\.(?=\d)", "", line or "")
    return [int(x) for x in re.findall(r"\b[0-9]{1,5}\b", line) if int(x) > 0]


def _decode_data_url(data_url: str):
    if not data_url or "," not in data_url:
        return None, "missing image"
    try:
        head, b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
    except Exception:
        return None, "invalid base64"
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(raw))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img, None
    except Exception:
        return None, "invalid image"


_OCR_ENGINE = None
_OCR_ENGINE_NAME = None
_OCR_INIT_ERR = None
_PADDLE_OCR_ENGINE = None
_PADDLE_OCR_ERR = None


def _get_engine():
    global _OCR_ENGINE, _OCR_ENGINE_NAME, _OCR_INIT_ERR
    if _OCR_ENGINE is not None or _OCR_INIT_ERR is not None:
        return _OCR_ENGINE
    try:
        from rapidocr import RapidOCR

        _OCR_ENGINE = RapidOCR()
        _OCR_ENGINE_NAME = "rapidocr"
        return _OCR_ENGINE
    except Exception as e:
        _OCR_INIT_ERR = str(e)
        return None


def _run_ocr(img, use_det: bool = True, use_rec: bool = True):
    eng = _get_engine()
    if eng is None:
        return None, f"OCR engine unavailable: {_OCR_INIT_ERR or 'missing dependency'}"
    try:
        import numpy as np

        arr = np.array(img)
        out = eng(arr, use_det=use_det, use_cls=False, use_rec=use_rec, text_score=0.25)
        boxes = getattr(out, "boxes", None)
        txts = getattr(out, "txts", None)
        scores = getattr(out, "scores", None)
        if boxes is None or txts is None or scores is None:
            return [], None
        res = []
        for box, t, s in zip(list(boxes), list(txts), list(scores)):
            res.append((box, str(t or ""), float(s or 0)))
        return res, None
    except Exception as e:
        return None, str(e)


def _get_paddle_ocr():
    global _PADDLE_OCR_ENGINE, _PADDLE_OCR_ERR
    if _PADDLE_OCR_ENGINE is not None or _PADDLE_OCR_ERR is not None:
        return _PADDLE_OCR_ENGINE
    try:
        from paddleocr import PaddleOCR

        _PADDLE_OCR_ENGINE = PaddleOCR(
            ocr_version=os.environ.get("PINDOU_PADDLE_OCR_VERSION") or "PP-OCRv5",
            lang=os.environ.get("PINDOU_PADDLE_OCR_LANG") or "en",
            text_detection_model_name=os.environ.get("PINDOU_PADDLE_DET_MODEL") or "PP-OCRv5_mobile_det",
            text_recognition_model_name=os.environ.get("PINDOU_PADDLE_REC_MODEL") or "en_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=float(os.environ.get("PINDOU_PADDLE_SCORE_THRESH") or "0.08"),
            text_det_limit_side_len=int(os.environ.get("PINDOU_PADDLE_LIMIT_SIDE") or "1600"),
            text_det_limit_type=os.environ.get("PINDOU_PADDLE_LIMIT_TYPE") or "max",
            text_det_thresh=float(os.environ.get("PINDOU_PADDLE_DET_THRESH") or "0.15"),
            text_det_box_thresh=float(os.environ.get("PINDOU_PADDLE_BOX_THRESH") or "0.20"),
        )
        return _PADDLE_OCR_ENGINE
    except Exception as e:
        _PADDLE_OCR_ERR = str(e)
        return None


def _run_paddle_ocr(img):
    eng = _get_paddle_ocr()
    if eng is None:
        return None, f"PaddleOCR unavailable: {_PADDLE_OCR_ERR or 'missing dependency'}"
    try:
        import numpy as np

        arr = np.array(img)
        out = eng.predict(arr)
        res = []
        for page in out or []:
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            boxes = page.get("rec_boxes")
            polys = page.get("rec_polys") or page.get("dt_polys") or []
            for i, text in enumerate(texts):
                score = float(scores[i] if i < len(scores) else 0.0)
                box = None
                try:
                    if boxes is not None and i < len(boxes):
                        b = boxes[i]
                        box = [(float(b[0]), float(b[1])), (float(b[2]), float(b[1])), (float(b[2]), float(b[3])), (float(b[0]), float(b[3]))]
                except Exception:
                    box = None
                if box is None and i < len(polys):
                    try:
                        box = [(float(p[0]), float(p[1])) for p in polys[i]]
                    except Exception:
                        box = None
                if box is None:
                    continue
                res.append((box, str(text or ""), score))
        return res, None
    except Exception as e:
        return None, str(e)


def _whiten_grid_lines(img):
    try:
        import numpy as np
    except Exception:
        return img
    arr = np.array(img).astype("uint8")
    r = arr[:, :, 0].astype("int16")
    g = arr[:, :, 1].astype("int16")
    b = arr[:, :, 2].astype("int16")
    is_mag = (r > 150) & (b > 150) & ((g + 45) < np.minimum(r, b))
    is_red = (r > 170) & ((r - g) > 70) & ((r - b) > 50) & (g < 150) & (b < 150)
    m = is_mag | is_red
    if m.any():
        arr[m] = 250
    try:
        from PIL import Image

        return Image.fromarray(arr, mode="RGB")
    except Exception:
        return img


def _grid_ocr_strips(img, rows: int, cols: int, allowed: Optional[Set[str]]):
    w, h = img.size
    cw = float(w) / float(cols)
    ch = float(h) / float(rows)
    best: Dict[Tuple[int, int], Tuple[str, str, float]] = {}
    detected = 0
    for rr in range(rows):
        y0 = int(round(rr * ch))
        y1 = int(round((rr + 1) * ch))
        if y1 <= y0:
            continue
        strip = img.crop((0, y0, w, y1))
        strip = _whiten_grid_lines(strip)
        res, e = _run_ocr(strip, use_det=True, use_rec=True)
        if e:
            continue
        for box, t, s in res or []:
            detected += 1
            tt = str(t or "").upper()
            tt = re.sub(r"[^A-Z0-9]", "", tt)
            m = re.search(r"[A-Z]{1,3}0*[0-9]{1,3}", tt)
            if not m:
                continue
            raw_code = _norm_code(m.group(0))
            code = raw_code
            if allowed:
                code = code if code in allowed else (_closest_allowed(code, allowed) or "")
            if not code:
                continue
            xs = [p[0] for p in box]
            cx = sum(xs) / 4.0
            cc = int(cx / cw)
            if cc < 0 or cc >= cols:
                continue
            key = (rr, cc)
            prev = best.get(key)
            if prev is None or s > prev[2]:
                best[key] = (code, raw_code, float(s))
    cells = [{"r": rc[0], "c": rc[1], "code": v[0], "raw": v[1], "conf": v[2]} for rc, v in best.items()]
    return cells, detected


def _grid_code_from_text(text: str, allowed: Optional[Set[str]]) -> str:
    tt = str(text or "").upper()
    tt = tt.replace(" ", "")
    tt = re.sub(r"[^A-Z0-9]", "", tt)
    candidates = []
    for m in re.finditer(r"[A-Z]{1,3}0*[0-9]{1,3}", tt):
        raw = _norm_code(m.group(0))
        if raw:
            candidates.append(raw)
    if not candidates:
        return ""
    if allowed:
        for raw in candidates:
            if raw in allowed:
                return raw
        return ""
    return candidates[0]


def _ocr_single_cell(img, allowed: Optional[Set[str]]) -> Tuple[str, str, float]:
    try:
        from PIL import ImageOps, ImageEnhance, ImageFilter
    except Exception:
        return "", "", 0.0

    w, h = img.size
    if w <= 3 or h <= 3:
        return "", "", 0.0
    pad_x = max(1, int(round(w * 0.10)))
    pad_y = max(1, int(round(h * 0.10)))
    crop = img.crop((pad_x, pad_y, max(pad_x + 1, w - pad_x), max(pad_y + 1, h - pad_y)))
    crop = _whiten_grid_lines(crop)
    scale = max(3, min(7, int(round(96 / max(1, min(crop.size))))))
    crop = crop.resize((crop.size[0] * scale, crop.size[1] * scale))

    variants = []
    variants.append(crop)
    gray = ImageOps.grayscale(crop)
    gray = ImageEnhance.Contrast(gray).enhance(2.2)
    variants.append(gray.convert("RGB"))
    variants.append(ImageOps.invert(gray).convert("RGB"))
    # Binarized variants help light text on dark cells and tiny black text on pale cells.
    for inv in (False, True):
        g = ImageOps.invert(gray) if inv else gray
        g = g.filter(ImageFilter.SHARPEN)
        bw = g.point(lambda p: 255 if p > 150 else 0)
        variants.append(bw.convert("RGB"))

    best_code, best_raw, best_conf = "", "", 0.0
    for im in variants:
        for use_det in (False, True):
            res, e = _run_ocr(im, use_det=use_det, use_rec=True)
            if e or not res:
                continue
            for _box, text, score in res:
                raw = _norm_code(str(text or ""))
                code = _grid_code_from_text(raw or text, allowed)
                if not code:
                    continue
                sc = float(score or 0.0)
                if sc > best_conf:
                    best_code, best_raw, best_conf = code, raw or str(text or ""), sc
    return best_code, best_raw, best_conf


def _grid_ocr_cells(img, rows: int, cols: int, allowed: Optional[Set[str]], candidates: Optional[List[Any]] = None):
    w, h = img.size
    cw = float(w) / float(cols)
    ch = float(h) / float(rows)
    targets: List[Tuple[int, int]] = []
    if isinstance(candidates, list) and candidates:
        seen = set()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            try:
                rr = int(item.get("r", item.get("row")))
                cc = int(item.get("c", item.get("col")))
            except Exception:
                continue
            if rr < 0 or cc < 0 or rr >= rows or cc >= cols:
                continue
            key = (rr, cc)
            if key in seen:
                continue
            seen.add(key)
            targets.append(key)
    else:
        targets = [(r, c) for r in range(rows) for c in range(cols)]
    max_cells = int(os.environ.get("PINDOU_GRID_CELL_OCR_MAX") or "1600")
    targets = targets[:max_cells]
    cells = []
    scanned = 0
    for rr, cc in targets:
        x0 = int(round(cc * cw))
        x1 = int(round((cc + 1) * cw))
        y0 = int(round(rr * ch))
        y1 = int(round((rr + 1) * ch))
        if x1 <= x0 or y1 <= y0:
            continue
        code, raw, conf = _ocr_single_cell(img.crop((x0, y0, x1, y1)), allowed)
        scanned += 1
        if code:
            cells.append({"r": rr, "c": cc, "code": code, "raw": raw, "conf": conf})
    return cells, scanned


def _candidate_set(candidates: Optional[List[Any]], rows: int, cols: int) -> Optional[Set[Tuple[int, int]]]:
    if not isinstance(candidates, list) or not candidates:
        return None
    out: Set[Tuple[int, int]] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        try:
            rr = int(item.get("r", item.get("row")))
            cc = int(item.get("c", item.get("col")))
        except Exception:
            continue
        if 0 <= rr < rows and 0 <= cc < cols:
            out.add((rr, cc))
    return out or None


def _grid_ocr_paddle(img, rows: int, cols: int, allowed: Optional[Set[str]], candidates: Optional[List[Any]] = None):
    img = _whiten_grid_lines(img)
    res, e = _run_paddle_ocr(img)
    if e:
        return None, 0, e
    w, h = img.size
    cw = float(w) / float(cols)
    ch = float(h) / float(rows)
    cand = _candidate_set(candidates, rows, cols)
    best: Dict[Tuple[int, int], Tuple[str, str, float]] = {}
    detected = 0
    for box, text, score in res or []:
        detected += 1
        code = _grid_code_from_text(text, allowed)
        if not code:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cx = sum(xs) / max(1, len(xs))
        cy = sum(ys) / max(1, len(ys))
        rr = int(cy / ch)
        cc = int(cx / cw)
        if rr < 0 or cc < 0 or rr >= rows or cc >= cols:
            continue
        if cand is not None and (rr, cc) not in cand:
            # If the detector lands slightly off-center, allow immediate neighbors.
            found = None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    key2 = (rr + dr, cc + dc)
                    if key2 in cand:
                        found = key2
                        break
                if found:
                    break
            if not found:
                continue
            rr, cc = found
        key = (rr, cc)
        prev = best.get(key)
        if prev is None or score > prev[2]:
            best[key] = (code, text, float(score))
    cells = [{"r": rc[0], "c": rc[1], "code": v[0], "raw": v[1], "conf": v[2]} for rc, v in best.items()]
    return cells, detected, None


def _boxes_to_text(res):
    if not res:
        return ""
    items = []
    for box, t, s in res:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cx = sum(xs) / 4.0
        cy = sum(ys) / 4.0
        items.append((cy, cx, t, s))
    items.sort(key=lambda x: (x[0], x[1]))
    lines = []
    cur = []
    cur_y = None
    for y, x, t, s in items:
        if cur_y is None:
            cur_y = y
            cur = [(x, t)]
            continue
        if abs(y - cur_y) > 16:
            cur.sort(key=lambda z: z[0])
            lines.append(" ".join([z[1] for z in cur]).strip())
            cur_y = y
            cur = [(x, t)]
        else:
            cur.append((x, t))
    if cur:
        cur.sort(key=lambda z: z[0])
        lines.append(" ".join([z[1] for z in cur]).strip())
    return "\n".join([ln for ln in lines if ln])


def _parse_legend_from_text(text: str, allowed: Optional[Set[str]]):
    if not text:
        return []
    norm = _clean_legend_text(text)
    items: List[Dict[str, Any]] = []

    re1 = re.compile(r"\b([A-Z]{1,3}0*[0-9]{1,3})\s*[:\(]\s*([0-9]{1,5})\s*\)?")
    for code, cnt in re1.findall(norm):
        c = _valid_or_close(code, allowed)
        n = int(cnt)
        if c and n > 0:
            items.append({"code": c, "count": n, "conf": 0.95, "source": "paren"})

    lines = [ln.strip() for ln in re.split(r"\n+", norm) if ln.strip()]
    re2 = re.compile(r"\b([A-Z]{1,3}0*[0-9]{1,3})\s+([0-9]{1,5})\b")
    for line in lines:
        if "(" not in line and ")" not in line and ":" not in line:
            for code, cnt in re2.findall(line):
                c = _valid_or_close(code, allowed)
                n = int(cnt)
                if c and n > 0:
                    items.append({"code": c, "count": n, "conf": 0.82, "source": "same-line"})
            for token in re.findall(r"\b[A-Z]{1,3}[0-9]{2,8}\b", line):
                split = _split_glued_code_count(token, allowed)
                if split:
                    c, n = split
                    items.append({"code": c, "count": n, "conf": 0.78, "source": "glued"})

    for i in range(len(lines) - 1):
        if any(ch in lines[i] for ch in "()[]:"):
            continue
        code_tokens = re.findall(r"\b[A-Z0-9]{2,8}\b", lines[i])
        codes: List[str] = []
        for tok in code_tokens:
            if tok.isdigit():
                continue
            c = _valid_or_close(tok, allowed)
            if c:
                codes.append(c)
        if len(codes) < 3:
            continue
        nums: List[int] = []
        for j in range(i + 1, min(len(lines), i + 4)):
            if re.search(r"[A-Z]{1,3}0*[0-9]{1,3}", lines[j]) and any(ch in lines[j] for ch in "()[]:"):
                break
            nums.extend(_extract_legend_counts(lines[j]))
            if len(nums) >= len(codes):
                break
        if len(codes) >= 3 and len(nums) == len(codes):
            for c, n in zip(codes, nums):
                items.append({"code": c, "count": n, "conf": 0.88, "source": "two-row"})

    return _merge_legend_items(items)


def _ocr_count_below_code(img, code_word: Dict[str, Any]) -> Optional[int]:
    try:
        from PIL import ImageOps
    except Exception:
        return None
    try:
        w, h = img.size
        bw = max(10.0, float(code_word.get("right", 0)) - float(code_word.get("left", 0)))
        cell_w = float(code_word.get("cell_w_est") or 0)
        bh = max(8.0, float(code_word.get("bottom", 0)) - float(code_word.get("top", 0)))
        cx = float(code_word.get("cx", 0))
        y0 = int(max(0, float(code_word.get("bottom", 0)) + bh * 0.05))
        y1 = int(min(h, float(code_word.get("bottom", 0)) + bh * 3.2))
        half_w = max(bw * 0.95, cell_w * 0.44, 18.0)
        x0 = int(max(0, cx - half_w))
        x1 = int(min(w, cx + half_w))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = img.crop((x0, y0, x1, y1))
        scale = 4
        crop = crop.resize((max(1, crop.width * scale), max(1, crop.height * scale)))
        gray = ImageOps.grayscale(crop)
        # Keep only dark digit strokes. Gray watermarks and colored blocks fade out.
        best: List[Tuple[float, int]] = []
        for th in (90, 110, 130, 150):
            bw_img = gray.point(lambda p, t=th: 0 if p < t else 255).convert("RGB")
            res, e = _run_ocr(bw_img, use_det=True, use_rec=True)
            if e:
                continue
            for _box, text, score in res or []:
                nums = _extract_legend_counts(str(text or ""))
                for n in nums:
                    if 0 < n <= 5000:
                        best.append((float(score or 0.0), n))
        if not best:
            return None
        best.sort(key=lambda x: (len(str(x[1])), x[0], x[1]), reverse=True)
        return int(best[0][1])
    except Exception:
        return None


def _parse_legend_from_boxes(res, allowed: Optional[Set[str]], img=None, color_map: Optional[Dict[str, Tuple[int, int, int]]] = None):
    if not res:
        return []
    words = []
    for box, t, s in res:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        text = _clean_legend_text(str(t or "")).strip()
        if not text:
            continue
        words.append({
            "text": text,
            "box": box,
            "left": min(xs),
            "right": max(xs),
            "top": min(ys),
            "bottom": max(ys),
            "cx": sum(xs) / 4.0,
            "cy": sum(ys) / 4.0,
            "h": max(1.0, max(ys) - min(ys)),
            "score": float(s or 0.0),
        })
    items: List[Dict[str, Any]] = []
    code_words = []
    num_words = []
    pair_words = words
    if len(words) >= 20:
        heights = sorted([w["h"] for w in words])
        med_h = heights[len(heights) // 2] if heights else 1.0
        min_y = min(w["cy"] for w in words)
        max_y = max(w["cy"] for w in words)
        if (max_y - min_y) > med_h * 8:
            floor_y = min_y + (max_y - min_y) * 0.45
            pair_words = [w for w in words if w["cy"] >= floor_y]

    legend_text_codes: Set[str] = set()
    for w in pair_words:
        compact0 = re.sub(r"\s+", "", w["text"])
        c0 = _valid_or_close(compact0, allowed)
        if c0:
            legend_text_codes.add(c0)

    for w in pair_words:
        compact = re.sub(r"\s+", "", w["text"])
        if re.fullmatch(r"[A-Z]{1,3}[0-9]{3,8}", compact):
            bg_rgb = _sample_box_bg_rgb(img, w.get("box")) if img is not None and w.get("box") is not None else None
            split = _split_glued_code_count(compact, allowed, color_map, bg_rgb)
            if split:
                c, n = split
                items.append({"code": c, "count": n, "conf": max(0.70, w["score"]), "source": "box-glued"})
                continue
        direct = _parse_legend_from_text(w["text"], allowed)
        if direct:
            for it in direct:
                it = dict(it)
                it["conf"] = max(float(it.get("conf") or 0.0), w["score"])
                it["source"] = it.get("source") or "box-direct"
                items.append(it)
            continue
        c = _valid_or_close(compact, allowed)
        if c:
            bg_rgb = _sample_box_bg_rgb(img, w.get("box")) if img is not None and w.get("box") is not None else None
            if c not in legend_text_codes:
                c = _color_correct_code(c, bg_rgb, legend_text_codes or allowed, color_map)
            code_words.append({**w, "code": c})
            continue
        num = compact.strip("()[]")
        if re.fullmatch(r"\d{1,5}", num) and int(num) > 0:
            num_words.append({**w, "count": int(num)})
            continue
        bg_rgb = _sample_box_bg_rgb(img, w.get("box")) if img is not None and w.get("box") is not None else None
        split = _split_glued_code_count(compact, allowed, color_map, bg_rgb)
        if split:
            c, n = split
            items.append({"code": c, "count": n, "conf": max(0.70, w["score"]), "source": "box-glued"})

    if len(code_words) >= 3:
        centers = sorted([float(w["cx"]) for w in code_words])
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1) if 4 <= centers[i + 1] - centers[i] <= 260]
        if gaps:
            gaps.sort()
            cell_w_est = gaps[len(gaps) // 2]
            for w in code_words:
                w["cell_w_est"] = cell_w_est

    pairs = []
    for ci, c in enumerate(code_words):
        for ni, n in enumerate(num_words):
            dy = abs(n["cy"] - c["cy"])
            h = max(c["h"], n["h"])
            if dy > h * 1.4:
                continue
            dx = n["left"] - c["right"]
            if dx < -c["h"] * 0.6 or dx > c["h"] * 10:
                continue
            pairs.append((max(0, dx) + dy * 2.0, ci, ni))
    pairs.sort()
    used_c = set()
    used_n = set()
    for _, ci, ni in pairs:
        if ci in used_c or ni in used_n:
            continue
        used_c.add(ci)
        used_n.add(ni)
        c = code_words[ci]
        n = num_words[ni]
        items.append({"code": c["code"], "count": n["count"], "conf": max(c["score"], n["score"], 0.72), "source": "box-same-row"})

    for ci, c in enumerate(code_words):
        if ci in used_c:
            continue
        best = None
        for ni, n in enumerate(num_words):
            if ni in used_n:
                continue
            if abs(n["cx"] - c["cx"]) > max(c["h"], n["h"]) * 3:
                continue
            dy = n["top"] - c["bottom"]
            if dy < -c["h"] * 0.2 or dy > c["h"] * 8:
                continue
            candidate = (dy, ni, n)
            if best is None or candidate < best:
                best = candidate
        if best:
            _, ni, n = best
            used_n.add(ni)
            items.append({"code": c["code"], "count": n["count"], "conf": max(c["score"], n["score"], 0.72), "source": "box-below"})

    if img is not None:
        for ci, c in enumerate(code_words):
            count = _ocr_count_below_code(img, c)
            if count and count > 0:
                items.append({"code": c["code"], "count": count, "conf": max(c["score"], 0.86), "source": "box-crop-below"})

    return _merge_legend_items(items)


def _extract_json_text(s: str) -> Any:
    raw = (s or "").strip()
    if not raw:
        raise ValueError("empty model response")
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError("model response is not JSON")


def _parse_vlm_text_items(text: str, allowed: Optional[Set[str]]) -> List[Dict[str, Any]]:
    norm = _clean_legend_text(text)
    norm = re.sub(r"[\[\]{}]", " ", norm)
    norm = re.sub(r"[()]+", " ", norm)
    norm = re.sub(r"[:：=,，;；/|]+", " ", norm)
    items: List[Dict[str, Any]] = []

    for m in re.finditer(r"\b([A-Z]{1,3}\d{1,3})\b\s+([0-9]{1,5})\b", norm):
        raw_code = _norm_code(m.group(1))
        code = _valid_or_close(raw_code, allowed) if allowed else raw_code
        if not code:
            continue
        try:
            count = int(m.group(2))
        except Exception:
            continue
        if count <= 0:
            continue
        items.append({"code": code, "count": count, "conf": 0.85, "source": "vlm-text"})

    if not items:
        compact = re.sub(r"\s+", "", norm)
        for m in re.finditer(r"\b([A-Z]{1,3})(\d{2,8})\b", compact):
            split = _split_glued_code_count(m.group(0), allowed, None, None)
            if not split:
                continue
            code, count = split
            items.append({"code": code, "count": count, "conf": 0.65, "source": "vlm-text-glued"})

    lines = [ln.strip() for ln in re.split(r"\n+", norm) if ln.strip()]
    for i in range(len(lines) - 1):
        code_tokens = []
        for tok in re.findall(r"\b[A-Z]{1,3}0*[0-9]{1,3}\b", lines[i]):
            c = _valid_or_close(tok, allowed)
            if c:
                code_tokens.append(c)
        if len(code_tokens) < 3:
            continue
        nums = _extract_legend_counts(lines[i + 1])
        # Top/bottom legends are commonly laid out as:
        # code code code ...
        # count count count ...
        # Pair by visual column/order, but only when the next line is mostly numbers.
        if len(nums) < len(code_tokens):
            continue
        if re.search(r"[A-Z]{1,3}0*[0-9]{1,3}", lines[i + 1]):
            continue
        for c, n in zip(code_tokens, nums):
            if n <= 0:
                continue
            items.append({"code": c, "count": int(n), "conf": 0.78, "source": "vlm-text-two-row"})

    return _merge_legend_items(items)


def _coerce_vlm_items(data: Any, allowed: Optional[Set[str]]) -> List[Dict[str, Any]]:
    if isinstance(data, str):
        return _parse_vlm_text_items(data, allowed)
    if isinstance(data, dict):
        if isinstance(data.get("code"), list) and isinstance(data.get("count"), list):
            rows = [
                {"code": c, "count": data.get("count")[i], "confidence": 0.8}
                for i, c in enumerate(data.get("code"))
                if i < len(data.get("count"))
            ]
        else:
            rows = data.get("items") if isinstance(data.get("items"), list) else data.get("colors")
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    if not isinstance(rows, list):
        rows = []
    items: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        code = _norm_code(str(row.get("code") or row.get("色号") or row.get("label") or ""))
        code = _valid_or_close(code, allowed) if code else ""
        try:
            count = int(str(row.get("count", row.get("qty", row.get("数量", 0)))).replace(",", "").strip())
        except Exception:
            count = 0
        if not code or count <= 0:
            continue
        try:
            conf = float(row.get("confidence", row.get("conf", 0.9)))
        except Exception:
            conf = 0.9
        items.append({"code": code, "count": count, "conf": conf, "source": "vlm"})
    return _merge_legend_items(items)


def _data_url_to_b64(data_url: str) -> str:
    s = data_url or ""
    if "," in s and s.lower().startswith("data:"):
        return s.split(",", 1)[1]
    return s


def _normalize_glm_model(model: str) -> str:
    name = (model or "").strip()
    aliases = {
        "4v": "glm-4v-flash",
        "glm-4v": "glm-4v-flash",
        "glm-4v-flash": "glm-4v-flash",
        "4.6v": "glm-4.6v-flash",
        "glm-4.6v": "glm-4.6v-flash",
        "glm-4.6v-flash": "glm-4.6v-flash",
    }
    return aliases.get(name.lower(), name or "glm-4v-flash")


def _glm_config(model_override: Optional[str] = None) -> Tuple[str, str, str, Optional[str]]:
    base_url = (os.environ.get("PINDOU_GLM_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
    model = _normalize_glm_model(model_override or os.environ.get("PINDOU_GLM_MODEL") or "glm-4v-flash")
    api_key = (os.environ.get("PINDOU_GLM_API_KEY") or "").strip()
    if not api_key:
        return base_url, model, api_key, "GLM API Key 未配置。请在 local_ai_server/vlm.env 设置 PINDOU_GLM_API_KEY 后重启本地识别服务。"
    return base_url, model, api_key, None


def _vlm_max_tokens(model: str, default: int = 4096) -> int:
    name = (model or "").lower()
    if "glm-4v" in name or "glm-4.6v" in name:
        return min(default, 1024)
    return default


def _run_vlm_legend(
    data_url: str,
    allowed: Optional[Set[str]],
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    base_url = (base_url or os.environ.get("PINDOU_VLM_BASE_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
    model = (model or os.environ.get("PINDOU_VLM_MODEL") or "").strip()
    api_key = (api_key if api_key is not None else os.environ.get("PINDOU_VLM_API_KEY") or "").strip()
    if not model:
        return None, "AI视觉模型未配置。请先设置 PINDOU_VLM_MODEL；本地 Ollama/LM Studio 可同时设置 PINDOU_VLM_BASE_URL。"
    allowed_list = sorted(list(allowed or []))
    allowed_hint = "、".join(allowed_list[:500])
    prompt = (
        "你在识别拼豆图纸底部/顶部的图例区。目标是读取每一个图例色块，不是读取大图网格。\n"
        "请先把图例按“一个色块/一个胶囊矩形”为单位切分；每个单位只输出一条 {code,count}。\n"
        "常见图例结构如下，必须按这些规则配对：\n"
        "A. 左右结构：同一色块里左侧/中间是色号，右侧或括号内是数量，例如 A11 (26)、C24 205、H2 130。\n"
        "B. 上下结构：同一色块上方是色号，正下方是数量；数量可能在色块外紧贴下方。必须按同一列、同一个色块上下配对。\n"
        "C. 多行上下结构：一排色块的色号在上，下一行对应的数字在下。请按每个色块的水平中心一一对应，不能把下一排数字整体错位。\n"
        "D. 混合结构：有些色块是左右结构，有些是上下结构。请逐个色块判断，不要套用整张图的一种格式。\n"
        "重要约束：\n"
        "- 不要把相邻两个色块的文字拼在一起，例如 C20 下方的 1 不能和 C21 下方的 9 拼成 19。\n"
        "- 不要把色号和隔壁色块的数量配对；数量必须来自该色块内部或正下方最近的数字。\n"
        "- 不要根据候选色号列表猜测；看不清就跳过。\n"
        "- 不要识别大图网格里的色号，不要识别水印、坐标、标题、总豆数。\n"
        "- 输出的 count 必须是该色块的需求数量，不是坐标、不是总数。\n"
        "只输出 JSON 数组，不要解释。格式必须是：[{\"code\":\"A11\",\"count\":26},{\"code\":\"C24\",\"count\":205},{\"code\":\"H2\",\"count\":130}]\n"
        "结果按图例色块从左到右、从上到下排列。\n"
    )
    if allowed_hint:
        prompt += f"可能出现的真实色号有：{allowed_hint}\n"
    if "127.0.0.1:11434" in base_url or "localhost:11434" in base_url:
        native_base = re.sub(r"/v1/?$", "", base_url)
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [_data_url_to_b64(data_url)],
            }],
        }
        req = urlrequest.Request(
            f"{native_base}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=240) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            msg = e.read().decode("utf-8", "ignore")[:500]
            return None, f"AI视觉模型请求失败：HTTP {e.code} {msg}"
        except URLError as e:
            return None, f"无法连接 AI视觉模型服务：{e.reason}"
        except Exception as e:
            return None, f"AI视觉模型调用失败：{e}"
        try:
            content = obj.get("message", {}).get("content", "")
            try:
                parsed = _extract_json_text(str(content))
                return _coerce_vlm_items(parsed, allowed), None
            except Exception:
                return _parse_vlm_text_items(str(content), allowed), None
        except Exception as e:
            return None, f"AI视觉模型返回格式无法解析：{e}"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": _vlm_max_tokens(model, 4096),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlrequest.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=240) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")[:500]
        return None, f"AI视觉模型请求失败：HTTP {e.code} {msg}"
    except URLError as e:
        return None, f"无法连接 AI视觉模型服务：{e.reason}"
    except Exception as e:
        return None, f"AI视觉模型调用失败：{e}"
    try:
        content = obj["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
        parsed = _extract_json_text(str(content))
        return _coerce_vlm_items(parsed, allowed), None
    except Exception as e:
        return None, f"AI视觉模型返回格式无法解析：{e}"


def _coerce_vlm_grid_cells(data: Any, rows: int, cols: int, allowed: Optional[Set[str]]) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []

    def add_cell(r: Any, c: Any, code: Any, conf: Any = 0.85):
        try:
            rr = int(r)
            cc = int(c)
        except Exception:
            return
        if rr < 0 or cc < 0 or rr >= rows or cc >= cols:
            return
        raw = _norm_code(str(code or ""))
        if not raw:
            return
        code2 = raw
        if allowed and code2 not in allowed:
            return
        if not code2:
            return
        try:
            cf = float(conf)
        except Exception:
            cf = 0.85
        cells.append({"r": rr, "c": cc, "code": code2, "raw": raw, "conf": cf})

    if isinstance(data, dict):
        if isinstance(data.get("cells"), list):
            for row in data.get("cells") or []:
                if not isinstance(row, dict):
                    continue
                add_cell(row.get("r", row.get("row")), row.get("c", row.get("col")), row.get("code"), row.get("confidence", row.get("conf", 0.85)))
        elif isinstance(data.get("rows"), list):
            data = data.get("rows")
        elif isinstance(data.get("grid"), list):
            data = data.get("grid")

    if isinstance(data, list):
        if all(isinstance(row, dict) and ("c" in row or "col" in row) and "code" in row for row in data):
            for row in data:
                add_cell(row.get("r", row.get("row", 0)), row.get("c", row.get("col")), row.get("code"), row.get("confidence", row.get("conf", 0.85)))
        else:
            for r, row in enumerate(data[:rows]):
                if isinstance(row, dict):
                    vals = row.get("cells") or row.get("values") or row.get("cols")
                    if isinstance(vals, list):
                        row = vals
                    else:
                        for c, v in row.items():
                            add_cell(r, c, v)
                        continue
                if isinstance(row, list):
                    for c, code in enumerate(row[:cols]):
                        add_cell(r, c, code)

    best: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for cell in cells:
        key = (cell["r"], cell["c"])
        prev = best.get(key)
        if prev is None or cell.get("conf", 0) > prev.get("conf", 0):
            best[key] = cell
    return list(best.values())


def _coerce_vlm_grid_rows(data: Any, cols: int, allowed: Optional[Set[str]]) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []

    def add_col(c: Any, code: Any, conf: Any = 0.85):
        try:
            cc = int(c)
        except Exception:
            return
        if cc < 0 or cc >= cols:
            return
        raw = _norm_code(str(code or ""))
        if not raw:
            return
        code2 = _valid_or_close(raw, allowed) if allowed else raw
        if not code2:
            return
        try:
            cf = float(conf)
        except Exception:
            cf = 0.85
        cells.append({"r": 0, "c": cc, "code": code2, "raw": raw, "conf": cf})

    if isinstance(data, dict):
        data = data.get("cells") if isinstance(data.get("cells"), list) else data.get("items")
    if isinstance(data, list):
        if all(isinstance(row, dict) for row in data):
            for row in data:
                add_col(row.get("c", row.get("col")), row.get("code"), row.get("confidence", row.get("conf", 0.85)))
        else:
            for c, code in enumerate(data[:cols]):
                add_col(c, code)
    best: Dict[int, Dict[str, Any]] = {}
    for cell in cells:
        prev = best.get(cell["c"])
        if prev is None or cell.get("conf", 0) > prev.get("conf", 0):
            best[cell["c"]] = cell
    return list(best.values())


def _run_vlm_grid_row(data_url: str, cols: int, allowed: Optional[Set[str]]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    base_url = (os.environ.get("PINDOU_VLM_BASE_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
    model = (os.environ.get("PINDOU_VLM_MODEL") or "").strip()
    if not model:
        return None, "AI视觉模型未配置。请先设置 PINDOU_VLM_MODEL。"
    allowed_hint = "、".join(sorted(list(allowed or []))[:500])
    prompt = (
        f"这是一行拼豆网格，共 {cols} 列，列号从左到右是 0 到 {cols - 1}。\n"
        "请读取每个格子里的色号文字，只列出有色号的格子。\n"
        "白底黑字、浅色底黑字也算有效色号，不是空白格。\n"
        "真正没有文字、只有水印/网格线/坐标的格子不要输出。\n"
        "输出 JSON 数组，每项格式 {\"c\":列号,\"code\":\"色号\"}。不要解释。\n"
    )
    if allowed_hint:
        prompt += f"色号只能从这些真实色号中选择：{allowed_hint}\n"

    if "127.0.0.1:11434" in base_url or "localhost:11434" in base_url:
        native_base = re.sub(r"/v1/?$", "", base_url)
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [_data_url_to_b64(data_url)],
            }],
        }
        req = urlrequest.Request(
            f"{native_base}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=240) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            content = str(obj.get("message", {}).get("content", ""))
            try:
                parsed = _extract_json_text(content)
                return _coerce_vlm_grid_rows(parsed, cols, allowed), None
            except Exception:
                return _parse_vlm_grid_text(content, 1, cols, allowed), None
        except HTTPError as e:
            msg = e.read().decode("utf-8", "ignore")[:500]
            return None, f"AI网格识别请求失败：HTTP {e.code} {msg}"
        except URLError as e:
            return None, f"无法连接 AI视觉模型服务：{e.reason}"
        except Exception as e:
            return None, f"AI网格识别失败：{e}"

    return None, "当前只支持本地 Ollama 的 AI 网格识别。"


def _coerce_vlm_cell_batch(data: Any, count: int, allowed: Optional[Set[str]]) -> List[Dict[str, Any]]:
    rows = data
    if isinstance(data, dict):
        rows = data.get("cells") or data.get("items") or data.get("results")
    if not isinstance(rows, list):
        rows = []
    cells: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("i", row.get("index", row.get("id"))))
        except Exception:
            continue
        if idx < 0 or idx >= count:
            continue
        raw_text = str(row.get("code") if row.get("code") is not None else row.get("色号") if row.get("色号") is not None else "")
        raw = _norm_code(raw_text)
        code = raw
        if code and allowed and code not in allowed:
            continue
        try:
            conf = float(row.get("confidence", row.get("conf", 0.9)))
        except Exception:
            conf = 0.9
        cells.append({"i": idx, "code": code, "raw": raw, "conf": conf})
    best: Dict[int, Dict[str, Any]] = {}
    for cell in cells:
        prev = best.get(cell["i"])
        if prev is None or cell.get("conf", 0) > prev.get("conf", 0):
            best[cell["i"]] = cell
    return list(best.values())


def _run_vlm_cell_batch(
    data_url: str,
    count: int,
    allowed: Optional[Set[str]],
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    base_url = (base_url or os.environ.get("PINDOU_VLM_BASE_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
    model = (model or os.environ.get("PINDOU_VLM_MODEL") or "").strip()
    if not model:
        return None, "AI视觉模型未配置。请先设置 PINDOU_VLM_MODEL。"
    allowed_hint = "、".join(sorted(list(allowed or []))[:500])
    prompt = (
        f"这是一组拼豆图纸网格小格，共 {count} 个编号。每个格子左上角有蓝底白字编号。\n"
        "这些小格可能来自同一块连续区域；请结合上下文判断灰色斜字水印，不要把水印当色号。\n"
        "请逐个检查每个编号格子里真正印刷的色号文字，每个编号都必须输出一项。\n"
        "真正没有色号文字的空白格，code 输出空字符串 \"\"。\n"
        "如果某个编号格子里没有清晰的“字母+数字”组合，code 必须输出空字符串。\n"
        "只能根据格子里的文字识别色号，不要根据底色、相邻格、常见颜色去猜。\n"
        "白底黑字、浅色底黑字也算有效色号。水印、坐标、网格线不是色号。\n"
        "如果文字看不清或不确定，code 必须输出空字符串，不要猜。\n"
        "只输出 JSON 数组，每项格式 {\"i\":编号,\"code\":\"色号\"}。不要解释。\n"
    )
    if allowed_hint:
        prompt += f"色号只能从这些真实色号中选择：{allowed_hint}\n"

    if "127.0.0.1:11434" in base_url or "localhost:11434" in base_url:
        native_base = re.sub(r"/v1/?$", "", base_url)
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [_data_url_to_b64(data_url)],
            }],
        }
        req = urlrequest.Request(
            f"{native_base}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=240) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            content = str(obj.get("message", {}).get("content", ""))
            parsed = _extract_json_text(content)
            return _coerce_vlm_cell_batch(parsed, count, allowed), None
        except HTTPError as e:
            msg = e.read().decode("utf-8", "ignore")[:500]
            return None, f"AI格子识别请求失败：HTTP {e.code} {msg}"
        except URLError as e:
            return None, f"无法连接 AI视觉模型服务：{e.reason}"
        except Exception as e:
            return None, f"AI格子识别失败：{e}"

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": _vlm_max_tokens(model, 4096),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    }
    headers = {"Content-Type": "application/json"}
    api_key = (api_key if api_key is not None else os.environ.get("PINDOU_VLM_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urlrequest.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=240) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        content = obj["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
        parsed = _extract_json_text(str(content))
        return _coerce_vlm_cell_batch(parsed, count, allowed), None
    except HTTPError as e:
        msg = e.read().decode("utf-8", "ignore")[:500]
        return None, f"AI格子识别请求失败：HTTP {e.code} {msg}"
    except URLError as e:
        return None, f"无法连接 AI视觉模型服务：{e.reason}"
    except Exception as e:
        return None, f"AI格子识别失败：{e}"


def _parse_vlm_grid_text(text: str, rows: int, cols: int, allowed: Optional[Set[str]]) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines:
        m = re.match(r"^(?:row|r|第)?\s*([0-9]{1,3})\s*(?:行|[:：\-])?\s*(.*)$", ln, flags=re.I)
        if m:
            r = int(m.group(1))
            rest = m.group(2)
            if r >= rows and 1 <= r <= rows:
                r -= 1
        else:
            r = len({x["r"] for x in cells})
            rest = ln
        if r < 0 or r >= rows:
            continue
        tokens = re.split(r"[\s,，|;；]+", rest)
        c = 0
        for tok in tokens:
            if c >= cols:
                break
            t = tok.strip().upper()
            if not t:
                continue
            if t in {"_", "-", "空", "空白", "BLANK", "NONE", "NULL"}:
                c += 1
                continue
            raw = _norm_code(re.sub(r"[^A-Z0-9]", "", t))
            code = _valid_or_close(raw, allowed) if raw else ""
            if code:
                cells.append({"r": r, "c": c, "code": code, "raw": raw, "conf": 0.72})
            c += 1
    return cells


def _run_vlm_grid(data_url: str, rows: int, cols: int, allowed: Optional[Set[str]]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    base_url = (os.environ.get("PINDOU_VLM_BASE_URL") or "http://127.0.0.1:11434/v1").rstrip("/")
    model = (os.environ.get("PINDOU_VLM_MODEL") or "").strip()
    if not model:
        return None, "AI视觉模型未配置。请先设置 PINDOU_VLM_MODEL。"
    allowed_hint = "、".join(sorted(list(allowed or []))[:500])
    prompt = (
        f"这是一块拼豆图纸网格区域，共 {rows} 行、{cols} 列。\n"
        "图片上叠加了蓝底白字的局部行号/列号，左上角格子是 r=0,c=0。\n"
        "请读取每个格子里真正印刷的色号文字，只输出有色号的格子。\n"
        "灰色斜向水印、红色/橙色网格线、坐标数字、蓝底白字行列号都不是色号，必须忽略。\n"
        "空白格、只有水印的格子、看不清或不确定的格子不要输出。\n"
        "可以利用周围格子判断哪些灰字是连续水印；不要根据底色或相邻格猜色号。\n"
        "只输出 JSON 数组，每项格式 {\"r\":行号,\"c\":列号,\"code\":\"色号\"}。不要解释。\n"
    )
    if allowed_hint:
        prompt += f"色号只能从这些真实色号中选择：{allowed_hint}\n"

    if "127.0.0.1:11434" in base_url or "localhost:11434" in base_url:
        native_base = re.sub(r"/v1/?$", "", base_url)
        payload = {
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [_data_url_to_b64(data_url)],
            }],
        }
        req = urlrequest.Request(
            f"{native_base}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=240) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            content = str(obj.get("message", {}).get("content", ""))
            try:
                parsed = _extract_json_text(content)
                return _coerce_vlm_grid_cells(parsed, rows, cols, allowed), None
            except Exception:
                return _parse_vlm_grid_text(content, rows, cols, allowed), None
        except HTTPError as e:
            msg = e.read().decode("utf-8", "ignore")[:500]
            return None, f"AI网格识别请求失败：HTTP {e.code} {msg}"
        except URLError as e:
            return None, f"无法连接 AI视觉模型服务：{e.reason}"
        except Exception as e:
            return None, f"AI网格识别失败：{e}"

    return None, "当前只支持本地 Ollama 的 AI 网格识别。"




class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/health":
            eng = _get_engine()
            ok = eng is not None
            return _json_response(self, 200, {"ok": ok, "service": "pindou-local-ai", "engine": _OCR_ENGINE_NAME, "error": _OCR_INIT_ERR})
        return _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        p = urlparse(self.path).path
        body = _read_json(self)

        if p == "/legend":
            img, err = _decode_data_url(body.get("image") or "")
            if err:
                return _json_response(self, 400, {"error": err})
            valid = body.get("validCodes") or None
            allowed = set([_norm_code(x) for x in valid]) if isinstance(valid, list) else None
            color_map: Dict[str, Tuple[int, int, int]] = {}
            for item in body.get("validColors") or []:
                if not isinstance(item, dict):
                    continue
                code = _norm_code(str(item.get("code") or ""))
                rgb = _hex_to_rgb(str(item.get("hex") or ""))
                if code and rgb:
                    color_map[code] = rgb
            t0 = time.time()
            res, e = _run_ocr(img, use_det=True, use_rec=True)
            if e:
                return _json_response(self, 500, {"error": e})
            text = _boxes_to_text(res)
            raw_items = _parse_legend_from_boxes(res, allowed, img, color_map) + _parse_legend_from_text(text, allowed)
            two_row_codes = {
                _norm_code(str(it.get("code") or ""))
                for it in raw_items
                if str(it.get("source") or "") == "two-row" and int(it.get("count") or 0) > 0
            }
            crop_codes = {
                _norm_code(str(it.get("code") or ""))
                for it in raw_items
                if str(it.get("source") or "") == "box-crop-below" and int(it.get("count") or 0) > 0
            }
            text_codes = set()
            for tok in re.findall(r"\b[A-Z]{1,3}0*[0-9]{1,3}\b", _clean_legend_text(text)):
                c = _valid_or_close(tok, allowed)
                if c:
                    text_codes.add(c)
            if len(two_row_codes) >= 6:
                weak_sources = {"box-same-row", "box-below"}
                raw_items = [
                    it for it in raw_items
                    if str(it.get("source") or "") not in weak_sources
                    or _norm_code(str(it.get("code") or "")) in two_row_codes
                ]
            elif len(crop_codes) >= 6:
                weak_sources = {"box-same-row", "box-below"}
                raw_items = [
                    it for it in raw_items
                    if str(it.get("source") or "") not in weak_sources
                    or _norm_code(str(it.get("code") or "")) in crop_codes
                    or _norm_code(str(it.get("code") or "")) in text_codes
                ]
            items = _merge_legend_items(_repair_low_count_long_code_items(_drop_shadow_glued_items(raw_items), allowed))
            return _json_response(self, 200, {"source": "local-ocr-layout", "text": text, "items": items, "elapsedMs": int((time.time() - t0) * 1000)})

        if p == "/legend-vlm":
            data_url = body.get("image") or ""
            img, err = _decode_data_url(data_url)
            if err:
                return _json_response(self, 400, {"error": err})
            valid = body.get("validCodes") or None
            allowed = set([_norm_code(x) for x in valid]) if isinstance(valid, list) else None
            t0 = time.time()
            items, e = _run_vlm_legend(data_url, allowed)
            if e:
                return _json_response(self, 500, {"error": e})
            return _json_response(self, 200, {"source": "vlm", "items": items or [], "elapsedMs": int((time.time() - t0) * 1000)})

        if p == "/legend-glm":
            data_url = body.get("image") or ""
            img, err = _decode_data_url(data_url)
            if err:
                return _json_response(self, 400, {"error": err})
            valid = body.get("validCodes") or None
            allowed = set([_norm_code(x) for x in valid]) if isinstance(valid, list) else None
            model_override = body.get("glmModel") or body.get("model") or None
            base_url, model, api_key, cfg_err = _glm_config(str(model_override) if model_override else None)
            if cfg_err:
                return _json_response(self, 500, {"error": cfg_err})
            t0 = time.time()
            items, e = _run_vlm_legend(data_url, allowed, base_url=base_url, model=model, api_key=api_key)
            if e:
                return _json_response(self, 500, {"error": e})
            return _json_response(self, 200, {"source": "glm", "model": model, "items": items or [], "elapsedMs": int((time.time() - t0) * 1000)})

        if p == "/grid":
            img, err = _decode_data_url(body.get("image") or "")
            if err:
                return _json_response(self, 400, {"error": err})
            try:
                rows = int(body.get("rows") or 0)
                cols = int(body.get("cols") or 0)
            except Exception:
                rows = cols = 0
            if rows <= 0 or cols <= 0:
                return _json_response(self, 400, {"error": "missing rows/cols"})
            allowed_raw = body.get("allowedCodes") or None
            allowed = set([_norm_code(x) for x in allowed_raw]) if isinstance(allowed_raw, list) else None
            mode = str(body.get("mode") or "strips").lower()
            t0 = time.time()
            if mode == "cells":
                img2 = _whiten_grid_lines(img)
                candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else None
                cells, scanned = _grid_ocr_cells(img2, rows, cols, allowed, candidates)
                return _json_response(self, 200, {"cells": cells, "mode": "cells", "scanned": scanned, "filled": len(cells), "elapsedMs": int((time.time() - t0) * 1000)})

            if mode == "paddle":
                candidates = body.get("candidates") if isinstance(body.get("candidates"), list) else None
                cells, detected, e = _grid_ocr_paddle(img, rows, cols, allowed, candidates)
                if e:
                    return _json_response(self, 500, {"error": e})
                return _json_response(self, 200, {"cells": cells or [], "mode": "paddle", "detected": detected, "filled": len(cells or []), "elapsedMs": int((time.time() - t0) * 1000)})

            if mode == "full":
                img2 = _whiten_grid_lines(img)
                res, e = _run_ocr(img2, use_det=True, use_rec=True)
                if e:
                    return _json_response(self, 500, {"error": e})
                w, h = img2.size
                cw = float(w) / float(cols)
                ch = float(h) / float(rows)
                best: Dict[Tuple[int, int], Tuple[str, str, float]] = {}
                detected = 0
                for box, t, s in res or []:
                    detected += 1
                    tt = str(t or "").upper()
                    tt = re.sub(r"[^A-Z0-9]", "", tt)
                    m = re.search(r"[A-Z]{1,3}0*[0-9]{1,3}", tt)
                    if not m:
                        continue
                    raw_code = _norm_code(m.group(0))
                    code = raw_code
                    if allowed:
                        code = code if code in allowed else (_closest_allowed(code, allowed) or "")
                    if not code:
                        continue
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    cx = sum(xs) / 4.0
                    cy = sum(ys) / 4.0
                    r = int(cy / ch)
                    c = int(cx / cw)
                    if r < 0 or c < 0 or r >= rows or c >= cols:
                        continue
                    key = (r, c)
                    prev = best.get(key)
                    if prev is None or s > prev[2]:
                        best[key] = (code, raw_code, float(s))
                cells = [{"r": rc[0], "c": rc[1], "code": v[0], "raw": v[1], "conf": v[2]} for rc, v in best.items()]
                return _json_response(self, 200, {"cells": cells, "mode": "full", "detected": detected, "filled": len(cells), "elapsedMs": int((time.time() - t0) * 1000)})

            cells, detected = _grid_ocr_strips(img, rows, cols, allowed)
            return _json_response(self, 200, {"cells": cells, "mode": "strips", "detected": detected, "filled": len(cells), "elapsedMs": int((time.time() - t0) * 1000)})

        if p == "/grid-vlm":
            data_url = body.get("image") or ""
            img, err = _decode_data_url(data_url)
            if err:
                return _json_response(self, 400, {"error": err})
            try:
                rows = int(body.get("rows") or 0)
                cols = int(body.get("cols") or 0)
            except Exception:
                rows = cols = 0
            if rows <= 0 or cols <= 0:
                return _json_response(self, 400, {"error": "missing rows/cols"})
            allowed_raw = body.get("allowedCodes") or None
            allowed = set([_norm_code(x) for x in allowed_raw]) if isinstance(allowed_raw, list) else None
            mode = str(body.get("mode") or "").lower()
            t0 = time.time()
            if rows == 1 and mode != "region":
                cells, e = _run_vlm_grid_row(data_url, cols, allowed)
            else:
                cells, e = _run_vlm_grid(data_url, rows, cols, allowed)
            if e:
                return _json_response(self, 500, {"error": e})
            return _json_response(self, 200, {"cells": cells or [], "mode": "vlm", "filled": len(cells or []), "elapsedMs": int((time.time() - t0) * 1000)})

        if p == "/cells-vlm":
            data_url = body.get("image") or ""
            img, err = _decode_data_url(data_url)
            if err:
                return _json_response(self, 400, {"error": err})
            try:
                count = int(body.get("count") or 0)
            except Exception:
                count = 0
            if count <= 0:
                return _json_response(self, 400, {"error": "missing count"})
            allowed_raw = body.get("allowedCodes") or None
            allowed = set([_norm_code(x) for x in allowed_raw]) if isinstance(allowed_raw, list) else None
            t0 = time.time()
            cells, e = _run_vlm_cell_batch(data_url, count, allowed)
            if e:
                return _json_response(self, 500, {"error": e})
            return _json_response(self, 200, {"cells": cells or [], "mode": "cells-vlm", "filled": len(cells or []), "elapsedMs": int((time.time() - t0) * 1000)})

        if p == "/cells-glm":
            data_url = body.get("image") or ""
            img, err = _decode_data_url(data_url)
            if err:
                return _json_response(self, 400, {"error": err})
            try:
                count = int(body.get("count") or 0)
            except Exception:
                count = 0
            if count <= 0 or count > 900:
                return _json_response(self, 400, {"error": "invalid count"})
            allowed_raw = body.get("allowedCodes") or None
            allowed = set([_norm_code(x) for x in allowed_raw]) if isinstance(allowed_raw, list) else None
            model_override = body.get("glmModel") or body.get("model") or None
            base_url, model, api_key, cfg_err = _glm_config(str(model_override) if model_override else None)
            if cfg_err:
                return _json_response(self, 500, {"error": cfg_err})
            t0 = time.time()
            cells, e = _run_vlm_cell_batch(data_url, count, allowed, base_url=base_url, model=model, api_key=api_key)
            if e:
                return _json_response(self, 500, {"error": e})
            return _json_response(self, 200, {"cells": cells or [], "mode": "cells-glm", "model": model, "filled": len(cells or []), "elapsedMs": int((time.time() - t0) * 1000)})

        return _json_response(self, 404, {"error": "not found"})

    def log_message(self, fmt, *args):
        return


def main():
    host = os.environ.get("PINDOU_AI_HOST") or "127.0.0.1"
    port = int(os.environ.get("PINDOU_AI_PORT") or "5055")
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    httpd = HTTPServer((host, port), Handler)
    print(f"pindou-local-ai listening on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

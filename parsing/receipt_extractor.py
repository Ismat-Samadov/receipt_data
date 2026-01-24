#!/usr/bin/env python3
"""
receipt_extractor.py
Usage:
  python receipt_extractor.py --input_dir receipts --output_dir outputs --workers 4
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import cv2
import numpy as np
import easyocr
import pandas as pd
from rapidfuzz import fuzz, process
from dateutil import parser as dateparser

# -------------------------
# CONFIG
# -------------------------
OUTPUT_JSON_DIRNAME = "ocr_json"
FINAL_CSV_NAME = "receipts_items.csv"
OCR_LANGS = ["az", "en"]  # Azerbaijani/English. Add 'ru' if many Cyrillic receipts.
BATCH_SIZE = 32           # number of files per internal batch (tune for memory)

# 30 fields requested:
FIELDS = [
    "filename","store_name","store_address","store_code","taxpayer_name","tax_id",
    "receipt_number","cashier_name","date","time","item_name","quantity","unit_price",
    "line_total","subtotal","vat_18_percent","total_tax","cashless_payment","cash_payment",
    "bonus_payment","advance_payment","credit_payment","queue_number","cash_register_model",
    "cash_register_serial","fiscal_id","fiscal_registration","refund_amount","refund_date","refund_time"
]

# -------------------------
# PREPROCESSING
# -------------------------
def preprocess_image_for_ocr(img_path, max_dim=1600):
    """Load image, convert to grayscale, deskew, denoise, resize modestly."""
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read {img_path}")
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # deskew using moments
    coords = np.column_stack(np.where(gray < 250))
    if coords.shape[0] > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # denoise + adaptive threshold might help but don't over-threshold; keep as input for recognizer
    # upscale modestly if small
    h, w = gray.shape
    scale = 1.0
    if max(h, w) < 1200:
        scale = min(max_dim / max(h, w), 2.0)
    if scale > 1.0:
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    # slight bilateral filter to keep edges
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # do mild contrast stretch
    p2, p98 = np.percentile(gray, (2, 98))
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray

# -------------------------
# OCR
# -------------------------
reader = None
def get_easyocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(OCR_LANGS, gpu=False)  # ensure gpu=False for CPU-only
    return reader

def run_ocr_on_image(img):
    r = get_easyocr_reader()
    # EasyOCR returns a list of results: [[bbox],[text],[conf]]
    results = r.readtext(img, detail=1)
    # Convert into simple lines by sorting by Y coordinate and joining words on same line
    lines = []
    # each entry: (bbox, text, conf)
    data = [{"bbox":res[0], "text":res[1], "conf": float(res[2]) if res[2] is not None else 0.0} for res in results]
    # sort by top-left y
    data.sort(key=lambda x: min(pt[1] for pt in x["bbox"]))
    # naive merge into lines by vertical proximity
    current_line = []
    current_y = None
    for entry in data:
        y = np.mean([pt[1] for pt in entry["bbox"]])
        if current_y is None or abs(y - current_y) < 12:  # threshold in pixels
            current_line.append(entry)
            current_y = y if current_y is None else (current_y*0.7 + y*0.3)
        else:
            lines.append(current_line)
            current_line = [entry]
            current_y = y
    if current_line:
        lines.append(current_line)
    # join words per line
    joined_lines = []
    for line in lines:
        text = " ".join([w["text"] for w in line])
        conf = np.mean([w["conf"] for w in line]) if len(line)>0 else 0.0
        joined_lines.append({"text": text.strip(), "conf": conf})
    return joined_lines, data, results

# -------------------------
# PARSING
# -------------------------
# Helpful small helpers
def parse_date_time_from_text(line):
    # try to parse date and/or time in many formats
    try:
        dt = dateparser.parse(line, dayfirst=True, fuzzy=True)
        if dt:
            return dt.date().strftime("%d.%m.%Y"), dt.time().strftime("%H:%M:%S")
    except Exception:
        pass
    return None, None

def find_first_match(lines, keywords, fuzzy_thresh=85):
    """
    Find first line that contains one of keywords (exact or fuzzy).
    Return (line_idx, text)
    """
    for i, ln in enumerate(lines):
        txt = ln["text"]
        txt_low = txt.lower()
        for kw in keywords:
            if kw.lower() in txt_low:
                return i, txt
        # fuzzy check
        best = process.extractOne(txt, keywords, scorer=fuzz.partial_ratio)
        if best and best[1] >= fuzzy_thresh:
            return i, txt
    return None, None

# Regular expressions adapted to receipts style
RE_VOEN = re.compile(r"\b(VOEN|VÖEN|V\.O\.E\.N\.|V[ÖO]EN)\b[:\s]*([0-9]{6,12})", re.I)
RE_RECEIPT_NO = re.compile(r"(Satış çeki №|Satış çeki №|Satış çeki|Çek №|Çek No|Çek №|Satış çeki №|Sale check №)?\s*([A-Za-z0-9\-]+)", re.I)
RE_PRICE = re.compile(r"([0-9]+(?:[.,][0-9]{1,2})?)")
RE_QUEUE = re.compile(r"Növb[əə] ar[çc]ında vurulmuş çek sayı:?\s*([0-9]+)", re.I)
RE_NKA_MODEL = re.compile(r"NKA[- ]?nin modeli[:\s]*([A-Za-z0-9\-\s]+)", re.I)
RE_NKA_SERIAL = re.compile(r"NKA[- ]?nin zavod nömr[əe]si[:\s]*([0-9A-Za-z]+)", re.I)
RE_FISCAL_ID = re.compile(r"Fiskal ID[:\s]*([A-Za-z0-9]+)", re.I)
RE_REFUND = re.compile(r"Geri qaytarılan məbləğ[:\s]*([0-9]+(?:[.,][0-9]{1,2})?)", re.I)
RE_REFUND_DATE = re.compile(r"Geri qaytarılma tarixi[:\s]*([0-9]{2}\.[0-9]{2}\.[0-9]{4})", re.I)

def safe_float(s):
    if s is None:
        return 0.0
    s = str(s).replace(",", ".")
    try:
        return float(re.findall(r"-?\d+\.?\d*", s)[0])
    except Exception:
        return 0.0

def parse_receipt_lines(lines, raw_data):
    """
    Given joined_lines and raw words, produce a dict with all 30 fields.
    Use many heuristics to locate blocks.
    """
    parsed = {k: None for k in FIELDS}
    parsed["filename"] = raw_data["filename"]
    # raw text for debugging
    parsed["_raw_lines"] = [ln["text"] for ln in lines]
    parsed["_avg_conf"] = np.mean([ln["conf"] for ln in lines]) if lines else 0.0
    # 1) store name (often in top 5 lines)
    top_text = " ".join(parsed["_raw_lines"][:6])
    # heuristics
    idx, store = find_first_match(lines[:8], ["Obyektin adı", "OBYEKTIN ADI", "Obyektin adı:", "MAGAZA", "OBA MARKET", "OBYEKTIN ADI"])
    if idx is not None:
        # extract after colon if exists
        txt = lines[idx]["text"]
        # try to strip prefix
        if ":" in txt:
            parsed["store_name"] = txt.split(":",1)[1].strip()
        else:
            parsed["store_name"] = txt.strip()
    else:
        # fallback: first non-empty short line
        for ln in lines[:6]:
            t = ln["text"].strip()
            if t and len(t) < 40 and any(ch.isalpha() for ch in t):
                parsed["store_name"] = t
                break

    # 2) store address / code / taxpayer / VÖEN
    full_text = "\n".join(parsed["_raw_lines"][:40])
    # VÖEN
    m = RE_VOEN.search(full_text)
    if m:
        parsed["tax_id"] = m.group(2)
    else:
        # fallback: find any 9-11 digit sequence prefixed with VOEN-like tokens
        m2 = re.search(r"\b([0-9]{7,12})\b", full_text)
        if m2:
            parsed["tax_id"] = m2.group(1)
    # store address guess: usually second or third lines
    addr_candidates = []
    for ln in parsed["_raw_lines"][:12]:
        if any(word in ln.lower() for word in ["rayonu", "şəhər", "ev", "küçə", "sahə", "səhəri", "bakı", "baki"]):
            addr_candidates.append(ln)
    parsed["store_address"] = addr_candidates[0] if addr_candidates else None

    # store code: sometimes "Obyektin kodu"
    mcode = re.search(r"Obyektin kodu[:\s]*([0-9\-]+)", full_text, re.I)
    if mcode:
        parsed["store_code"] = mcode.group(1)

    # taxpayer name lines
    m_taxpayer = re.search(r"Vergi ödəyicisinin adı[:\s]*([\w\s\.\-ƏəÜüÇçĞğİıÖöŞş]+)", full_text, re.I)
    if m_taxpayer:
        parsed["taxpayer_name"] = m_taxpayer.group(1).strip()

    # receipt number
    mrec = re.search(r"Satış çeki №\s*([0-9A-Za-z\-]+)|Satış çeki №\s*([0-9A-Za-z\-]+)|Satış çeki №\s*([0-9A-Za-z\-]+)|Çek №\s*([0-9A-Za-z\-]+)", full_text, re.I)
    if mrec:
        for g in mrec.groups():
            if g:
                parsed["receipt_number"] = g.strip()
                break

    # cashier, date, time - find lines with "Kassir" or "Kassir:" etc
    for ln in parsed["_raw_lines"]:
        if re.search(r"Kassir|Kassir:", ln, re.I):
            # split on colon
            parts = ln.split(":")
            if len(parts) >= 2:
                parsed["cashier_name"] = parts[1].strip()
    # date/time patterns
    for ln in parsed["_raw_lines"]:
        if re.search(r"\b[0-3]?[0-9]\.[01]?[0-9]\.[12][0-9]{3}\b", ln):
            dmatch = re.search(r"([0-3]?[0-9]\.[01]?[0-9]\.[12][0-9]{3})", ln)
            if dmatch:
                parsed["date"] = dmatch.group(1)
            tmatch = re.search(r"([0-2]?[0-9]:[0-5][0-9]:[0-5][0-9])", ln)
            if tmatch:
                parsed["time"] = tmatch.group(1)

    # queue, NKA model, NKA serial, fiscal id, fiscal registration
    m = RE_QUEUE.search(full_text)
    if m:
        parsed["queue_number"] = safe_float(m.group(1))
    m = RE_NKA_MODEL.search(full_text)
    if m:
        parsed["cash_register_model"] = m.group(1).strip()
    m = RE_NKA_SERIAL.search(full_text)
    if m:
        parsed["cash_register_serial"] = m.group(1).strip()
    m = RE_FISCAL_ID.search(full_text)
    if m:
        parsed["fiscal_id"] = m.group(1).strip()
    # NMQ or fiscal registration fallback
    m_fr = re.search(r"NMQ-?nin qeydiyyat nömrəsi[:\s]*([0-9]+)", full_text, re.I)
    if m_fr:
        parsed["fiscal_registration"] = m_fr.group(1).strip()

    # refund info
    m = RE_REFUND.search(full_text)
    if m:
        parsed["refund_amount"] = safe_float(m.group(1))
    m = RE_REFUND_DATE.search(full_text)
    if m:
        parsed["refund_date"] = m.group(1)
    # sometimes refund time is stored separately, search pattern like 22.01.2026 15:06
    m_dt = re.search(r"([0-3]?[0-9]\.[01]?[0-9]\.[12][0-9]{3})\s+([0-2]?[0-9]:[0-5][0-9])", full_text)
    if m_dt:
        parsed["refund_date"] = m_dt.group(1)
        parsed["refund_time"] = m_dt.group(2)+":00"

    # Payment block: find lines with "Nağd", "Nağdsız", "Bonus", "Avans", etc
    for ln in parsed["_raw_lines"][-40:]:
        if re.search(r"Nağd", ln, re.I):
            mnum = RE_PRICE.search(ln)
            if mnum:
                parsed["cash_payment"] = safe_float(mnum.group(1))
        if re.search(r"Nağdsız|Nağdsız:", ln, re.I):
            mnum = RE_PRICE.search(ln)
            if mnum:
                parsed["cashless_payment"] = safe_float(mnum.group(1))
        if re.search(r"Bonus", ln, re.I):
            mnum = RE_PRICE.search(ln)
            if mnum:
                parsed["bonus_payment"] = safe_float(mnum.group(1))
        if re.search(r"Avans", ln, re.I):
            mnum = RE_PRICE.search(ln)
            if mnum:
                parsed["advance_payment"] = safe_float(mnum.group(1))
        if re.search(r"Nisya|Nisya:", ln, re.I):
            mnum = RE_PRICE.search(ln)
            if mnum:
                parsed["credit_payment"] = safe_float(mnum.group(1))

    # VAT and subtotal: search for "Cəmi" and "// *ƏDV" patterns
    # Capture numeric occurrences close to "Cəmi" or "Cami" spelling variations
    for ln in parsed["_raw_lines"]:
        if re.search(r"^\s*Cəmi|^\s*Cami|^\s*Cəmi:", ln, re.I):
            # extract the number
            mnum = RE_PRICE.search(ln)
            if mnum:
                parsed["subtotal"] = safe_float(mnum.group(1))
        if re.search(r"ƏDV 18%|*ƏDV 18%|%ƏDV 18|EDV 18|ƏDV: 18%", ln, re.I):
            mnum = RE_PRICE.search(ln)
            if mnum:
                parsed["vat_18_percent"] = safe_float(mnum.group(1))

    # ITEM LINES parsing: find block where "Məhsulun adı" header is present, then parse following lines as items
    items = []
    # find header index
    header_idx = None
    for i, ln in enumerate(parsed["_raw_lines"]):
        if re.search(r"Məhsulun adı|Mehsulun adi|Məhsulun adı", ln, re.I) or re.search(r"Say\s+Qiymət\s+Cəmi", ln, re.I):
            header_idx = i
            break
    if header_idx is None:
        # try scanning for lines that look like item rows: pattern name + quantity + price + total
        for ln in parsed["_raw_lines"]:
            # if contains numeric price and space then numeric quantity etc
            if re.search(r"\d+[.,]?\d*\s+[0-9]+[.,]?\d*", ln):
                header_idx = None
                break

    # naive item extraction: scan lines and look for lines with at least two numbers
    for ln in parsed["_raw_lines"]:
        nums = re.findall(r"[0-9]+(?:[.,][0-9]{1,2})?", ln)
        if len(nums) >= 2 and len(ln) < 200:
            # assume price and total are present; first text is item name
            # split by numbers to get name
            pieces = re.split(r"([0-9]+(?:[.,][0-9]{1,2})?)", ln)
            # name is first text portion
            name = pieces[0].strip()
            # take last two numeric tokens as unit price and total, or qty and unit price
            numeric_tokens = [safe_float(n) for n in nums]
            # heuristic: if first numeric looks like a weight or qty (<10 and has decimal) treat as quantity
            quantity = 1.0
            unit_price = None
            line_total = None
            if len(numeric_tokens) >= 2:
                # common case: qty (maybe), unit_price, total
                unit_price = numeric_tokens[-2]
                line_total = numeric_tokens[-1]
                # if the first numeric is plausibly qty and not price:
                if len(numeric_tokens) >= 3:
                    possible_qty = numeric_tokens[-3]
                    if possible_qty < 100 and not float(possible_qty).is_integer() or possible_qty <= 10:
                        quantity = possible_qty
                else:
                    # assume quantity 1 if not explicit
                    quantity = 1.0
            # sanity-check: fix common OCR errors: 1000 -> 1, 2000->2 etc
            if quantity >= 100 and int(quantity) % 1000 == 0:
                quantity = quantity / 1000.0
            if unit_price is None:
                continue
            # recalc if mismatch
            calc_line_total = round(quantity * unit_price, 2)
            if line_total is None or abs(calc_line_total - line_total) > 0.5:
                # trust calculated total
                line_total = calc_line_total
            item = {"item_name": clean_item_name(name), "quantity": round(quantity, 3),
                    "unit_price": round(unit_price, 2), "line_total": round(line_total, 2)}
            items.append(item)

    # If no items detected from heuristics, attempt to find explicit small item lines under header area
    if not items:
        # fallback: try to parse block between header and next dashed line
        # get all lines after header_idx
        start = header_idx+1 if header_idx is not None else 0
        for ln in parsed["_raw_lines"][start:start+80]:
            nums = re.findall(r"[0-9]+(?:[.,][0-9]{1,2})?", ln)
            if len(nums) >= 2:
                # same as above
                pieces = re.split(r"([0-9]+(?:[.,][0-9]{1,2})?)", ln)
                name = pieces[0].strip()
                numeric_tokens = [safe_float(n) for n in nums]
                quantity = 1.0
                unit_price = numeric_tokens[-2]
                line_total = numeric_tokens[-1]
                item = {"item_name": clean_item_name(name), "quantity": round(quantity, 3),
                        "unit_price": round(unit_price, 2), "line_total": round(line_total, 2)}
                items.append(item)

    # if still empty, flag
    parsed["_items"] = items
    parsed["_needs_review"] = True if parsed["_avg_conf"] < 0.85 or len(items) == 0 else False

    return parsed

def clean_item_name(name):
    # remove _ or weird characters and remove common prefixes
    name = re.sub(r'[\*\"“”\']', '', name)
    name = re.sub(r'\s{2,}', ' ', name).strip()
    # remove EDV mentions or percentage tokens
    name = re.sub(r"ƏDV.*", "", name, flags=re.I)
    name = re.sub(r"\bKG\b", "", name)
    return name.strip()

# -------------------------
# MAIN processing per file
# -------------------------
def process_file(path, out_json_dir):
    try:
        gray = preprocess_image_for_ocr(path)
        joined_lines, raw_words, full_results = run_ocr_on_image(gray)
        raw_data = {
            "filename": path.name,
            "ocr_lines": [l["text"] for l in joined_lines],
            "ocr_line_conf": [l["conf"] for l in joined_lines],
        }
        parsed = parse_receipt_lines(joined_lines, {"filename": path.name, "raw_words": raw_words})
        final = {k: parsed.get(k, None) for k in FIELDS}
        # handle item expansion: return one parsed object per item
        items = parsed.get("_items", [])
        outputs = []
        if items:
            for it in items:
                row = final.copy()
                row["item_name"] = it["item_name"]
                row["quantity"] = it["quantity"]
                row["unit_price"] = it["unit_price"]
                row["line_total"] = it["line_total"]
                outputs.append(row)
        else:
            # no items found: create one row with raw fallback
            row = final.copy()
            row["item_name"] = None
            row["quantity"] = None
            row["unit_price"] = None
            row["line_total"] = None
            outputs.append(row)

        # Save raw OCR + parsed JSON
        j = {
            "filename": path.name,
            "parsed": parsed,
            "final_rows": outputs,
            "raw_easyocr_results": [ (r[1], float(r[2])) for r in full_results ]
        }
        jpath = out_json_dir / (path.stem + ".json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
        return outputs, None
    except Exception as e:
        return None, str(e)

# -------------------------
# DRIVER
# -------------------------
def main(input_dir, output_dir, workers=2):
    input_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_json = out_dir / OUTPUT_JSON_DIRNAME
    out_json.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in input_dir.glob("*.jpg")] + [p for p in input_dir.glob("*.jpeg")] + [p for p in input_dir.glob("*.png")])
    all_rows = []
    errors = []
    # process sequentially or with simple batching to limit memory
    for i in tqdm(range(0, len(files), BATCH_SIZE), desc="Batches"):
        batch = files[i:i+BATCH_SIZE]
        for p in tqdm(batch, desc="Files", leave=False):
            rows, err = process_file(p, out_json)
            if err:
                errors.append({"file": p.name, "error": err})
            else:
                for r in rows:
                    all_rows.append(r)
    # final CSV
    df = pd.DataFrame(all_rows, columns=FIELDS)
    csv_path = out_dir / "csv" 
    csv_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path / FINAL_CSV_NAME, index=False, encoding="utf-8-sig")
    # summary
    print("Processed files:", len(files))
    print("Rows written:", len(df))
    print("Errors:", len(errors))
    if errors:
        with open(out_dir / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, workers=args.workers)

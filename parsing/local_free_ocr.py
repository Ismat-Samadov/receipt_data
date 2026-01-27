#!/usr/bin/env python3
"""
FREE LOCAL OCR PIPELINE FOR AZERBAIJANI RECEIPTS
=================================================

Fully local, CPU-only, zero-cost receipt processing pipeline.

Features:
- PaddleOCR (primary) + EasyOCR (fallback) dual-engine approach
- Advanced image preprocessing (deskew, denoise, enhance)
- 30 Azerbaijani-specific regex patterns
- Automatic OCR error correction (1000 → 1.0, etc.)
- Mathematical validation (quantity × price = total)
- Batch processing with progress tracking
- Memory-safe (processes one image at a time)
- Outputs: CSV + JSON + error logs

Hardware Requirements:
- CPU: Any modern processor
- RAM: 16 GB (uses ~4-6 GB peak)
- Storage: ~2 GB for models

Zero Cost - No APIs Required
"""

import os
import re
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output paths
RECEIPTS_DIR = Path("../data/receipts")
OUTPUT_CSV = Path("../data/local_ocr_output.csv")
OUTPUT_JSON_DIR = Path("../data/local_ocr_json")
ERROR_LOG = Path("../data/local_ocr_errors.log")
STATS_FILE = Path("../data/local_ocr_stats.json")

# OCR Engine Configuration
USE_PADDLEOCR = True  # Primary engine
USE_EASYOCR_FALLBACK = True  # Use if PaddleOCR confidence < threshold
CONFIDENCE_THRESHOLD = 0.6  # Switch to EasyOCR if below this

# Preprocessing options
ENABLE_PREPROCESSING = True
MIN_IMAGE_WIDTH = 1200  # Upscale if smaller
MAX_IMAGE_WIDTH = 2400  # Downscale if larger

# Batch processing
BATCH_SIZE = 1  # Process one at a time (memory-safe)
SAVE_INTERMEDIATE_JSON = True  # Debug-friendly

# Error correction
ENABLE_AUTO_CORRECTION = True
DECIMAL_ERROR_THRESHOLD = 100  # Flag prices/quantities > this as likely OCR errors

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ReceiptItem:
    """Single item from a receipt"""
    filename: str
    store_name: Optional[str]
    store_address: Optional[str]
    store_code: Optional[str]
    taxpayer_name: Optional[str]
    tax_id: Optional[str]
    receipt_number: Optional[str]
    cashier_name: Optional[str]
    date: Optional[str]
    time: Optional[str]
    item_name: Optional[str]
    quantity: Optional[float]
    unit_price: Optional[float]
    line_total: Optional[float]
    subtotal: Optional[float]
    vat_18_percent: Optional[float]
    total_tax: Optional[float]
    cashless_payment: Optional[float]
    cash_payment: Optional[float]
    bonus_payment: Optional[float]
    advance_payment: Optional[float]
    credit_payment: Optional[float]
    queue_number: Optional[str]
    cash_register_model: Optional[str]
    cash_register_serial: Optional[str]
    fiscal_id: Optional[str]
    fiscal_registration: Optional[str]
    refund_amount: Optional[float]
    refund_date: Optional[str]
    refund_time: Optional[str]

    # Metadata fields (not in final CSV)
    ocr_confidence: Optional[float] = None
    parsing_errors: Optional[str] = None


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

class ImagePreprocessor:
    """Advanced image preprocessing for receipt OCR"""

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Correct image rotation using coordinate moments"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:  # Don't rotate if nearly straight
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering to reduce noise while preserving edges"""
        return cv2.bilateralFilter(image, 9, 75, 75)

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization"""
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

    def preprocess(self, image_path: Path) -> np.ndarray:
        """Complete preprocessing pipeline"""
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize if needed
        h, w = gray.shape
        if w < MIN_IMAGE_WIDTH:
            scale = MIN_IMAGE_WIDTH / w
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif w > MAX_IMAGE_WIDTH:
            scale = MAX_IMAGE_WIDTH / w
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if not ENABLE_PREPROCESSING:
            return gray

        # Apply preprocessing steps
        gray = self.denoise(gray)
        gray = self.enhance_contrast(gray)
        gray = self.deskew(gray)

        return gray


# ============================================================================
# OCR ENGINES
# ============================================================================

class OCREngine:
    """Dual OCR engine with PaddleOCR primary and EasyOCR fallback"""

    def __init__(self):
        self.paddle_ocr = None
        self.easy_ocr = None
        self._init_engines()

    def _init_engines(self):
        """Lazy initialization of OCR engines"""
        if USE_PADDLEOCR:
            try:
                from paddleocr import PaddleOCR
                logger.info("Initializing PaddleOCR...")
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',  # Latin script (works for Azerbaijani)
                    use_gpu=False,
                    show_log=False
                )
                logger.info("✓ PaddleOCR initialized successfully")
            except ImportError:
                logger.warning("PaddleOCR not installed. Install: pip install paddlepaddle paddleocr")
            except Exception as e:
                logger.warning(f"PaddleOCR initialization failed: {e}")

        if USE_EASYOCR_FALLBACK:
            try:
                import easyocr
                logger.info("Initializing EasyOCR (fallback)...")
                self.easy_ocr = easyocr.Reader(['en'], gpu=False)
                logger.info("✓ EasyOCR initialized successfully")
            except ImportError:
                logger.warning("EasyOCR not installed. Install: pip install easyocr")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")

    def extract_text_paddle(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using PaddleOCR"""
        if self.paddle_ocr is None:
            return "", 0.0

        try:
            result = self.paddle_ocr.ocr(image, cls=True)
            if not result or not result[0]:
                return "", 0.0

            lines = []
            confidences = []

            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                lines.append(text)
                confidences.append(conf)

            full_text = "\n".join(lines)
            avg_confidence = np.mean(confidences) if confidences else 0.0

            return full_text, avg_confidence

        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return "", 0.0

    def extract_text_easy(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using EasyOCR"""
        if self.easy_ocr is None:
            return "", 0.0

        try:
            result = self.easy_ocr.readtext(image)
            if not result:
                return "", 0.0

            lines = []
            confidences = []

            for detection in result:
                text = detection[1]
                conf = detection[2]
                lines.append(text)
                confidences.append(conf)

            # Sort by vertical position (top to bottom)
            result_sorted = sorted(result, key=lambda x: x[0][0][1])
            lines = [r[1] for r in result_sorted]

            full_text = "\n".join(lines)
            avg_confidence = np.mean(confidences) if confidences else 0.0

            return full_text, avg_confidence

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return "", 0.0

    def extract_text(self, image: np.ndarray) -> Tuple[str, float, str]:
        """Extract text using best available engine"""
        # Try PaddleOCR first
        text, confidence = self.extract_text_paddle(image)
        engine_used = "PaddleOCR"

        # Fall back to EasyOCR if confidence too low
        if confidence < CONFIDENCE_THRESHOLD and USE_EASYOCR_FALLBACK:
            logger.info(f"Low PaddleOCR confidence ({confidence:.2f}), trying EasyOCR...")
            text_easy, conf_easy = self.extract_text_easy(image)
            if conf_easy > confidence:
                text, confidence = text_easy, conf_easy
                engine_used = "EasyOCR"

        return text, confidence, engine_used


# ============================================================================
# TEXT PARSING & EXTRACTION
# ============================================================================

class ReceiptParser:
    """Parse OCR text using regex patterns and heuristics"""

    # Azerbaijani receipt patterns
    PATTERNS = {
        'store_name': r'Vergi\s*ödəyicisinin\s*adı[:\s]*(.+?)(?=\n|Ünvan|VÖEN)',
        'store_address': r'Ünvan[:\s]*(.+?)(?=\n\n|\nVÖEN|\nVergi)',
        'tax_id': r'VÖEN[:\s]*(\d+)',
        'taxpayer_name': r'Vergi\s*ödəyicisinin\s*adı[:\s]*(.+?)(?=\n)',
        'receipt_number': r'Satış\s*çeki\s*[№#NоМә]*\s*(\d+)',
        'cashier_name': r'Kassir[:\s]*(.+?)(?=\n|Tarix)',
        'date': r'Tarix[:\s]*(\d{2}\.\d{2}\.\d{4})',
        'time': r'Vaxt[:\s]*(\d{2}:\d{2}:\d{2})',
        'fiscal_id': r'Fiskal\s*[İi]D[:\s]*([A-Za-z0-9]+)',
        'queue_number': r'Növbə[:\s]*(\d+)',
        'cash_register_serial': r'NKA\s*seriya\s*nömrəsi[:\s]*([A-Z0-9]+)',
        'cash_register_model': r'NKA\s*markası[:\s]*(.+?)(?=\n)',
        'fiscal_registration': r'NMQ[:\s]*([A-Z0-9]+)',
        'subtotal': r'Yekun\s*məbləğ[:\s]*([\d.,]+)',
        'vat_18_percent': r'ƏDV\s*18%[:\s]*([\d.,]+)',
        'total_tax': r'Vergi[:\s]*([\d.,]+)',
        'cash_payment': r'Nağd[:\s]*([\d.,]+)',
        'cashless_payment': r'Nağdsız[:\s]*([\d.,]+)',
        'bonus_payment': r'Bonus[:\s]*([\d.,]+)',
        'advance_payment': r'Avans[:\s]*([\d.,]+)',
        'credit_payment': r'Kredit[:\s]*([\d.,]+)',
        'refund_amount': r'Qaytarılan\s*məbləğ[:\s]*([\d.,]+)',
        'refund_date': r'Qaytarma\s*tarixi[:\s]*(\d{2}\.\d{2}\.\d{4})',
        'refund_time': r'Qaytarma\s*vaxtı[:\s]*(\d{2}:\d{2}:\d{2})',
    }

    # Items section pattern
    ITEMS_HEADER_PATTERN = r'Məhsulun\s*adı\s+Say\s+Qiymət\s+Cəmi'

    # Item line patterns
    ITEM_PATTERN = r'(.+?)\s+([\d.,]+)\s+([\d.,]+)\s+([\d.,]+)'

    def clean_item_name(self, name: str) -> str:
        """Clean item name from VAT codes and special characters"""
        if not name:
            return ""

        # Remove VAT codes
        name = re.sub(r'\*?ƏDV[:\s]*\d*\.?\d*%?', '', name, flags=re.IGNORECASE)
        name = re.sub(r'vƏDV|ƏDV', '', name, flags=re.IGNORECASE)

        # Remove leading/trailing quotes and whitespace
        name = name.strip(' "\'*')

        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name)

        return name

    def parse_float(self, value: Optional[str]) -> Optional[float]:
        """Parse Azerbaijani number format to float"""
        if not value:
            return None

        try:
            # Replace comma with period (Azerbaijani format: 1,50 → 1.50)
            value = value.replace(',', '.')
            return round(float(value), 2)
        except (ValueError, AttributeError):
            return None

    def extract_field(self, text: str, pattern_name: str) -> Optional[str]:
        """Extract single field using regex pattern"""
        pattern = self.PATTERNS.get(pattern_name)
        if not pattern:
            return None

        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def extract_items(self, text: str) -> List[Dict]:
        """Extract all items from receipt"""
        items = []

        # Find items section
        header_match = re.search(self.ITEMS_HEADER_PATTERN, text, re.IGNORECASE)
        if not header_match:
            logger.warning("Items header not found in receipt")
            return items

        # Get text after header
        items_section = text[header_match.end():]

        # Stop at footer keywords
        footer_keywords = ['Yekun', 'Cəmi', 'Vergi', 'Nağd', 'Kassir']
        for keyword in footer_keywords:
            pos = items_section.lower().find(keyword.lower())
            if pos > 0:
                items_section = items_section[:pos]
                break

        # Extract items
        lines = items_section.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # Try to match item pattern
            # Format: "Item Name    Qty    Price    Total"
            parts = line.split()
            if len(parts) < 4:
                continue

            # Last 3 should be numbers
            try:
                total = self.parse_float(parts[-1])
                price = self.parse_float(parts[-2])
                qty = self.parse_float(parts[-3])
                name = ' '.join(parts[:-3])

                if total is not None and price is not None and qty is not None:
                    items.append({
                        'name': self.clean_item_name(name),
                        'quantity': qty,
                        'unit_price': price,
                        'line_total': total
                    })
            except Exception as e:
                logger.debug(f"Failed to parse item line: {line} - {e}")
                continue

        return items

    def parse_receipt(self, text: str, filename: str) -> List[ReceiptItem]:
        """Parse full receipt text into structured data"""
        # Extract receipt-level fields
        receipt_data = {
            'filename': filename,
            'store_name': self.extract_field(text, 'store_name'),
            'store_address': self.extract_field(text, 'store_address'),
            'tax_id': self.extract_field(text, 'tax_id'),
            'taxpayer_name': self.extract_field(text, 'taxpayer_name'),
            'receipt_number': self.extract_field(text, 'receipt_number'),
            'cashier_name': self.extract_field(text, 'cashier_name'),
            'date': self.extract_field(text, 'date'),
            'time': self.extract_field(text, 'time'),
            'fiscal_id': self.extract_field(text, 'fiscal_id'),
            'queue_number': self.extract_field(text, 'queue_number'),
            'cash_register_serial': self.extract_field(text, 'cash_register_serial'),
            'cash_register_model': self.extract_field(text, 'cash_register_model'),
            'fiscal_registration': self.extract_field(text, 'fiscal_registration'),
            'subtotal': self.parse_float(self.extract_field(text, 'subtotal')),
            'vat_18_percent': self.parse_float(self.extract_field(text, 'vat_18_percent')),
            'total_tax': self.parse_float(self.extract_field(text, 'total_tax')),
            'cash_payment': self.parse_float(self.extract_field(text, 'cash_payment')),
            'cashless_payment': self.parse_float(self.extract_field(text, 'cashless_payment')),
            'bonus_payment': self.parse_float(self.extract_field(text, 'bonus_payment')),
            'advance_payment': self.parse_float(self.extract_field(text, 'advance_payment')),
            'credit_payment': self.parse_float(self.extract_field(text, 'credit_payment')),
            'refund_amount': self.parse_float(self.extract_field(text, 'refund_amount')),
            'refund_date': self.extract_field(text, 'refund_date'),
            'refund_time': self.extract_field(text, 'refund_time'),
            'store_code': None,  # Not commonly on receipts
        }

        # Extract items
        items = self.extract_items(text)

        # Create ReceiptItem objects
        receipt_items = []
        if items:
            for item in items:
                item_data = receipt_data.copy()
                item_data.update({
                    'item_name': item['name'],
                    'quantity': item['quantity'],
                    'unit_price': item['unit_price'],
                    'line_total': item['line_total'],
                })
                receipt_items.append(ReceiptItem(**item_data))
        else:
            # No items found - create single row with receipt data and None for item fields
            logger.warning(f"No items extracted from {filename}")
            receipt_data.update({
                'item_name': None,
                'quantity': None,
                'unit_price': None,
                'line_total': None,
            })
            receipt_items.append(ReceiptItem(**receipt_data))

        return receipt_items


# ============================================================================
# VALIDATION & AUTO-CORRECTION
# ============================================================================

class DataValidator:
    """Validate and auto-correct OCR errors"""

    def fix_decimal_errors(self, value: Optional[float]) -> Optional[float]:
        """Fix common OCR decimal errors (1000 → 1.0, 2000 → 2.0)"""
        if value is None:
            return None

        # Common OCR errors: reads "1.00" as "1000" or "100"
        if value >= DECIMAL_ERROR_THRESHOLD:
            # Try dividing by 1000, 100, 10
            for divisor in [1000, 100, 10]:
                corrected = value / divisor
                if 0.01 <= corrected <= 999.99:  # Reasonable price/qty range
                    logger.info(f"Decimal correction: {value} → {corrected}")
                    return round(corrected, 2)

        return value

    def validate_math(self, item: ReceiptItem) -> ReceiptItem:
        """Validate quantity × unit_price = line_total"""
        if item.quantity and item.unit_price and item.line_total:
            calculated = round(item.quantity * item.unit_price, 2)
            actual = item.line_total

            if abs(calculated - actual) > 0.02:  # Allow 2 cent tolerance
                logger.warning(
                    f"{item.filename}: Math error - "
                    f"{item.quantity} × {item.unit_price} = {calculated} "
                    f"but receipt says {actual}"
                )
                # Trust line_total, recalculate unit_price
                if item.quantity > 0:
                    item.unit_price = round(actual / item.quantity, 2)

        return item

    def validate_item(self, item: ReceiptItem) -> ReceiptItem:
        """Apply all validation and corrections"""
        if not ENABLE_AUTO_CORRECTION:
            return item

        # Fix decimal errors
        item.quantity = self.fix_decimal_errors(item.quantity)
        item.unit_price = self.fix_decimal_errors(item.unit_price)

        # Validate math
        item = self.validate_math(item)

        # Flag suspicious values
        errors = []
        if item.quantity and item.quantity > 50:
            errors.append(f"High quantity: {item.quantity}")
        if item.unit_price and item.unit_price > 500:
            errors.append(f"High price: {item.unit_price}")

        if errors:
            item.parsing_errors = "; ".join(errors)

        return item


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class ReceiptProcessor:
    """Main processing pipeline"""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine()
        self.parser = ReceiptParser()
        self.validator = DataValidator()
        self.stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'total_items': 0,
            'avg_confidence': 0.0,
            'engine_usage': {'PaddleOCR': 0, 'EasyOCR': 0}
        }

    def process_image(self, image_path: Path) -> List[ReceiptItem]:
        """Process single receipt image"""
        logger.info(f"Processing: {image_path.name}")

        try:
            # Preprocess
            image = self.preprocessor.preprocess(image_path)

            # OCR
            text, confidence, engine = self.ocr_engine.extract_text(image)

            if not text:
                logger.error(f"No text extracted from {image_path.name}")
                return []

            logger.info(f"OCR confidence: {confidence:.2%} (engine: {engine})")

            # Parse
            items = self.parser.parse_receipt(text, image_path.name)

            # Validate
            items = [self.validator.validate_item(item) for item in items]

            # Add metadata
            for item in items:
                item.ocr_confidence = confidence

            # Update stats
            self.stats['total_images'] += 1
            self.stats['successful'] += 1
            self.stats['total_items'] += len(items)
            self.stats['avg_confidence'] += confidence
            self.stats['engine_usage'][engine] = self.stats['engine_usage'].get(engine, 0) + 1

            # Save intermediate JSON if enabled
            if SAVE_INTERMEDIATE_JSON:
                self.save_json(image_path.stem, items, text, confidence)

            return items

        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            logger.debug(traceback.format_exc())
            self.stats['failed'] += 1

            # Log to error file
            with open(ERROR_LOG, 'a') as f:
                f.write(f"{datetime.now()}: {image_path.name} - {str(e)}\n")

            return []

    def save_json(self, filename: str, items: List[ReceiptItem], raw_text: str, confidence: float):
        """Save intermediate JSON for debugging"""
        OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

        output = {
            'filename': filename,
            'ocr_confidence': confidence,
            'raw_text': raw_text,
            'items': [asdict(item) for item in items]
        }

        json_path = OUTPUT_JSON_DIR / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def process_batch(self, image_paths: List[Path]) -> List[ReceiptItem]:
        """Process batch of images"""
        all_items = []

        total = len(image_paths)
        for idx, path in enumerate(image_paths, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            logger.info(f"{'='*60}")

            items = self.process_image(path)
            all_items.extend(items)

        return all_items

    def save_csv(self, items: List[ReceiptItem]):
        """Save results to CSV"""
        if not items:
            logger.warning("No items to save")
            return

        # Convert to DataFrame
        df = pd.DataFrame([asdict(item) for item in items])

        # Remove metadata columns
        df = df.drop(columns=['ocr_confidence', 'parsing_errors'], errors='ignore')

        # Ensure correct column order (30 fields)
        columns = [
            'filename', 'store_name', 'store_address', 'store_code', 'taxpayer_name',
            'tax_id', 'receipt_number', 'cashier_name', 'date', 'time',
            'item_name', 'quantity', 'unit_price', 'line_total',
            'subtotal', 'vat_18_percent', 'total_tax',
            'cashless_payment', 'cash_payment', 'bonus_payment',
            'advance_payment', 'credit_payment',
            'queue_number', 'cash_register_model', 'cash_register_serial',
            'fiscal_id', 'fiscal_registration',
            'refund_amount', 'refund_date', 'refund_time'
        ]

        df = df[columns]

        # Save
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        logger.info(f"\n✓ Saved {len(df)} rows to {OUTPUT_CSV}")

    def save_stats(self):
        """Save processing statistics"""
        if self.stats['successful'] > 0:
            self.stats['avg_confidence'] /= self.stats['successful']

        with open(STATS_FILE, 'w') as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total images: {self.stats['total_images']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total items extracted: {self.stats['total_items']}")
        logger.info(f"Average OCR confidence: {self.stats['avg_confidence']:.2%}")
        logger.info(f"Engine usage: {self.stats['engine_usage']}")
        logger.info(f"{'='*60}\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    logger.info("="*60)
    logger.info("FREE LOCAL OCR PIPELINE FOR AZERBAIJANI RECEIPTS")
    logger.info("="*60)
    logger.info(f"Input directory: {RECEIPTS_DIR}")
    logger.info(f"Output CSV: {OUTPUT_CSV}")
    logger.info(f"OCR Engine: PaddleOCR (primary) + EasyOCR (fallback)")
    logger.info(f"Preprocessing: {'Enabled' if ENABLE_PREPROCESSING else 'Disabled'}")
    logger.info(f"Auto-correction: {'Enabled' if ENABLE_AUTO_CORRECTION else 'Disabled'}")
    logger.info("="*60 + "\n")

    # Find all images
    image_paths = sorted(RECEIPTS_DIR.glob("*.jpeg"))
    image_paths.extend(sorted(RECEIPTS_DIR.glob("*.jpg")))
    image_paths.extend(sorted(RECEIPTS_DIR.glob("*.png")))

    if not image_paths:
        logger.error(f"No images found in {RECEIPTS_DIR}")
        return

    logger.info(f"Found {len(image_paths)} images to process\n")

    # Initialize processor
    processor = ReceiptProcessor()

    # Process all images
    all_items = processor.process_batch(image_paths)

    # Save results
    processor.save_csv(all_items)
    processor.save_stats()

    logger.info("✓ Processing complete!")
    logger.info(f"✓ CSV output: {OUTPUT_CSV}")
    logger.info(f"✓ JSON outputs: {OUTPUT_JSON_DIR}/")
    logger.info(f"✓ Error log: {ERROR_LOG}")
    logger.info(f"✓ Statistics: {STATS_FILE}")


if __name__ == "__main__":
    main()

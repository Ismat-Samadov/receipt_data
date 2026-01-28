#!/usr/bin/env python3
"""
VISION LANGUAGE MODEL OCR FOR AZERBAIJANI RECEIPTS
==================================================

Uses local VLMs (Moondream, Qwen2-VL, or LLaVA) via Ollama for receipt extraction.

Requirements:
- Ollama installed: brew install ollama
- Model downloaded: ollama pull moondream (or qwen2-vl:2b)

Memory Usage:
- Moondream (1.6B): ~2-3 GB RAM
- Qwen2-VL-2B: ~3-4 GB RAM
- LLaVA-7B-Q4: ~4-5 GB RAM

Performance: ~5-10 seconds per receipt (depending on model)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import sys

try:
    import ollama
except ImportError:
    print("Installing ollama-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
    import ollama

import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose your model (install with: ollama pull <model>)
MODEL = "llava:7b-v1.6"  # Options: moondream, qwen2-vl:2b, llava:7b-v1.6

# Input/Output
RECEIPTS_DIR = Path("../data/receipts")
OUTPUT_CSV = Path("../data/vllm_output.csv")
OUTPUT_JSON_DIR = Path("../data/vllm_json")
ERROR_LOG = Path("../data/vllm_errors.log")

# Processing
BATCH_SIZE = 1  # Process one at a time

# Prompt template
EXTRACTION_PROMPT = """You are an advanced OCR system specialized in Azerbaijani receipts. Extract ALL information from this receipt image with maximum accuracy.

CRITICAL AZERBAIJANI KEYWORDS:
- "Vergi ödəyicisinin adı" = Taxpayer name / Store name
- "VÖEN" = Tax ID (numeric)
- "Satış çeki" = Sales receipt
- "Tarix" = Date (format: DD.MM.YYYY)
- "Vaxt" = Time (format: HH:MM:SS)
- "Kassir" = Cashier name
- "Məhsulun adı" = Product name
- "Say" = Quantity
- "Qiymət" = Unit price
- "Cəmi" / "Yekun" = Total/Subtotal
- "ƏDV" = VAT (18%)
- "Nağd" = Cash payment
- "Nağdsız" = Cashless payment
- "Bonus" = Bonus payment
- "Avans" = Advance payment
- "Kredit" = Credit payment
- "Fiskal ID" = Fiscal ID
- "NKA seriya nömrəsi" = Cash register serial
- "NKA markası" = Cash register model
- "NMQ" = Fiscal registration number
- "Növbə" = Queue number
- "Qaytarılan məbləğ" = Refund amount
- "Qaytarma tarixi" = Refund date
- "Qaytarma vaxtı" = Refund time

IMPORTANT INSTRUCTIONS:
1. Extract ALL items from the receipt (look for section starting with "Məhsulun adı Say Qiymət Cəmi")
2. Clean item names: Remove ƏDV codes, VAT markers, quotes
3. Fix OCR errors: quantities like 1000 → 1.0, 2000 → 2.0 (decimal misreads)
4. Validate: quantity × unit_price should equal total for each item
5. Use realistic Azerbaijan market prices (water: 0.5-2 AZN, bread: 0.5-4 AZN, etc.)

Extract REAL data from the receipt image and return it as a JSON object with these fields:
- store_name: actual store name from receipt
- store_address: actual address from receipt
- store_code: store code if present
- taxpayer_name: taxpayer name (usually same as store)
- tax_id: VÖEN number (numeric)
- receipt_number: receipt/check number
- cashier_name: cashier's actual name
- date: date in DD.MM.YYYY format
- time: time in HH:MM:SS format
- items: array of products with name, quantity (float), unit_price (float), total (float)
- subtotal: subtotal amount
- vat_18_percent: VAT amount
- total_tax: total tax amount
- cashless_payment: cashless payment amount
- cash_payment: cash payment amount
- bonus_payment: bonus payment amount
- advance_payment: advance payment amount
- credit_payment: credit payment amount
- queue_number: queue/line number
- cash_register_model: NKA model/brand
- cash_register_serial: NKA serial number
- fiscal_id: fiscal ID code
- fiscal_registration: NMQ registration number
- refund_amount: refund amount if present
- refund_date: refund date if present
- refund_time: refund time if present

Return ONLY a valid JSON object with the actual extracted data, no markdown, no explanations."""

# ============================================================================
# VLM PROCESSOR
# ============================================================================

class VLMReceiptProcessor:
    """Process receipts using Vision Language Models"""

    def __init__(self, model_name: str = MODEL):
        self.model = model_name
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_items': 0
        }
        self._check_ollama()
        self._check_model()

    def _check_ollama(self):
        """Check if Ollama is installed and running"""
        try:
            subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                check=True
            )
            logger.info("✓ Ollama is installed and running")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Ollama not found. Install it:")
            logger.error("   brew install ollama")
            logger.error("   ollama serve  # Run in another terminal")
            sys.exit(1)

    def _check_model(self):
        """Check if model is downloaded"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True
            )

            if self.model not in result.stdout:
                logger.warning(f"Model {self.model} not found. Downloading...")
                logger.info(f"Running: ollama pull {self.model}")
                subprocess.run(['ollama', 'pull', self.model], check=True)
                logger.info(f"✓ Model {self.model} downloaded")
            else:
                logger.info(f"✓ Model {self.model} ready")
        except Exception as e:
            logger.error(f"Error checking model: {e}")
            logger.error(f"Please run: ollama pull {self.model}")
            sys.exit(1)

    def extract_json_from_response(self, text: str) -> Optional[Dict]:
        """Extract JSON from VLM response"""
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from response: {text[:200]}...")
            return None

    def process_image(self, image_path: Path) -> Optional[Dict]:
        """Process single receipt image with VLM"""
        logger.info(f"Processing: {image_path.name}")

        try:
            # Call VLM
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': EXTRACTION_PROMPT,
                    'images': [str(image_path)]
                }]
            )

            response_text = response['message']['content']

            # Parse JSON response
            data = self.extract_json_from_response(response_text)

            if data:
                data['filename'] = image_path.name
                data['raw_response'] = response_text

                # Count items
                items_count = len(data.get('items', []))
                self.stats['total_items'] += items_count
                self.stats['success'] += 1

                logger.info(f"✓ Extracted {items_count} items")

                # Save intermediate JSON
                self._save_json(image_path.stem, data)

                return data
            else:
                logger.error(f"Failed to parse response for {image_path.name}")
                self.stats['failed'] += 1
                return None

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            self.stats['failed'] += 1

            # Log error
            with open(ERROR_LOG, 'a') as f:
                f.write(f"{datetime.now()}: {image_path.name} - {str(e)}\n")

            return None

    def _save_json(self, filename: str, data: Dict):
        """Save intermediate JSON"""
        OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
        json_path = OUTPUT_JSON_DIR / f"{filename}.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def process_batch(self, image_paths: List[Path]) -> List[Dict]:
        """Process batch of images"""
        results = []
        total = len(image_paths)

        for idx, path in enumerate(image_paths, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            logger.info(f"{'='*60}")

            self.stats['total'] += 1
            result = self.process_image(path)

            if result:
                results.append(result)

        return results

    def convert_to_csv(self, results: List[Dict]) -> pd.DataFrame:
        """Convert VLM results to CSV format (one row per item)"""
        rows = []

        for receipt in results:
            items = receipt.get('items', [])

            # Receipt-level data - ALL 30 fields
            base_data = {
                'filename': receipt.get('filename'),
                'store_name': receipt.get('store_name'),
                'store_address': receipt.get('store_address'),
                'store_code': receipt.get('store_code'),
                'taxpayer_name': receipt.get('taxpayer_name') or receipt.get('store_name'),
                'tax_id': receipt.get('tax_id'),
                'receipt_number': receipt.get('receipt_number'),
                'cashier_name': receipt.get('cashier_name'),
                'date': receipt.get('date'),
                'time': receipt.get('time'),
                'subtotal': receipt.get('subtotal'),
                'vat_18_percent': receipt.get('vat_18_percent'),
                'total_tax': receipt.get('total_tax'),
                'cashless_payment': receipt.get('cashless_payment'),
                'cash_payment': receipt.get('cash_payment'),
                'bonus_payment': receipt.get('bonus_payment'),
                'advance_payment': receipt.get('advance_payment'),
                'credit_payment': receipt.get('credit_payment'),
                'queue_number': receipt.get('queue_number'),
                'cash_register_model': receipt.get('cash_register_model'),
                'cash_register_serial': receipt.get('cash_register_serial'),
                'fiscal_id': receipt.get('fiscal_id'),
                'fiscal_registration': receipt.get('fiscal_registration'),
                'refund_amount': receipt.get('refund_amount'),
                'refund_date': receipt.get('refund_date'),
                'refund_time': receipt.get('refund_time'),
            }

            if items:
                # One row per item
                for item in items:
                    row = base_data.copy()
                    row.update({
                        'item_name': item.get('name'),
                        'quantity': item.get('quantity'),
                        'unit_price': item.get('unit_price'),
                        'line_total': item.get('total'),
                    })
                    rows.append(row)
            else:
                # No items - create single row
                row = base_data.copy()
                row.update({
                    'item_name': None,
                    'quantity': None,
                    'unit_price': None,
                    'line_total': None,
                })
                rows.append(row)

        return pd.DataFrame(rows)

    def save_csv(self, results: List[Dict]):
        """Save results to CSV"""
        if not results:
            logger.warning("No results to save")
            return

        df = self.convert_to_csv(results)

        # Ensure column order
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

        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

        logger.info(f"\n✓ Saved {len(df)} rows to {OUTPUT_CSV}")

    def print_stats(self):
        """Print processing statistics"""
        logger.info(f"\n{'='*60}")
        logger.info("PROCESSING STATISTICS")
        logger.info(f"{'='*60}")
        logger.info(f"Total images: {self.stats['total']}")
        logger.info(f"Successful: {self.stats['success']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Total items extracted: {self.stats['total_items']}")
        logger.info(f"Success rate: {self.stats['success']/self.stats['total']*100:.1f}%")
        logger.info(f"{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("="*60)
    logger.info("VISION LANGUAGE MODEL OCR PIPELINE")
    logger.info("="*60)
    logger.info(f"Model: {MODEL}")
    logger.info(f"Input: {RECEIPTS_DIR}")
    logger.info(f"Output: {OUTPUT_CSV}")
    logger.info("="*60 + "\n")

    # Find images
    image_paths = sorted(RECEIPTS_DIR.glob("*.jpeg"))
    image_paths.extend(sorted(RECEIPTS_DIR.glob("*.jpg")))
    image_paths.extend(sorted(RECEIPTS_DIR.glob("*.png")))

    if not image_paths:
        logger.error(f"No images found in {RECEIPTS_DIR}")
        return

    logger.info(f"Found {len(image_paths)} images\n")

    # Process
    processor = VLMReceiptProcessor(MODEL)
    results = processor.process_batch(image_paths)

    # Save
    processor.save_csv(results)
    processor.print_stats()

    logger.info("✓ Processing complete!")
    logger.info(f"✓ CSV: {OUTPUT_CSV}")
    logger.info(f"✓ JSON: {OUTPUT_JSON_DIR}/")


if __name__ == "__main__":
    main()

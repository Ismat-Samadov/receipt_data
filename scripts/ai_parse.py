#!/usr/bin/env python3
"""
Improved AI-powered receipt parser using OpenAI GPT-4o Vision API.

Key improvements:
- Direct image analysis with GPT-4o Vision (no pytesseract needed)
- Structured outputs with JSON Schema for guaranteed format
- Better error handling and rate limiting
- Cost tracking and optimization
- Base64 image encoding for reliability
- Interruption handling with checkpoint/resume capability
"""

import os
import re
import json
import base64
import pandas as pd
from pathlib import Path
import logging
from openai import OpenAI
from dotenv import load_dotenv
import time
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
RECEIPTS_DIR = Path('../data/receipts')
OUTPUT_CSV = Path('../data/items.csv')
CHECKPOINT_FILE = Path('../data/parse_checkpoint.json')
COMPLETED_FILE = Path('../data/parse_completed.txt')
BATCH_SIZE = 5  # Smaller batches for vision API
MAX_WORKERS = 3  # Reduced for API rate limiting
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2  # seconds

# Cost tracking (GPT-4o pricing as of 2025)
COST_PER_1K_INPUT_TOKENS = 0.0025  # $0.0025 per 1K input tokens
COST_PER_1K_OUTPUT_TOKENS = 0.010  # $0.010 per 1K output tokens
COST_PER_IMAGE = 0.00765  # Additional cost for vision

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY must be set in .env file")

# Initialize client with explicit base_url to avoid environment variable override
client = OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1"  # Explicit URL to prevent OPENAI_BASE_URL env override
)

# Thread-safe counters and state
counter_lock = threading.Lock()
processed_count = 0
total_cost = 0.0
total_input_tokens = 0
total_output_tokens = 0
stop_requested = False

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global stop_requested
    logger.warning(f"\n⚠️  Interrupt signal received ({signum}). Finishing current batch and saving progress...")
    logger.warning("Press Ctrl+C again to force quit (may lose current batch data)")
    stop_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# JSON Schema for structured output
RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "description": "Array of all items from the receipt",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "store_name": {"type": ["string", "null"]},
                    "store_address": {"type": ["string", "null"]},
                    "store_code": {"type": ["string", "null"]},
                    "taxpayer_name": {"type": ["string", "null"]},
                    "tax_id": {"type": ["string", "null"]},
                    "receipt_number": {"type": ["string", "null"]},
                    "cashier_name": {"type": ["string", "null"]},
                    "date": {"type": ["string", "null"], "description": "DD.MM.YYYY format"},
                    "time": {"type": ["string", "null"], "description": "HH:MM:SS format"},
                    "item_name": {"type": ["string", "null"]},
                    "quantity": {"type": ["string", "null"]},
                    "unit_price": {"type": ["string", "null"]},
                    "line_total": {"type": ["string", "null"]},
                    "subtotal": {"type": ["string", "null"]},
                    "vat_18_percent": {"type": ["string", "null"]},
                    "total_tax": {"type": ["string", "null"]},
                    "cashless_payment": {"type": ["string", "null"]},
                    "cash_payment": {"type": ["string", "null"]},
                    "bonus_payment": {"type": ["string", "null"]},
                    "advance_payment": {"type": ["string", "null"]},
                    "credit_payment": {"type": ["string", "null"]},
                    "queue_number": {"type": ["string", "null"]},
                    "cash_register_model": {"type": ["string", "null"]},
                    "cash_register_serial": {"type": ["string", "null"]},
                    "fiscal_id": {"type": ["string", "null"]},
                    "fiscal_registration": {"type": ["string", "null"]},
                    "refund_amount": {"type": ["string", "null"]},
                    "refund_date": {"type": ["string", "null"]},
                    "refund_time": {"type": ["string", "null"]}
                },
                "required": ["filename", "item_name", "quantity", "unit_price", "line_total"],
                "additionalProperties": False
            }
        }
    },
    "required": ["items"],
    "additionalProperties": False
}


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def calculate_cost(input_tokens: int, output_tokens: int, has_image: bool = True) -> float:
    """Calculate API call cost."""
    cost = (input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS +
            output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)
    if has_image:
        cost += COST_PER_IMAGE
    return cost


def extract_items_with_vision_ai(image_path: Path, filename: str, max_retries: int = MAX_RETRIES) -> List[Dict]:
    """
    Extract receipt data using GPT-4o Vision API with structured outputs.

    This uses OpenAI's vision capabilities to directly analyze receipt images
    without needing traditional OCR preprocessing.
    """

    system_prompt = """You are an expert in Azerbaijani fiscal receipt processing with deep knowledge of:
- Azerbaijan retail market pricing and products
- Azerbaijani language and retail terminology
- Receipt format standards in Azerbaijan
- Common OCR errors in receipt images

Your task is to extract ALL items from Azerbaijani receipts with maximum accuracy."""

    user_prompt = f"""Analyze this Azerbaijani receipt image and extract ALL items with complete details.

CRITICAL REQUIREMENTS:

1. **Extract EVERY Item**: Most receipts have 2-15 items. Extract all of them.

2. **Item Section Identification**: Look for "Məhsulun adı Say Qiymət Cəmi" header.

3. **OCR Error Correction**:
   - Fix decimal misreads: "1000" → "1.0", "2000" → "2.0" (OCR often misreads "1.000" as "1000")
   - Typical quantities are 1-10 units (flag >50 as suspicious)
   - Character confusion: "0" vs "O", "1" vs "l", "5" vs "S"

4. **Azerbaijan Market Pricing** (2025):
   - Water/beverages: 0.40-2.00 AZN
   - Bread/bakery: 0.30-4.00 AZN
   - Dairy: 1.00-8.00 AZN
   - Snacks: 0.50-5.00 AZN
   - Household: 0.20-20.00 AZN
   - Adjust unrealistic prices based on product type

5. **Mathematical Validation**:
   - Verify: quantity × unit_price = line_total
   - If mismatch, correct the most likely error
   - Receipt totals are usually accurate

6. **Text Cleaning**:
   - Remove VAT codes: "*ƏDV", "vƏDV", "ƏDV:", "ƏDV-dən azad"
   - Remove quotes, extra spaces, OCR artifacts

7. **Receipt Metadata**: Same for all items from one receipt:
   - Store name, address, date, time
   - Cashier, receipt number, fiscal ID
   - Totals, VAT, payment methods

Return a JSON object with an "items" array. Each item object must have all 30 fields.

Filename: {filename}"""

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"  Retry {attempt + 1}/{max_retries} for {filename}")

            # Encode image to base64
            base64_image = encode_image_to_base64(image_path)

            # Make API call with vision and structured output
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"  # High detail for better OCR
                                }
                            }
                        ]
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "receipt_extraction",
                        "strict": True,
                        "schema": RECEIPT_SCHEMA
                    }
                },
                temperature=0.1,  # Low temperature for consistency
                max_tokens=4000   # Sufficient for multiple items
            )

            # Extract response
            ai_response = response.choices[0].message.content

            # Track costs
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            cost = calculate_cost(input_tokens, output_tokens, has_image=True)

            with counter_lock:
                global total_cost, total_input_tokens, total_output_tokens
                total_cost += cost
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

            # Parse JSON response
            result = json.loads(ai_response)
            items = result.get('items', [])

            # Validate and clean items
            validated_items = []
            for item in items:
                validated_item = validate_and_clean_item(item)
                if validated_item:
                    validated_items.append(validated_item)

            if not validated_items:
                logger.warning(f"  No valid items extracted from {filename}")
                if attempt < max_retries - 1:
                    time.sleep(INITIAL_RETRY_DELAY * (2 ** attempt))
                    continue
                return []

            logger.info(f"  ✓ Extracted {len(validated_items)} items | Cost: ${cost:.4f}")
            return validated_items

        except Exception as e:
            logger.error(f"  Error processing {filename} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                logger.info(f"  Waiting {delay}s before retry...")
                time.sleep(delay)
            else:
                logger.error(f"  Failed after {max_retries} attempts")
                return []

    return []


def validate_and_clean_item(item: Dict) -> Optional[Dict]:
    """Validate and clean individual item data."""

    try:
        # Check required fields
        if not item.get('item_name') or item['item_name'] in [None, "null", "", "N/A"]:
            return None

        # Validate numeric fields
        required_fields = ['quantity', 'unit_price', 'line_total']
        for field in required_fields:
            if item.get(field) in [None, "null", "", "N/A"]:
                logger.warning(f"  Missing {field} for item {item.get('item_name', 'unknown')}")
                return None

        # Parse and fix quantities
        quantity = float(item['quantity'])
        unit_price = float(item['unit_price'])
        line_total = float(item['line_total'])

        # Fix OCR quantity errors (1000 → 1.0, 2000 → 2.0, etc.)
        if quantity >= 1000:
            quantity = round(quantity / 1000, 1)
            item['quantity'] = str(quantity)
            logger.info(f"  Fixed quantity: {float(item['quantity']) * 1000:.0f} → {quantity}")

        # Validate calculation: quantity × unit_price = line_total
        expected_total = quantity * unit_price
        if abs(line_total - expected_total) > 0.01:
            # Try to fix quantity based on line_total
            if unit_price > 0:
                corrected_quantity = round(line_total / unit_price, 1)
                if 0 < corrected_quantity <= 100:
                    item['quantity'] = str(corrected_quantity)
                    logger.info(f"  Fixed calculation: {quantity} → {corrected_quantity}")
                else:
                    # Fix line_total instead
                    item['line_total'] = f"{expected_total:.2f}"

        # Warn about suspicious values
        if quantity > 50:
            logger.warning(f"  High quantity for {item['item_name']}: {quantity}")
        if unit_price > 500:
            logger.warning(f"  High price for {item['item_name']}: {unit_price} AZN")

        # Format monetary values
        monetary_fields = [
            'unit_price', 'line_total', 'subtotal', 'vat_18_percent', 'total_tax',
            'cashless_payment', 'cash_payment', 'bonus_payment', 'advance_payment',
            'credit_payment', 'refund_amount'
        ]

        for field in monetary_fields:
            if item.get(field) and item[field] not in [None, "null", ""]:
                try:
                    value = float(item[field])
                    item[field] = f"{value:.2f}"
                except (ValueError, TypeError):
                    item[field] = "0.00"

        # Clean item name
        if item.get('item_name'):
            name = item['item_name']
            # Remove VAT indicators and prefixes
            name = re.sub(r'^v?ƏDV[:\s]*\d*[:\s]*', '', name)
            name = re.sub(r'^"?ƏDV[:\s]*\d*[:\s]*', '', name)
            name = re.sub(r'^ƏDV-dən\s+azad\s+', '', name)
            name = re.sub(r'^Ticarət\s+əlavəsi[:\s]*\d*\s*', '', name)
            name = re.sub(r'^["\']+|["\']+$', '', name)
            name = re.sub(r'\s+', ' ', name).strip()
            item['item_name'] = name

        return item

    except Exception as e:
        logger.error(f"  Error validating item: {e}")
        return None


def load_checkpoint() -> Dict:
    """Load checkpoint data if it exists."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
                logger.info(f"📂 Checkpoint loaded: {checkpoint.get('processed_files', 0)} files completed")
                return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return {
        'processed_files': 0,
        'total_cost': 0.0,
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'last_processed_file': None,
        'timestamp': None
    }


def save_checkpoint(processed_files: int, last_file: str = None):
    """Save current progress to checkpoint file."""
    try:
        checkpoint = {
            'processed_files': processed_files,
            'total_cost': total_cost,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'last_processed_file': last_file,
            'timestamp': time.time()
        }

        # Atomic write
        temp_file = CHECKPOINT_FILE.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.replace(CHECKPOINT_FILE)

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def load_completed_files() -> set:
    """Load set of completed filenames."""
    completed = set()
    if COMPLETED_FILE.exists():
        try:
            with open(COMPLETED_FILE, 'r') as f:
                completed = set(line.strip() for line in f if line.strip())
            logger.info(f"📋 Loaded {len(completed)} completed files")
        except Exception as e:
            logger.warning(f"Failed to load completed files: {e}")
    return completed


def mark_file_completed(filename: str):
    """Mark a file as completed (append-only, crash-safe)."""
    try:
        with open(COMPLETED_FILE, 'a') as f:
            f.write(f"{filename}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        logger.error(f"Failed to mark file completed: {e}")


def save_partial_results(results: List[Dict], append: bool = True):
    """Save results to CSV, appending if file exists."""
    try:
        if not results:
            return

        df_new = pd.DataFrame(results)

        # Define column order
        columns = [
            'filename', 'store_name', 'store_address', 'store_code', 'taxpayer_name',
            'tax_id', 'receipt_number', 'cashier_name', 'date', 'time',
            'item_name', 'quantity', 'unit_price', 'line_total', 'subtotal',
            'vat_18_percent', 'total_tax', 'cashless_payment', 'cash_payment', 'bonus_payment',
            'advance_payment', 'credit_payment', 'queue_number', 'cash_register_model',
            'cash_register_serial', 'fiscal_id', 'fiscal_registration', 'refund_amount',
            'refund_date', 'refund_time'
        ]

        # Ensure all columns exist
        for col in columns:
            if col not in df_new.columns:
                df_new[col] = None
        df_new = df_new[columns]

        # Append or create new file
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

        if append and OUTPUT_CSV.exists():
            # Append without header
            df_new.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding='utf-8')
            logger.info(f"  💾 Appended {len(df_new)} rows to {OUTPUT_CSV}")
        else:
            # Write new file with header
            df_new.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
            logger.info(f"  💾 Created {OUTPUT_CSV} with {len(df_new)} rows")

    except Exception as e:
        logger.error(f"Failed to save partial results: {e}")


def create_fallback_data(filename: str) -> List[Dict]:
    """Create fallback data when extraction fails."""
    return [{
        'filename': filename,
        'store_name': None,
        'store_address': None,
        'store_code': None,
        'taxpayer_name': None,
        'tax_id': None,
        'receipt_number': None,
        'cashier_name': None,
        'date': None,
        'time': None,
        'item_name': None,
        'quantity': None,
        'unit_price': None,
        'line_total': None,
        'subtotal': None,
        'vat_18_percent': None,
        'total_tax': None,
        'cashless_payment': "0.00",
        'cash_payment': "0.00",
        'bonus_payment': "0.00",
        'advance_payment': "0.00",
        'credit_payment': "0.00",
        'queue_number': None,
        'cash_register_model': None,
        'cash_register_serial': None,
        'fiscal_id': None,
        'fiscal_registration': None,
        'refund_amount': None,
        'refund_date': None,
        'refund_time': None,
        'error': 'Extraction failed'
    }]


def process_receipt(image_path: Path, filename: str) -> List[Dict]:
    """Process a single receipt with vision AI."""

    global processed_count

    try:
        logger.info(f"Processing {filename}...")

        # Extract items using vision API
        items = extract_items_with_vision_ai(image_path, filename)

        if not items:
            logger.warning(f"  No items extracted, using fallback")
            items = create_fallback_data(filename)

        # Update progress
        with counter_lock:
            processed_count += 1

        return items

    except Exception as e:
        logger.error(f"  Unexpected error processing {filename}: {e}")
        with counter_lock:
            processed_count += 1
        return create_fallback_data(filename)


def process_batch(batch_files: List[Path], batch_num: int, total_batches: int, completed_files: set) -> List[Dict]:
    """Process a batch of receipts with controlled concurrency and interruption handling."""

    global stop_requested

    logger.info(f"\n{'='*60}")
    logger.info(f"Batch {batch_num}/{total_batches} - {len(batch_files)} receipts")
    logger.info(f"{'='*60}")

    batch_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all files
        future_to_file = {
            executor.submit(process_receipt, filepath, filepath.name): filepath
            for filepath in batch_files
        }

        # Collect results with timeout
        for future in as_completed(future_to_file, timeout=120):
            # Check for stop request
            if stop_requested:
                logger.warning("⚠️  Stop requested, cancelling remaining tasks...")
                for f in future_to_file:
                    f.cancel()
                break

            filepath = future_to_file[future]
            try:
                result = future.result(timeout=60)
                batch_results.extend(result)

                # Mark file as completed
                mark_file_completed(filepath.name)
                completed_files.add(filepath.name)

            except Exception as e:
                logger.error(f"  Failed {filepath.name}: {e}")
                batch_results.extend(create_fallback_data(filepath.name))

    return batch_results


def main():
    """Main function to run AI-enhanced receipt processing with resume capability."""

    global processed_count, total_cost, total_input_tokens, total_output_tokens, stop_requested

    logger.info("\n" + "="*70)
    logger.info("🚀 AI-Enhanced Receipt Parser with GPT-4o Vision")
    logger.info("="*70)

    # Validate directories
    if not RECEIPTS_DIR.exists():
        logger.error(f"Receipts directory not found: {RECEIPTS_DIR}")
        return

    # Load checkpoint and completed files
    checkpoint = load_checkpoint()
    completed_files = load_completed_files()

    # Restore state from checkpoint
    total_cost = checkpoint.get('total_cost', 0.0)
    total_input_tokens = checkpoint.get('total_input_tokens', 0)
    total_output_tokens = checkpoint.get('total_output_tokens', 0)

    # Get all image files
    all_image_files = sorted([
        f for f in RECEIPTS_DIR.iterdir()
        if f.suffix.lower() in ['.jpeg', '.jpg', '.png', '.tiff']
    ])

    # Filter out completed files
    image_files = [f for f in all_image_files if f.name not in completed_files]

    total_files = len(all_image_files)
    remaining_files = len(image_files)

    if remaining_files == 0:
        logger.info("✅ All files already processed!")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Total cost: ${total_cost:.4f}")
        return

    total_batches = (remaining_files + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info(f"\n📊 Configuration:")
    logger.info(f"   Total receipts: {total_files}")
    logger.info(f"   Already completed: {len(completed_files)}")
    logger.info(f"   Remaining: {remaining_files}")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Total batches: {total_batches}")
    logger.info(f"   Max workers: {MAX_WORKERS}")
    logger.info(f"   Model: GPT-4o with Vision")
    logger.info(f"   Output: {OUTPUT_CSV}")

    if len(completed_files) > 0:
        logger.info(f"\n🔄 Resuming from checkpoint...")
        logger.info(f"   Previous cost: ${total_cost:.4f}")

    # Estimate remaining cost
    avg_cost_per_receipt = 0.020  # Conservative estimate
    estimated_cost = remaining_files * avg_cost_per_receipt
    logger.info(f"\n💰 Cost Estimate:")
    logger.info(f"   Estimated for remaining {remaining_files} receipts: ${estimated_cost:.2f}")
    logger.info(f"   Total estimated (including completed): ${total_cost + estimated_cost:.2f}")

    # Process in batches
    start_time = time.time()
    append_mode = len(completed_files) > 0  # Append if resuming

    try:
        for i in range(0, remaining_files, BATCH_SIZE):
            if stop_requested:
                logger.warning("\n⚠️  Processing interrupted by user")
                break

            batch_files = image_files[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1

            batch_start = time.time()
            batch_results = process_batch(batch_files, batch_num, total_batches, completed_files)

            # Save results immediately after each batch (crash-safe)
            if batch_results:
                save_partial_results(batch_results, append=append_mode)
                append_mode = True  # After first batch, always append

            batch_time = time.time() - batch_start
            avg_time = batch_time / len(batch_files) if batch_files else 0

            # Update checkpoint
            processed_count = len(completed_files)
            save_checkpoint(processed_count, batch_files[-1].name if batch_files else None)

            # Progress update
            logger.info(f"\n✅ Batch {batch_num}/{total_batches} completed:")
            logger.info(f"   Time: {batch_time:.1f}s ({avg_time:.1f}s/receipt)")
            logger.info(f"   Items extracted: {len(batch_results)}")
            logger.info(f"   Total cost so far: ${total_cost:.4f}")
            logger.info(f"   Progress: {len(completed_files)}/{total_files} ({len(completed_files)/total_files*100:.1f}%)")

            # Rate limiting between batches
            if batch_num < total_batches and not stop_requested:
                logger.info(f"   Waiting 2s before next batch...")
                time.sleep(2)

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Force interrupted! Progress has been saved.")
        logger.warning(f"   Completed: {len(completed_files)}/{total_files} files")
        logger.warning(f"   Resume by running the script again")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        logger.error("Progress has been saved. Resume by running the script again.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Final statistics
    total_time = time.time() - start_time
    processed_this_run = len(completed_files) - len(all_image_files) + remaining_files

    logger.info(f"\n{'='*70}")
    if stop_requested:
        logger.info("⚠️  PROCESSING INTERRUPTED")
    else:
        logger.info("🎯 EXTRACTION COMPLETE")
    logger.info(f"{'='*70}")

    logger.info(f"\n📈 Statistics:")
    logger.info(f"   Total time this run: {total_time:.1f}s")
    logger.info(f"   Files processed this run: {processed_this_run}")
    logger.info(f"   Total files completed: {len(completed_files)}/{total_files}")
    logger.info(f"   Remaining: {total_files - len(completed_files)}")

    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        logger.info(f"   Total items in CSV: {len(df)}")
        logger.info(f"   Unique receipts: {len(df['filename'].unique())}")
        if len(df) > 0:
            logger.info(f"   Avg items/receipt: {len(df) / len(df['filename'].unique()):.1f}")

    logger.info(f"\n💰 Cost Analysis:")
    logger.info(f"   Total cost: ${total_cost:.4f}")
    if len(completed_files) > 0:
        logger.info(f"   Cost per receipt: ${total_cost / len(completed_files):.4f}")
    logger.info(f"   Input tokens: {total_input_tokens:,}")
    logger.info(f"   Output tokens: {total_output_tokens:,}")

    logger.info(f"\n✅ Data saved to: {OUTPUT_CSV}")
    logger.info(f"📋 Checkpoint: {CHECKPOINT_FILE}")
    logger.info(f"📝 Completed files: {COMPLETED_FILE}")

    if total_files - len(completed_files) > 0:
        logger.info(f"\n🔄 To resume, simply run the script again")

    logger.info(f"{'='*70}\n")


if __name__ == '__main__':
    main()

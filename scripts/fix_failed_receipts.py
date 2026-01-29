#!/usr/bin/env python3
"""
Fix the two failed receipts (7Dtzs6HZM8hK.jpeg and 8YHk6pGsqK6v.jpeg)
using a simplified approach with shorter prompts to avoid JSON truncation.
"""

import os
import json
import base64
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Get API key
api_key = os.getenv('OPENAI_API_KEY')
if api_key and api_key.startswith('sk-or-'):
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('OPENAI_API_KEY='):
                    api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                    break

client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

# Problem receipts
FAILED_RECEIPTS = ['7Dtzs6HZM8hK.jpeg', '8YHk6pGsqK6v.jpeg']
RECEIPTS_DIR = Path('../data/receipts')
OUTPUT_CSV = Path('../data/items.csv')

# Simplified schema - ask for shorter responses
SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
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
                    "date": {"type": ["string", "null"]},
                    "time": {"type": ["string", "null"]},
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
                "required": [
                    "filename", "store_name", "store_address", "store_code", "taxpayer_name",
                    "tax_id", "receipt_number", "cashier_name", "date", "time",
                    "item_name", "quantity", "unit_price", "line_total", "subtotal",
                    "vat_18_percent", "total_tax", "cashless_payment", "cash_payment",
                    "bonus_payment", "advance_payment", "credit_payment", "queue_number",
                    "cash_register_model", "cash_register_serial", "fiscal_id",
                    "fiscal_registration", "refund_amount", "refund_date", "refund_time"
                ],
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


def extract_receipt_simple(image_path: Path, filename: str):
    """Extract receipt using simpler prompt to avoid JSON truncation."""

    # Shorter, more concise prompt
    prompt = f"""Extract ALL items from this Azerbaijani receipt image.

IMPORTANT:
- Keep item names SHORT (max 30 chars)
- Extract ALL items visible
- Fix OCR errors (1000 → 1.0 for quantities)
- Use DD.MM.YYYY for dates, HH:MM:SS for times
- Return null if field not found

Filename: {filename}"""

    base64_image = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use mini model for shorter responses
        messages=[
            {
                "role": "system",
                "content": "You are a receipt data extractor. Extract data accurately and concisely."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
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
                "schema": SIMPLE_SCHEMA
            }
        },
        temperature=0.1,
        max_tokens=3000  # Limit response length
    )

    result = json.loads(response.choices[0].message.content)
    return result.get('items', [])


def update_csv(filename: str, new_items: list):
    """Replace failed rows in CSV with new data."""
    # Read existing CSV
    df = pd.read_csv(OUTPUT_CSV)

    # Remove old failed rows
    df = df[df['filename'] != filename]

    # Add new rows
    new_df = pd.DataFrame(new_items)
    df = pd.concat([df, new_df], ignore_index=True)

    # Save
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"✓ Updated {filename}: {len(new_items)} items")


def main():
    print("🔧 Fixing failed receipts...\n")

    for filename in FAILED_RECEIPTS:
        image_path = RECEIPTS_DIR / filename

        if not image_path.exists():
            print(f"✗ {filename} not found")
            continue

        print(f"Processing {filename}...")

        try:
            items = extract_receipt_simple(image_path, filename)

            if items and len(items) > 0:
                # Check if we have real data (not all nulls)
                has_data = any(item.get('item_name') for item in items)
                if has_data:
                    update_csv(filename, items)
                else:
                    print(f"✗ {filename}: No valid items found")
            else:
                print(f"✗ {filename}: Empty response")

        except Exception as e:
            print(f"✗ {filename}: {e}")

    print("\n✅ Done!")


if __name__ == '__main__':
    main()

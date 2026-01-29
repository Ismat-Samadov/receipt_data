#!/usr/bin/env python3
"""
Last attempt for 8YHk6pGsqK6v.jpeg - using unstructured JSON mode
and asking for only the most critical fields.
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

FAILED_RECEIPTS = ['J3oZk5eZquzg.jpeg', 'J7vwEWsHqgJP.jpeg']
RECEIPTS_DIR = Path('../data/receipts')
OUTPUT_CSV = Path('../data/items.csv')


def encode_image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_minimal(filename):
    """Extract with minimal fields and very short item names."""

    image_path = RECEIPTS_DIR / filename

    # Very minimal prompt
    prompt = f"""Extract items from this receipt. Return JSON with:
{{
  "store_name": "...",
  "date": "DD.MM.YYYY",
  "time": "HH:MM:SS",
  "items": [
    {{"name": "short_name", "qty": "1.0", "price": "1.50", "total": "1.50"}},
    ...
  ],
  "subtotal": "10.00",
  "total": "10.00"
}}

Keep item names under 20 characters. Extract ALL items."""

    base64_image = encode_image_to_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
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
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=2000
    )

    return json.loads(response.choices[0].message.content)


def convert_to_full_format(minimal_data, filename):
    """Convert minimal format to full 30-field format."""
    items = []

    store_name = minimal_data.get('store_name')
    date = minimal_data.get('date')
    time = minimal_data.get('time')
    subtotal = minimal_data.get('subtotal', '0.00')
    total = minimal_data.get('total', subtotal)

    for item in minimal_data.get('items', []):
        items.append({
            'filename': filename,
            'store_name': store_name,
            'store_address': None,
            'store_code': None,
            'taxpayer_name': None,
            'tax_id': None,
            'receipt_number': None,
            'cashier_name': None,
            'date': date,
            'time': time,
            'item_name': item.get('name'),
            'quantity': item.get('qty'),
            'unit_price': item.get('price'),
            'line_total': item.get('total'),
            'subtotal': subtotal,
            'vat_18_percent': None,
            'total_tax': None,
            'cashless_payment': None,
            'cash_payment': total,
            'bonus_payment': '0.00',
            'advance_payment': '0.00',
            'credit_payment': '0.00',
            'queue_number': None,
            'cash_register_model': None,
            'cash_register_serial': None,
            'fiscal_id': None,
            'fiscal_registration': None,
            'refund_amount': None,
            'refund_date': None,
            'refund_time': None
        })

    return items


def main():
    print(f"🔧 Fixing {len(FAILED_RECEIPTS)} failed receipts...\n")

    for filename in FAILED_RECEIPTS:
        print(f"Processing {filename}...")
        try:
            minimal_data = extract_minimal(filename)
            print(f"✓ Extracted minimal data: {len(minimal_data.get('items', []))} items")

            full_items = convert_to_full_format(minimal_data, filename)

            # Update CSV
            df = pd.read_csv(OUTPUT_CSV)
            df = df[df['filename'] != filename]
            new_df = pd.DataFrame(full_items)
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')

            print(f"✅ Updated {filename}: {len(full_items)} items\n")

        except Exception as e:
            print(f"✗ Failed: {e}\n")

    print("✅ Done!")


if __name__ == '__main__':
    main()

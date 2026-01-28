#!/usr/bin/env python3
"""Quick test of Moondream VLM for receipt OCR"""

import ollama
from pathlib import Path
import json

# Test on one receipt
receipts = list(Path("data/receipts").glob("*.jpeg"))
if not receipts:
    print("No receipts found!")
    exit(1)

receipt_path = receipts[0]
print(f"Testing Moondream on: {receipt_path.name}\n")

response = ollama.chat(
    model='moondream',
    messages=[{
        'role': 'user',
        'content': '''This is an Azerbaijani receipt. Extract all text and information.

Return a JSON object with:
{
  "store_name": "",
  "tax_id": "",
  "date": "",
  "time": "",
  "items": [
    {"name": "", "quantity": 0, "price": 0, "total": 0}
  ],
  "total": 0
}

Important Azerbaijani keywords:
- VÖEN = Tax ID
- Tarix = Date
- Vaxt = Time
- Kassir = Cashier
- Məhsulun adı = Product name
- Say = Quantity
- Qiymət = Price
- Cəmi / Yekun = Total''',
        'images': [str(receipt_path)]
    }]
)

print("="*60)
print("MOONDREAM RESPONSE:")
print("="*60)
print(response['message']['content'])
print("="*60)

# Try to parse as JSON
try:
    # Extract JSON from response
    import re
    json_match = re.search(r'\{.*\}', response['message']['content'], re.DOTALL)
    if json_match:
        data = json.loads(json_match.group())
        print("\n✓ Successfully parsed JSON!")
        print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"\n⚠ Could not parse as JSON: {e}")

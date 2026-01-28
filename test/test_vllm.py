#!/usr/bin/env python3
"""Quick test of VLM for receipt OCR"""

import ollama
from pathlib import Path

# Test on one receipt
receipt_path = Path("data/receipts/2R6KTHPDUq5Z.jpeg")

if not receipt_path.exists():
    # Find any receipt
    receipts = list(Path("data/receipts").glob("*.jpeg"))
    if receipts:
        receipt_path = receipts[0]
    else:
        print("No receipts found!")
        exit(1)

print(f"Testing VLM on: {receipt_path.name}\n")

response = ollama.chat(
    model='llava:7b-v1.6',
    messages=[{
        'role': 'user',
        'content': '''Extract information from this Azerbaijani receipt.

The receipt contains Azerbaijani text. Common keywords:
- "Vergi ödəyicisinin adı" = Store name
- "VÖEN" = Tax ID
- "Tarix" = Date
- "Məhsulun adı" = Product name
- "Say" = Quantity
- "Qiymət" = Price
- "Cəmi" = Total

Please extract:
1. Store name
2. Date
3. All items with quantities and prices
4. Total amount

Return the information in a structured format.''',
        'images': [str(receipt_path)]
    }]
)

print("="*60)
print("VLM RESPONSE:")
print("="*60)
print(response['message']['content'])
print("="*60)

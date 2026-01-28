#!/usr/bin/env python3
"""
VLM BENCHMARKING FOR AZERBAIJANI RECEIPT OCR
=============================================

Test multiple Vision Language Models on 5 sample receipts to find the best one.

Models to test:
1. moondream (1.7GB) - Lightweight, fast
2. llava:7b-v1.6 (4.7GB) - Balanced, accurate
3. qwen2-vl:2b (2.3GB) - Multilingual specialist

Output: Benchmark results saved to docs/vllm_benchmark.md
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import sys
from datetime import datetime

try:
    import ollama
except ImportError:
    print("Installing ollama-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
    import ollama

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

RECEIPTS_DIR = Path("../data/receipts")
OUTPUT_DIR = Path("../docs/benchmark")
BENCHMARK_REPORT = Path("../docs/vllm_benchmark.md")

# Test on first 5 receipts
NUM_TEST_RECEIPTS = 5

# Models to test
MODELS = [
    "moondream",
    "llava:7b-v1.6",
    # "qwen2-vl:2b"  # Uncomment if you want to test this too
]

# Extraction prompt (removed template structure to avoid confusion)
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

Return ONLY a valid JSON object with the actual extracted data from the image, no markdown, no explanations, no template placeholders."""

# ============================================================================
# BENCHMARKING
# ============================================================================

class VLMBenchmark:
    """Benchmark multiple VLMs for receipt extraction"""

    def __init__(self):
        self.results = {}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def check_model(self, model_name: str) -> bool:
        """Check if model is available"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True
            )
            return model_name in result.stdout
        except Exception as e:
            logger.error(f"Error checking model {model_name}: {e}")
            return False

    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from VLM response"""
        import re

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
            logger.warning(f"Could not parse JSON from response")
            return None

    def process_receipt(self, model_name: str, image_path: Path) -> Dict:
        """Process one receipt with a VLM"""
        logger.info(f"  Processing {image_path.name} with {model_name}...")

        start_time = time.time()

        try:
            response = ollama.chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': EXTRACTION_PROMPT,
                    'images': [str(image_path)]
                }]
            )

            elapsed = time.time() - start_time
            response_text = response['message']['content']

            # Parse JSON
            data = self.extract_json(response_text)

            result = {
                'filename': image_path.name,
                'model': model_name,
                'success': data is not None,
                'time_seconds': round(elapsed, 2),
                'items_count': len(data.get('items', [])) if data else 0,
                'has_store_name': bool(data.get('store_name')) if data else False,
                'has_date': bool(data.get('date')) if data else False,
                'has_tax_id': bool(data.get('tax_id')) if data else False,
                'extracted_data': data,
                'raw_response': response_text[:500]  # First 500 chars
            }

            # Check for placeholder text (template echo problem)
            if data:
                is_placeholder = (
                    data.get('store_name') == 'Store business name' or
                    data.get('store_name') == 'actual store name from receipt' or
                    data.get('date') == 'DD.MM.YYYY' or
                    not data.get('store_name')  # Empty/null
                )
                result['is_placeholder'] = is_placeholder
            else:
                result['is_placeholder'] = True

            logger.info(f"    ✓ {elapsed:.1f}s | Items: {result['items_count']} | Placeholder: {result['is_placeholder']}")

            return result

        except Exception as e:
            logger.error(f"    ✗ Error: {e}")
            return {
                'filename': image_path.name,
                'model': model_name,
                'success': False,
                'error': str(e),
                'time_seconds': time.time() - start_time
            }

    def benchmark_model(self, model_name: str, image_paths: List[Path]) -> Dict:
        """Benchmark one model on test receipts"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_name}")
        logger.info(f"{'='*60}")

        if not self.check_model(model_name):
            logger.warning(f"Model {model_name} not found. Downloading...")
            try:
                subprocess.run(['ollama', 'pull', model_name], check=True)
                logger.info(f"✓ Downloaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                return {'model': model_name, 'error': 'Model not available'}

        results = []
        for img_path in image_paths:
            result = self.process_receipt(model_name, img_path)
            results.append(result)

            # Save individual result
            output_file = OUTPUT_DIR / f"{model_name.replace(':', '_')}_{img_path.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        # Calculate statistics
        successful = [r for r in results if r.get('success')]
        with_items = [r for r in successful if r.get('items_count', 0) > 0]
        non_placeholders = [r for r in successful if not r.get('is_placeholder', True)]

        stats = {
            'model': model_name,
            'total_tested': len(results),
            'successful_parse': len(successful),
            'with_items': len(with_items),
            'real_data': len(non_placeholders),  # Not placeholder text
            'avg_time_seconds': round(sum(r.get('time_seconds', 0) for r in results) / len(results), 2),
            'avg_items_count': round(sum(r.get('items_count', 0) for r in with_items) / len(with_items), 1) if with_items else 0,
            'success_rate': round(len(non_placeholders) / len(results) * 100, 1),
            'results': results
        }

        logger.info(f"\n{model_name} Stats:")
        logger.info(f"  Success: {len(non_placeholders)}/{len(results)} ({stats['success_rate']}%)")
        logger.info(f"  Avg items: {stats['avg_items_count']}")
        logger.info(f"  Avg time: {stats['avg_time_seconds']}s")

        return stats

    def run_benchmark(self, image_paths: List[Path]):
        """Run benchmark on all models"""
        logger.info(f"\n{'='*60}")
        logger.info(f"VLM BENCHMARK FOR AZERBAIJANI RECEIPTS")
        logger.info(f"{'='*60}")
        logger.info(f"Testing {len(MODELS)} models on {len(image_paths)} receipts")
        logger.info(f"{'='*60}\n")

        for model in MODELS:
            stats = self.benchmark_model(model, image_paths)
            self.results[model] = stats

    def generate_report(self):
        """Generate markdown benchmark report"""
        logger.info(f"\nGenerating benchmark report...")

        report = f"""# VLM Benchmark for Azerbaijani Receipt OCR

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Test Set**: {NUM_TEST_RECEIPTS} receipts
**Models Tested**: {len(MODELS)}

## Summary

| Model | Success Rate | Avg Items | Avg Time (s) | Real Data* |
|-------|-------------|-----------|--------------|-----------|
"""

        # Sort by success rate
        sorted_results = sorted(
            self.results.values(),
            key=lambda x: x.get('success_rate', 0),
            reverse=True
        )

        for stats in sorted_results:
            if 'error' in stats:
                report += f"| {stats['model']} | ERROR | - | - | - |\n"
            else:
                report += f"| {stats['model']} | {stats['success_rate']}% | {stats['avg_items_count']} | {stats['avg_time_seconds']} | {stats['real_data']}/{stats['total_tested']} |\n"

        report += "\n*Real Data = Successfully extracted actual receipt data (not placeholder text)\n\n"

        # Best model
        best = sorted_results[0] if sorted_results else None
        if best and 'error' not in best:
            report += f"""## Recommendation

**Best Model**: `{best['model']}`

- Success Rate: {best['success_rate']}%
- Average Items Extracted: {best['avg_items_count']}
- Average Processing Time: {best['avg_time_seconds']} seconds
- Real Data Extractions: {best['real_data']}/{best['total_tested']}

Use this model for processing all {len(list(RECEIPTS_DIR.glob('*.jpeg')))} receipts.

"""

        # Detailed results
        report += "## Detailed Results\n\n"

        for stats in sorted_results:
            if 'error' in stats:
                report += f"### {stats['model']}\n\nERROR: {stats['error']}\n\n"
                continue

            report += f"### {stats['model']}\n\n"
            report += f"- **Success Rate**: {stats['success_rate']}%\n"
            report += f"- **Parsed Successfully**: {stats['successful_parse']}/{stats['total_tested']}\n"
            report += f"- **With Items**: {stats['with_items']}/{stats['total_tested']}\n"
            report += f"- **Real Data (not placeholders)**: {stats['real_data']}/{stats['total_tested']}\n"
            report += f"- **Avg Processing Time**: {stats['avg_time_seconds']}s\n"
            report += f"- **Avg Items per Receipt**: {stats['avg_items_count']}\n\n"

            # Sample results
            report += "**Sample Extractions**:\n\n"
            for result in stats['results'][:3]:  # First 3 receipts
                report += f"**{result['filename']}**:\n"
                if result.get('success'):
                    report += f"- Items: {result['items_count']}\n"
                    report += f"- Time: {result['time_seconds']}s\n"
                    report += f"- Store Name: {result['extracted_data'].get('store_name', 'N/A')}\n"
                    report += f"- Placeholder: {'Yes' if result.get('is_placeholder') else 'No'}\n"
                else:
                    report += f"- **FAILED**: {result.get('error', 'Unknown error')}\n"
                report += "\n"

        report += f"\n## Next Steps\n\n"
        report += f"1. Review sample extractions in `docs/benchmark/`\n"
        report += f"2. If satisfied with quality, run full extraction:\n"
        report += f"   ```bash\n"
        report += f"   cd parsing\n"
        report += f"   python vllm_ocr.py  # Will use best model\n"
        report += f"   ```\n"
        report += f"3. Output will be saved to `data/vllm_output.csv`\n"

        # Save report
        BENCHMARK_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(BENCHMARK_REPORT, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✓ Benchmark report saved to {BENCHMARK_REPORT}")

        return report


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Find test images
    image_paths = sorted(RECEIPTS_DIR.glob("*.jpeg"))
    if not image_paths:
        image_paths = sorted(RECEIPTS_DIR.glob("*.jpg"))

    if len(image_paths) < NUM_TEST_RECEIPTS:
        logger.error(f"Not enough images found. Need {NUM_TEST_RECEIPTS}, found {len(image_paths)}")
        return

    # Use first N receipts for testing
    test_images = image_paths[:NUM_TEST_RECEIPTS]

    logger.info(f"Selected test receipts:")
    for img in test_images:
        logger.info(f"  - {img.name}")

    # Run benchmark
    benchmark = VLMBenchmark()
    benchmark.run_benchmark(test_images)

    # Generate report
    report = benchmark.generate_report()

    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Report: {BENCHMARK_REPORT}")
    logger.info(f"Sample JSON: {OUTPUT_DIR}/")
    logger.info(f"{'='*60}\n")

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    for model, stats in benchmark.results.items():
        if 'error' not in stats:
            print(f"\n{model}:")
            print(f"  Success: {stats['success_rate']}% ({stats['real_data']}/{stats['total_tested']} real data)")
            print(f"  Avg items: {stats['avg_items_count']}")
            print(f"  Avg time: {stats['avg_time_seconds']}s")

    print("\n" + "="*60)
    print(f"Full report: {BENCHMARK_REPORT}")
    print("="*60)


if __name__ == "__main__":
    main()

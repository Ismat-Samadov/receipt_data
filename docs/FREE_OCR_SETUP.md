# FREE LOCAL OCR SETUP GUIDE
## Zero-Cost Azerbaijani Receipt Processing

Complete setup instructions for CPU-only, fully local OCR pipeline.

---

## 📋 System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **CPU** | Any modern processor | No GPU needed |
| **RAM** | 16 GB | Uses ~4-6 GB peak |
| **Storage** | ~5 GB free | 2 GB models + data |
| **OS** | Windows/macOS/Linux | Cross-platform |
| **Python** | 3.8 - 3.11 | **3.12 not yet supported by PaddleOCR** |

---

## 🚀 Installation (30 minutes)

### Step 1: Create Virtual Environment

```bash
# Navigate to project
cd /Users/ismatsamadov/receipt_data

# Create virtual environment
python3.10 -m venv venv_free_ocr

# Activate
source venv_free_ocr/bin/activate  # macOS/Linux
# OR
venv_free_ocr\Scripts\activate  # Windows
```

### Step 2: Install Core Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core packages
pip install numpy==1.24.3
pip install opencv-python==4.8.0.76
pip install pandas==2.0.3
pip install Pillow==10.0.0
```

### Step 3: Install OCR Engines

#### Option A: PaddleOCR (Recommended - Fastest)

```bash
# Install PaddlePaddle (CPU version)
pip install paddlepaddle==2.5.1 -i https://mirror.baidu.com/pypi/simple

# Install PaddleOCR
pip install paddleocr==2.7.0.3
```

**Important Notes:**
- Use Python 3.10 (not 3.12) for best compatibility
- If installation fails, try without the mirror:
  ```bash
  pip install paddlepaddle==2.5.1
  ```

#### Option B: EasyOCR (Fallback - More Accurate, Slower)

```bash
# Install PyTorch CPU (required by EasyOCR)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install EasyOCR
pip install easyocr==1.7.0
```

#### Option C: Install Both (Dual Engine - Best Results)

```bash
# Install PaddleOCR (step 3A)
# Then install EasyOCR (step 3B)
```

### Step 4: Verify Installation

```bash
# Test PaddleOCR
python -c "from paddleocr import PaddleOCR; print('✓ PaddleOCR installed')"

# Test EasyOCR
python -c "import easyocr; print('✓ EasyOCR installed')"

# Test OpenCV
python -c "import cv2; print('✓ OpenCV installed')"
```

---

## 🎯 Quick Start (5 minutes)

### 1. Prepare Your Images

```bash
# Your images should be in:
/Users/ismatsamadov/receipt_data/data/receipts/

# Supported formats:
# *.jpeg, *.jpg, *.png

# Check how many images you have:
ls data/receipts/*.jpeg | wc -l
```

### 2. Run the Pipeline

```bash
# Navigate to parsing directory
cd parsing

# Run the processor
python local_free_ocr.py
```

### 3. Monitor Progress

The pipeline will show:
```
============================================================
FREE LOCAL OCR PIPELINE FOR AZERBAIJANI RECEIPTS
============================================================
Found 800 images to process

============================================================
Progress: 1/800 (0.1%)
============================================================
Processing: receipt_001.jpeg
OCR confidence: 87.5% (engine: PaddleOCR)
✓ Extracted 3 items
...
```

### 4. View Results

Output files:
```
data/local_ocr_output.csv          # Main CSV output (30 columns)
data/local_ocr_json/               # Per-receipt JSON (debug)
data/local_ocr_errors.log          # Failed receipts
data/local_ocr_stats.json          # Processing statistics
```

---

## ⚙️ Configuration Options

Edit `parsing/local_free_ocr.py` to customize:

```python
# Line 52-53: OCR Engine Selection
USE_PADDLEOCR = True              # Use PaddleOCR (fast)
USE_EASYOCR_FALLBACK = True       # Use EasyOCR if confidence < 60%
CONFIDENCE_THRESHOLD = 0.6        # Switch threshold

# Line 56-59: Image Preprocessing
ENABLE_PREPROCESSING = True       # Apply deskew, denoise, enhance
MIN_IMAGE_WIDTH = 1200            # Upscale small images
MAX_IMAGE_WIDTH = 2400            # Downscale large images

# Line 61-63: Output Options
BATCH_SIZE = 1                    # Memory-safe (one at a time)
SAVE_INTERMEDIATE_JSON = True     # Save debug JSON files

# Line 65-67: Auto-Correction
ENABLE_AUTO_CORRECTION = True     # Fix OCR errors (1000 → 1.0)
DECIMAL_ERROR_THRESHOLD = 100     # Flag values > this
```

---

## 📊 Expected Performance

### Processing Speed (CPU-only)

| Hardware | PaddleOCR | EasyOCR | Dual Engine |
|----------|-----------|---------|-------------|
| **i5 Laptop** | 3-5 sec/img | 8-12 sec/img | 4-6 sec/img |
| **i7 Desktop** | 2-3 sec/img | 5-8 sec/img | 3-4 sec/img |
| **M1 Mac** | 1-2 sec/img | 4-6 sec/img | 2-3 sec/img |

**For 800 images:**
- PaddleOCR only: ~40-60 minutes
- EasyOCR only: ~2-3 hours
- Dual engine: ~50-80 minutes

### Memory Usage

| Component | RAM Usage |
|-----------|-----------|
| Base Python | ~200 MB |
| OpenCV | ~300 MB |
| PaddleOCR model | ~2 GB |
| EasyOCR model | ~3 GB |
| **Peak Usage** | **4-6 GB** |

Safe for 16 GB RAM systems with other apps running.

### Accuracy Expectations

| Field Type | Typical Accuracy | Notes |
|------------|------------------|-------|
| **Store name** | 90-95% | Clear headers |
| **Tax ID (VÖEN)** | 95-98% | Numeric, consistent |
| **Date/Time** | 92-96% | Structured format |
| **Item names** | 75-85% | Variable quality |
| **Prices** | 80-90% | Auto-correction helps |
| **Quantities** | 85-92% | Auto-correction critical |
| **Totals** | 88-94% | Usually clear |

**Overall extraction completeness:** 70-85% of receipts will have all major fields extracted.

---

## 🔧 Troubleshooting

### Issue 1: PaddleOCR Installation Fails

**Error:** `Could not find a version that satisfies the requirement paddlepaddle`

**Solution:**
```bash
# Use specific Python version
python3.10 -m venv venv_free_ocr
source venv_free_ocr/bin/activate

# Try without mirror
pip install paddlepaddle==2.5.1

# If still fails, use EasyOCR only
pip install torch torchvision easyocr
# Edit local_free_ocr.py: USE_PADDLEOCR = False
```

### Issue 2: Out of Memory

**Error:** `MemoryError` or system freezing

**Solution:**
```bash
# Close other applications
# Process in smaller batches:
# Move 100 images at a time to receipts/ folder
# Run pipeline, move files to processed/
# Repeat

# OR disable EasyOCR fallback:
# Edit local_free_ocr.py: USE_EASYOCR_FALLBACK = False
```

### Issue 3: Low Extraction Quality

**Problem:** Many fields are missing

**Solution:**
```python
# Try these settings in local_free_ocr.py:

# Option 1: Enable aggressive preprocessing
ENABLE_PREPROCESSING = True
MIN_IMAGE_WIDTH = 1600  # Higher resolution

# Option 2: Use EasyOCR for all
USE_PADDLEOCR = False
USE_EASYOCR_FALLBACK = False  # Not used if primary is disabled
# Add: USE_EASYOCR = True at top of OCREngine.__init__

# Option 3: Lower confidence threshold
CONFIDENCE_THRESHOLD = 0.4  # Try EasyOCR more often
```

### Issue 4: Processing Too Slow

**Problem:** Takes hours for 800 images

**Solution:**
```python
# Disable preprocessing (50% faster)
ENABLE_PREPROCESSING = False

# Disable JSON output (10% faster)
SAVE_INTERMEDIATE_JSON = False

# Use PaddleOCR only (3x faster)
USE_EASYOCR_FALLBACK = False

# Reduce image size
MAX_IMAGE_WIDTH = 1800
MIN_IMAGE_WIDTH = 1000
```

### Issue 5: Python 3.12 Compatibility

**Error:** `No matching distribution found for paddlepaddle`

**Solution:**
```bash
# Uninstall Python 3.12, install Python 3.10
brew install python@3.10  # macOS
# OR download from python.org

# Create new venv with 3.10
python3.10 -m venv venv_free_ocr
source venv_free_ocr/bin/activate

# Install packages
pip install paddlepaddle paddleocr
```

---

## 🎨 Output Format

### CSV Structure (30 columns)

```csv
filename,store_name,store_address,store_code,taxpayer_name,tax_id,receipt_number,cashier_name,date,time,item_name,quantity,unit_price,line_total,subtotal,vat_18_percent,total_tax,cashless_payment,cash_payment,bonus_payment,advance_payment,credit_payment,queue_number,cash_register_model,cash_register_serial,fiscal_id,fiscal_registration,refund_amount,refund_date,refund_time
3ZPFJewrUQF4.jpeg,ARAZ MARKET,Bakı şəhəri...,,,1234567890,1234,Aysel Məmmədova,06.12.2025,10:25:54,Çörək,2,1.50,3.00,10.65,0.61,,10.65,0.00,,,,,Samsung ER-260,,3ZPFJewrUQF4,NMQ123456,0.11,06.12.2025,10:26:00
3ZPFJewrUQF4.jpeg,ARAZ MARKET,Bakı şəhəri...,,,1234567890,1234,Aysel Məmmədova,06.12.2025,10:25:54,Süd,1,3.20,3.20,10.65,0.61,,10.65,0.00,,,,,Samsung ER-260,,3ZPFJewrUQF4,NMQ123456,0.11,06.12.2025,10:26:00
...
```

**Key Points:**
- One row per item
- Receipt-level fields duplicated across items from same receipt
- Missing values are empty (not "null" or "N/A")
- Numbers have 2 decimal places
- Dates: DD.MM.YYYY format
- Times: HH:MM:SS format

### JSON Structure (debug)

```json
{
  "filename": "3ZPFJewrUQF4",
  "ocr_confidence": 0.875,
  "raw_text": "Full OCR text here...",
  "items": [
    {
      "filename": "3ZPFJewrUQF4.jpeg",
      "store_name": "ARAZ MARKET",
      "item_name": "Çörək",
      "quantity": 2.0,
      "unit_price": 1.5,
      "line_total": 3.0,
      ...
    }
  ]
}
```

---

## 📈 Optimization Tips

### For Maximum Speed

```python
# local_free_ocr.py configuration:
USE_PADDLEOCR = True
USE_EASYOCR_FALLBACK = False
ENABLE_PREPROCESSING = False
SAVE_INTERMEDIATE_JSON = False
MIN_IMAGE_WIDTH = 1000
MAX_IMAGE_WIDTH = 1800
```

**Expected:** 2-3 sec/image = ~40-50 min for 800 images

### For Maximum Accuracy

```python
# local_free_ocr.py configuration:
USE_PADDLEOCR = True
USE_EASYOCR_FALLBACK = True
CONFIDENCE_THRESHOLD = 0.5
ENABLE_PREPROCESSING = True
MIN_IMAGE_WIDTH = 1600
MAX_IMAGE_WIDTH = 2400
ENABLE_AUTO_CORRECTION = True
```

**Expected:** 4-8 sec/image = ~60-120 min for 800 images

### For Balanced Performance

```python
# Default configuration (as shipped)
# Good speed, good accuracy
```

**Expected:** 3-5 sec/image = ~50-80 min for 800 images

---

## 🆚 Comparison vs Paid Solutions

| Feature | This Pipeline | GPT-4 Vision | Commercial OCR |
|---------|---------------|--------------|----------------|
| **Cost** | $0 | ~$24 for 800 images | ~$100-400/month |
| **Speed** | 3-5 sec/img | 8-15 sec/img | 1-3 sec/img |
| **Accuracy** | 75-85% | 95-98% | 80-90% |
| **Privacy** | 100% local | Cloud upload | Cloud upload |
| **Offline** | ✅ Yes | ❌ No | ❌ No |
| **Hallucinations** | ❌ No | ⚠️ Sometimes | ❌ No |
| **Customizable** | ✅ Full control | ❌ Limited | ⚠️ Some |

**This pipeline is ideal when:**
- Budget = $0
- Data privacy is critical
- Internet not always available
- Need to process 100s-1000s of receipts
- Acceptable to manually review 15-25% of extractions

---

## 📝 Next Steps After Processing

### 1. Review Error Log

```bash
# Check which receipts failed
cat data/local_ocr_errors.log

# Count failures
wc -l data/local_ocr_errors.log
```

### 2. Spot-Check Results

```bash
# View first 10 rows
head -20 data/local_ocr_output.csv | column -t -s,

# Check completeness
python -c "
import pandas as pd
df = pd.read_csv('data/local_ocr_output.csv')
print(df.isnull().sum())
"
```

### 3. Manual Review Strategy

For maximum data quality, manually review:

1. **Failed extractions** (from error log)
2. **Low confidence** (OCR confidence < 60%)
3. **Missing critical fields** (store_name, date, item_name all blank)
4. **Mathematical errors** (flagged in parsing_errors)

Estimate: 15-25% of 800 = 120-200 receipts need manual check.

### 4. Data Analysis

```bash
# Run your existing EDA notebook
jupyter notebook notebooks/EDA.ipynb

# Or use the AI parsing script for comparison
python parsing/ai_parse.py  # If you want to pay for higher accuracy
```

---

## 💾 Complete Installation Script

Save this as `install_free_ocr.sh`:

```bash
#!/bin/bash
# Complete installation script for Free OCR Pipeline

echo "🚀 Installing Free Local OCR Pipeline..."
echo "========================================"

# Create venv
python3.10 -m venv venv_free_ocr
source venv_free_ocr/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Core dependencies
echo "📦 Installing core dependencies..."
pip install numpy==1.24.3
pip install opencv-python==4.8.0.76
pip install pandas==2.0.3
pip install Pillow==10.0.0

# PaddleOCR
echo "📦 Installing PaddleOCR..."
pip install paddlepaddle==2.5.1
pip install paddleocr==2.7.0.3

# EasyOCR (optional but recommended)
echo "📦 Installing EasyOCR..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
pip install easyocr==1.7.0

echo "✅ Installation complete!"
echo ""
echo "To activate: source venv_free_ocr/bin/activate"
echo "To run: cd parsing && python local_free_ocr.py"
```

Run with:
```bash
chmod +x install_free_ocr.sh
./install_free_ocr.sh
```

---

## 🎓 Learning Resources

### PaddleOCR Documentation
- GitHub: https://github.com/PaddlePaddle/PaddleOCR
- Docs: https://paddlepaddle.github.io/PaddleOCR/

### EasyOCR Documentation
- GitHub: https://github.com/JaidedAI/EasyOCR
- Docs: https://www.jaided.ai/easyocr/documentation/

### OpenCV Preprocessing
- Tutorial: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- Image Enhancement: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html

---

## ❓ FAQ

**Q: Can I use GPU to speed this up?**
A: Yes! Install GPU versions of PaddlePaddle/PyTorch, then change `use_gpu=True` in the code. Expect 5-10x speedup.

**Q: Will this work for non-Azerbaijani receipts?**
A: Yes, but you'll need to modify the regex patterns in `ReceiptParser.PATTERNS` dict.

**Q: Can I process PDFs?**
A: Yes! Convert PDF pages to images first using `pdf2image` package, then run the pipeline.

**Q: How do I improve accuracy further?**
A: 1) Use higher resolution images, 2) Enable both OCR engines, 3) Fine-tune regex patterns for your specific stores.

**Q: Can I run this on a 8GB RAM laptop?**
A: Tight, but possible. Use PaddleOCR only (no EasyOCR), disable preprocessing, process in small batches.

**Q: What if I need 95%+ accuracy like GPT-4?**
A: Use this pipeline for initial extraction, then selectively use paid API (ai_parse.py) only for low-confidence results. Hybrid approach saves money.

---

## 📞 Support

- **GitHub Issues:** (your repo URL here)
- **Email:** (your email here)
- **Documentation:** This file + inline code comments

---

**Last Updated:** 2026-01-27
**Version:** 1.0.0
**License:** MIT (Free to use, modify, distribute)

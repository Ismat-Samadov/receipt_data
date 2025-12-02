import os
import re
import pandas as pd
from PIL import Image
import pytesseract
import logging

# Configure logging to hide unnecessary output and show errors
logging.basicConfig(level=logging.ERROR)

# --- CONFIGURATION ---
# Set the path to your Tesseract installation if it's not in your system's PATH
# For Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For Linux/macOS, it's often found automatically if installed via package managers.

RECEIPTS_DIR = 'data/receipts'
OUTPUT_CSV = 'data/traditional.csv'

def clean_item_name(item_name):
    """
    Clean item name by removing VAT codes and other unwanted prefixes.
    
    Args:
        item_name (str): Raw item name from OCR
        
    Returns:
        str: Cleaned item name
    """
    if not item_name:
        return ""
    
    # Remove VAT codes like "ƏDV: 189:", "ƏDV: 1894", "vƏDV: 189:", etc.
    item_name = re.sub(r'^v?ƏDV[:\s]*\d+[:\s]*', '', item_name)
    
    # Handle cases where VAT codes are in the middle with quotes or spaces
    item_name = re.sub(r'^"?ƏDV[:\s]*\d+[:\s]*', '', item_name)
    
    # Remove "ƏDV-dən azad" (VAT-free) prefix
    item_name = re.sub(r'^ƏDV-dən\s+azad\s+', '', item_name)
    
    # Remove "Ticarət əlavəsi:" prefix
    item_name = re.sub(r'^Ticarət\s+əlavəsi[:\s]*\d*\s*', '', item_name)
    
    # Remove quotes at the beginning and end
    item_name = re.sub(r'^["\']|["\']$', '', item_name)
    
    # Clean up extra whitespace
    item_name = re.sub(r'\s+', ' ', item_name).strip()
    
    return item_name

def parse_receipt_text(text, filename):
    """
    Enhanced parsing function that extracts all 25 required columns from Azerbaijani receipts.
    
    Args:
        text (str): The OCR-extracted text from a receipt.
        filename (str): The original filename of the receipt image.

    Returns:
        list: A list of dictionaries, where each dictionary represents one item
              on the receipt, including all 25 required columns.
    """
    
    # Enhanced regex patterns handling OCR variations and character encoding issues
    patterns = {
        # Store name should be extracted from taxpayer name (more reliable)
        'store_name_temp': r'Vergi\s*ödəyicisinin\s*adı[:\s]*(.*?)(?:\n.*?)?(?:\nVÖEN|\nMƏHDUD|\nCƏMİYYƏTİ|\n|$)',
        
        # Store address with variations
        'store_address': r'(?:Obyektin\s*ünvanı|fani)[:\s]*(.*?)(?:\n|$)',
        
        # Store code with variations  
        'store_code': r'(?:Obyektin\s*kodu|ÖV.*?obyektin\s*kodu)[:\s]*([\d\-]+)',
        
        # Taxpayer name - handle multiline
        'taxpayer_name': r'Vergi\s*ödəyicisinin\s*adı[:\s]*(.*?)(?:\n.*?)?(?:\nVÖEN|\nMƏHDUD|\nCƏMİYYƏTİ|\n|$)',
        
        # VOEN number (tax ID)
        'tax_id': r'VÖEN[:\s]*(\d+)',
        
        # Receipt number - handle special characters
        'receipt_number': r'Satış\s*çeki\s*[№#NоМә]*\s*(\d+)',
        
        # Cashier name - exclude date patterns
        'cashier_name': r'Kassir[:\s]*((?!Tarix)[^\n\d]*?)(?:\s+Tarix|\n|$)',
        
        # Date and time with better matching
        'datetime': r'Tarix[:\s]*(\d{2}\.\d{2}\.\d{4})\s*Vaxt[:\s]*(\d{2}:\d{2}:\d{2})',
        
        # Subtotal with variations
        'subtotal': r'Cəmi\s+(\d+\.\d{2})',
        
        # VAT 18% with multiple formats
        'vat_18_percent': r'ƏDV\s*18%?\s*=\s*(\d+\.\d{2})',
        
        # Total tax
        'total_tax': r'Toplam\s*vergi\s*=\s*(\d+\.\d{2})',
        
        # Payment methods - handle OCR variations
        'cashless_payment': r'Nağdsız[:\s]*(\d+\.\d{2})',
        'cash_payment': r'Nağd[:\s]*(\d+\.\d{2})',
        'bonus_payment': r'Bonus[:\s]*(\d+\.\d{2})',
        'advance_payment': r'Avans\s*\([^)]*\)[:\s]*(\d+\.\d{2})',
        'credit_payment': r'Nisyə[:\s]*(\d+\.\d{2})',
        
        # Queue number
        'queue_number': r'Növbə\s*ərzində\s*vurulmuş\s*çek\s*sayı[:\s]*(\d+)',
        
        # NKA model (cash register model)
        'cash_register_model': r'NKA-nın\s*modeli[:\s]*(.*?)(?:\n|$)',
        
        # NKA serial number (cash register serial)
        'cash_register_serial': r'NKA-nın\s*zavod\s*nömrəsi[:\s]*(.*?)(?:\n|$)',
        
        # Fiscal ID - handle İ/I variations
        'fiscal_id': r'Fiskal\s*[İI]D[:\s]*(\S+)',
        
        # NMQ registration (fiscal registration)
        'fiscal_registration': r'NMQ-nin\s*qeydiyyat\s*nömrəsi[:\s]*(.*?)(?:\n|$)',
        
        # Refund amount
        'refund_amount': r'Geri\s*qaytarılan\s*məbləğ[:\s]*(\d+\.\d{2})',
        
        # Refund date and time
        'refund_datetime': r'Geri\s*qaytarılma\s*tarixi[:\s]*(\d{2}\.\d{2}\.\d{4})\s+(\d{2}:\d{2})'
    }

    # Extract general receipt info
    data = {}
    
    # Initialize all 30 columns with None (replaced payment_methods with 5 payment types)
    columns = [
        'filename', 'store_name', 'store_address', 'store_code', 'taxpayer_name',
        'tax_id', 'receipt_number', 'cashier_name', 'date', 'time',
        'item_name', 'quantity', 'unit_price', 'line_total', 'subtotal',
        'vat_18_percent', 'total_tax', 'cashless_payment', 'cash_payment', 'bonus_payment', 'advance_payment', 'credit_payment',
        'queue_number', 'cash_register_model', 'cash_register_serial', 'fiscal_id', 'fiscal_registration',
        'refund_amount', 'refund_date', 'refund_time'
    ]
    
    for col in columns:
        data[col] = None
    
    # Set filename
    data['filename'] = filename
    
    # Extract fields using patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match:
            if key == 'datetime':
                data['date'] = match.group(1)
                data['time'] = match.group(2)
            elif key == 'refund_datetime':
                data['refund_date'] = match.group(1)
                data['refund_time'] = match.group(2)
            elif key == 'taxpayer_name':
                # Clean up taxpayer name by removing extra whitespace and newlines
                taxpayer_text = match.group(1).strip()
                # Remove quotes and extra formatting
                taxpayer_text = re.sub(r'["\']+', '', taxpayer_text)
                taxpayer_text = re.sub(r'\s+', ' ', taxpayer_text)
                data[key] = taxpayer_text
            elif key == 'store_name_temp':
                # Use taxpayer name as store name (more reliable)
                store_text = match.group(1).strip()
                # Remove quotes and extra formatting
                store_text = re.sub(r'["\']+', '', store_text)
                store_text = re.sub(r'\s+', ' ', store_text)
                data['store_name'] = store_text
            elif key == 'cashier_name':
                # Clean cashier name - remove common OCR artifacts
                cashier_text = match.group(1).strip()
                # Skip if it looks like a date
                if not re.match(r'.*\d{2}\.\d{2}\.\d{4}', cashier_text):
                    data[key] = cashier_text
            else:
                data[key] = match.group(1).strip()
    
    # Additional fallback patterns for critical missing fields
    if not data.get('fiscal_id'):
        # Try alternative fiscal ID patterns
        fiscal_alt = re.search(r'Fiskal\s*[İI]D[:\s]*(\w+)', text, re.IGNORECASE)
        if fiscal_alt:
            data['fiscal_id'] = fiscal_alt.group(1).strip()
    
    # Ensure store_name is set from taxpayer_name if not already set
    if not data.get('store_name') and data.get('taxpayer_name'):
        data['store_name'] = data['taxpayer_name']
    
    # Process payment methods - set individual values, defaulting to 0.00 if not found
    payment_types = ['cashless_payment', 'cash_payment', 'bonus_payment', 'advance_payment', 'credit_payment']
    
    for payment_type in payment_types:
        if data.get(payment_type):
            try:
                # Keep the extracted value as float, but format as string for CSV
                amount = float(data[payment_type])
                data[payment_type] = f"{amount:.2f}"
            except (ValueError, TypeError):
                data[payment_type] = "0.00"
        else:
            data[payment_type] = "0.00"
    
    # --- Enhanced Item Extraction ---
    items_data = []
    try:
        # Find the items section more precisely
        items_block_match = re.search(
            r'Məhsulun adı\s+Say\s+Qiymət\s+Cəmi\s*\n(.*?)(?=\n-+\s*\nCəmi|\nCəmi\s+\d+\.\d{2})', 
            text, re.DOTALL | re.MULTILINE
        )
        
        if items_block_match:
            items_block = items_block_match.group(1)
            # Split into lines and filter out VAT and empty lines
            item_lines = [line.strip() for line in items_block.split('\n') 
                         if line.strip() and not line.strip().startswith('*ƏDV')]

            i = 0
            while i < len(item_lines):
                line = item_lines[i].strip()
                
                # Enhanced regex for item parsing - handles various formats
                item_match = re.match(r'(.+?)\s*\(([^)]+)\)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)$', line)
                
                if not item_match:
                    # Try without unit indicator
                    item_match = re.match(r'(.+?)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)$', line)
                
                if item_match:
                    if len(item_match.groups()) == 5:  # With unit indicator
                        item_name = item_match.group(1).strip()
                        quantity = float(item_match.group(3))
                        unit_price = float(item_match.group(4))
                        line_total = float(item_match.group(5))
                    else:  # Without unit indicator
                        item_name = item_match.group(1).strip()
                        quantity = float(item_match.group(2))
                        unit_price = float(item_match.group(3))
                        line_total = float(item_match.group(4))

                    # Clean item name by removing VAT codes
                    item_name = clean_item_name(item_name)
                    
                    # Only add if item name is not empty after cleaning
                    if item_name:
                        items_data.append({
                            'item_name': item_name,
                            'quantity': quantity,
                            'unit_price': unit_price,
                            'line_total': line_total
                        })
                elif i + 1 < len(item_lines):
                    # Handle multi-line item names
                    combined_line = f"{line} {item_lines[i+1].strip()}"
                    item_match = re.match(r'(.+?)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)$', combined_line)
                    if item_match:
                        item_name = item_match.group(1).strip()
                        quantity = float(item_match.group(2))
                        unit_price = float(item_match.group(3))
                        line_total = float(item_match.group(4))

                        # Clean item name by removing VAT codes
                        item_name = clean_item_name(item_name)
                        
                        # Only add if item name is not empty after cleaning
                        if item_name:
                            items_data.append({
                                'item_name': item_name,
                                'quantity': quantity,
                                'unit_price': unit_price,
                                'line_total': line_total
                            })
                        i += 1
                i += 1
    except Exception as e:
        print(f"Could not parse items for {filename}: {e}")

    # If no items were parsed, create a single record with receipt-level info
    if not items_data:
        data['error'] = 'Item parsing failed'
        return [data]

    # Combine general receipt info with each item
    full_receipt_data = []
    for item in items_data:
        record = data.copy()
        record.update(item)
        full_receipt_data.append(record)
        
    return full_receipt_data

def process_receipts_folder(directory, output_file):
    """
    Processes all images in a directory, extracts receipt data, and saves to CSV.
    """
    all_receipts_data = []
    
    if not os.path.exists(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    print(f"Starting processing of receipts in '{directory}'...")
    
    # Get a list of image files to process
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpeg', '.jpg', '.png', '.tiff'))]
    
    if not image_files:
        print("No image files found in the directory.")
        return

    for filename in image_files:
        filepath = os.path.join(directory, filename)
        try:
            print(f"Processing {filename}...")
            # Use pytesseract to do OCR on the image, specifying Azerbaijani language
            text = pytesseract.image_to_string(Image.open(filepath), lang='aze')
            
            # Parse the extracted text
            parsed_data = parse_receipt_text(text, filename)
            if parsed_data:
                all_receipts_data.extend(parsed_data)
                
        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")
            all_receipts_data.append({'filename': filename, 'error': str(e)})

    if not all_receipts_data:
        print("No data could be extracted from any of the images.")
        return

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(all_receipts_data)
    
    # Define the exact 30 columns in the required order (replaced payment_methods with 5 payment types)
    column_order = [
        'filename', 'store_name', 'store_address', 'store_code', 'taxpayer_name',
        'tax_id', 'receipt_number', 'cashier_name', 'date', 'time',
        'item_name', 'quantity', 'unit_price', 'line_total', 'subtotal',
        'vat_18_percent', 'total_tax', 'cashless_payment', 'cash_payment', 'bonus_payment', 'advance_payment', 'credit_payment',
        'queue_number', 'cash_register_model', 'cash_register_serial', 'fiscal_id', 'fiscal_registration',
        'refund_amount', 'refund_date', 'refund_time'
    ]
    
    # Ensure all columns exist in the dataframe
    for col in column_order:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns to match the required 30-column structure
    df = df.reindex(columns=column_order)

    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Success! All data has been extracted and saved to '{output_file}'")


# --- RUN THE SCRIPT ---
if __name__ == '__main__':
    process_receipts_folder(RECEIPTS_DIR, OUTPUT_CSV)
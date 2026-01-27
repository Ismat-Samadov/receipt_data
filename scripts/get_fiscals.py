#!/usr/bin/env python3
"""
Kapital Bank Cashback History Scraper
Fetches fiscal IDs and receipt data from Kapital Bank's EDV portal API
"""

import requests
import json
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
import argparse

# Configuration
API_URL = "https://edvgerial.kapitalbank.az/api/v1/cashback/history"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "fiscals.csv"
PAGE_LIMIT = 20  # Records per page
REQUEST_DELAY = 1.0  # Delay between requests (seconds)
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Date range configuration
DATE_RANGE_DAYS = 365  # Last 365 days (1 year)
TO_DATE = datetime.now()
FROM_DATE = TO_DATE - timedelta(days=DATE_RANGE_DAYS)


class KapitalBankScraper:
    """Scraper for Kapital Bank cashback history API"""

    def __init__(self, xsrf_token=None, cookies=None):
        self.session = requests.Session()
        self.total_records = 0
        self.xsrf_token = xsrf_token or os.environ.get('XSRF_TOKEN')
        self.cookies = cookies or os.environ.get('KB_COOKIES')
        self.setup_headers()

    def setup_headers(self):
        """
        Setup authentication headers

        IMPORTANT: Provide tokens via environment variables or command-line:
        - XSRF_TOKEN: The xsrf-token from request headers
        - KB_COOKIES: The entire cookie string from request headers

        Or update the default values in this method from your browser's Network tab:
        1. Go to https://edvgerial.kapitalbank.az/az/dashboard
        2. Open DevTools (F12) -> Network tab
        3. Make a request to /api/v1/cashback/history
        4. Copy the cookies and xsrf-token from the request headers
        """

        self.session.headers.update({
            'accept': '*/*',
            'accept-language': 'az',
            'content-type': 'application/json',
            'origin': 'https://edvgerial.kapitalbank.az',
            'referer': 'https://edvgerial.kapitalbank.az/az/dashboard',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36',
            'xsrf-token': self.xsrf_token or '',
        })

        # Parse cookies if provided as string
        if self.cookies:
            cookie_dict = {}
            for item in self.cookies.split('; '):
                if '=' in item:
                    key, value = item.split('=', 1)
                    cookie_dict[key] = value
            self.session.cookies.update(cookie_dict)
        else:
            # Default cookies - Update from your browser
            self.session.cookies.update({
                '_csrf': 'GsXaXtAPBBtW9IaKcCv1pqJY',
                '__cf_bm': '8.71EWNh3JB1G_i7vkDolbAXSBu0nDoA_1HVYEPrsTk-1769540013-1.0.1.1-6SRbyDqoqeKA1ZRUVOEJlh3PS9yEujUhhZAYMH47mbinTKmFlif_HQP8GhcRxwfkaGKVaPVkqDGQFoLMElpv2t4KYFA09tUNLtfAiHaeg2o',
                '_cfuvid': 'z2cUFvMdzqXi7fZ3k0hhr5nBYhtWfSLTHveKeUbVNKg-1769540013328-0.0.1.1-604800000',
                '_ga': 'GA1.2.432138089.1769540013',
                '_gid': 'GA1.2.1613765365.1769540013',
                '_fbp': 'fb.1.1769540013423.8565377561980374',
                '_clck': '5bnshr%5E2%5Eg32%5E0%5E2218',
                '_gcl_au': '1.1.330248463.1769540018',
                'ac_session': 'eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiIrOTk0NTA0Nzg3NDYzIiwidGl0bGUiOiJTxo9Nxo9ET1YgxLBTTcaPVCBBWsaPUiBPxJ5MVSIsImdlbmRlciI6Ik1BTEUiLCJzdGF0dXMiOjIsInVpZCI6NjUxMzgsImF1dGgiOiJVU0VSIiwidG9rZW5fdHlwZSI6IkFDQ0VTUyIsImV4cCI6MTc2OTU0MDQ1NX0.UYqZf8FSLh_rXGFVN6ZNiNnJH0NsyQ2SYueMU-k-_01KAtRNeroxUanmZDHFhcDC6DS1ueT0WnPAJ_RLgkQ5gQ',
                'rf_session': 'eyJhbGciOiJIUzUxMiJ9.eyJ1aWQiOjY1MTM4LCJ0b2tlbl90eXBlIjoiUkVGUkVTSCIsImV4cCI6MTc2OTU0MTA1NX0.TKYgpR0vmBEZsDND3iou_Tr4zUQUZVmgbMnQpiBqJtkcEBPhjANxIHd9v0JTddqXtQwBr0cXPF_SyklAVDcxfg',
            })

    def fetch_page(self, page: int, from_date: datetime, to_date: datetime) -> dict:
        """
        Fetch a single page of cashback history

        Args:
            page: Page number (1-indexed)
            from_date: Start date for the query
            to_date: End date for the query

        Returns:
            JSON response as dict
        """
        payload = {
            "from": from_date.strftime("%Y-%m-%dT20:00:00Z"),
            "to": to_date.strftime("%Y-%m-%dT20:00:00Z"),
            "states": ["SCHEDULED", "COMPLETED", "FAIL"],
            "paging": {
                "page": page,
                "limit": PAGE_LIMIT
            }
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    API_URL,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    print(f"❌ Authentication failed. Please update your cookies and tokens in the script.")
                    sys.exit(1)
                elif response.status_code == 429:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(f"⚠️  Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️  HTTP {response.status_code} on page {page}. Retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(RETRY_DELAY)

            except requests.exceptions.RequestException as e:
                print(f"⚠️  Request error on page {page}: {e}. Retry {attempt + 1}/{MAX_RETRIES}...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        return None

    def extract_value(self, amount_obj: dict) -> str:
        """Extract value from amount object"""
        if amount_obj and isinstance(amount_obj, dict):
            return amount_obj.get('value', '0.00')
        return '0.00'

    def parse_record(self, record: dict) -> dict:
        """
        Parse a single record from API response

        Args:
            record: Single record from API data array

        Returns:
            Parsed record as dict
        """
        return {
            'id': record.get('id', ''),
            'fiscal_id': record.get('fiscalId', ''),
            'insert_date': record.get('insertDate', ''),
            'cheque_date': record.get('chequeDate', ''),
            'store_name': record.get('storeName', ''),
            'buy_amount': self.extract_value(record.get('buyAmount')),
            'cash_amount': self.extract_value(record.get('cashAmount')),
            'cashless_amount': self.extract_value(record.get('cashlessAmount')),
            'vat_amount': self.extract_value(record.get('vatAmount')),
            'refund_amount': self.extract_value(record.get('refundAmount')),
            'state': record.get('state', ''),
            'channel_type': record.get('channelType', ''),
            'cheque_status_message': record.get('chequeStatusMessage', ''),
            'with_card': record.get('withCard', ''),
            'card_cashback_amount': self.extract_value(record.get('cardCashbackAmount')) if record.get('cardCashbackAmount') else '0.00'
        }

    def fetch_all_records(self, from_date: datetime, to_date: datetime) -> list:
        """
        Fetch all records with pagination

        Args:
            from_date: Start date for the query
            to_date: End date for the query

        Returns:
            List of all parsed records
        """
        all_records = []
        page = 1

        print(f"📅 Fetching records from {from_date.date()} to {to_date.date()}...")

        while True:
            print(f"📄 Fetching page {page}...", end=' ')

            response = self.fetch_page(page, from_date, to_date)

            if not response:
                print("❌ Failed to fetch page")
                break

            if response.get('code') != 200:
                print(f"❌ API error: {response.get('code')}")
                break

            data = response.get('data', [])

            if not data:
                print("✓ No more records")
                break

            print(f"✓ Found {len(data)} records")

            for record in data:
                parsed = self.parse_record(record)
                all_records.append(parsed)

            self.total_records += len(data)

            # If we got fewer records than the page limit, we've reached the end
            if len(data) < PAGE_LIMIT:
                break

            page += 1
            time.sleep(REQUEST_DELAY)

        return all_records

    def save_to_csv(self, records: list):
        """
        Save records to CSV file

        Args:
            records: List of parsed records
        """
        if not records:
            print("⚠️  No records to save")
            return

        # Ensure output directory exists
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

        # Define CSV columns
        fieldnames = [
            'id', 'fiscal_id', 'insert_date', 'cheque_date', 'store_name',
            'buy_amount', 'cash_amount', 'cashless_amount', 'vat_amount',
            'refund_amount', 'state', 'channel_type', 'cheque_status_message',
            'with_card', 'card_cashback_amount'
        ]

        # Write CSV
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"✅ Saved {len(records)} records to {OUTPUT_CSV}")

    def run(self):
        """Main execution method"""
        print("=" * 60)
        print("Kapital Bank Cashback History Scraper")
        print("=" * 60)

        try:
            records = self.fetch_all_records(FROM_DATE, TO_DATE)
            self.save_to_csv(records)

            print("\n" + "=" * 60)
            print(f"✅ Successfully fetched {self.total_records} records")
            print(f"📊 Output saved to: {OUTPUT_CSV}")
            print("=" * 60)

            # Print summary statistics
            if records:
                print("\n📈 Summary Statistics:")
                print(f"   Total Receipts: {len(records)}")
                print(f"   Unique Stores: {len(set(r['store_name'] for r in records))}")
                total_amount = sum(float(r['buy_amount']) for r in records)
                print(f"   Total Amount: {total_amount:.2f} AZN")
                completed = sum(1 for r in records if r['state'] == 'COMPLETED')
                print(f"   Completed: {completed} ({completed/len(records)*100:.1f}%)")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(
        description='Fetch fiscal IDs from Kapital Bank EDV API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Using environment variables
  export XSRF_TOKEN="your_token_here"
  export KB_COOKIES="cookie1=value1; cookie2=value2; ..."
  python scripts/get_fiscals.py

  # Using command-line arguments
  python scripts/get_fiscals.py --xsrf-token "your_token" --cookies "cookie_string"
        '''
    )
    parser.add_argument('--xsrf-token', help='XSRF token from browser request headers')
    parser.add_argument('--cookies', help='Cookie string from browser request headers')

    args = parser.parse_args()

    scraper = KapitalBankScraper(xsrf_token=args.xsrf_token, cookies=args.cookies)
    scraper.run()


if __name__ == "__main__":
    main()

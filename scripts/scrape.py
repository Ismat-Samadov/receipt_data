import requests
import os
import time
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
# Base URL for the receipts
BASE_URL = "https://monitoring.e-kassa.gov.az/pks-monitoring/2.0.0/documents/"

# Input file containing fiscal IDs (one ID per line)
FISCAL_IDS_FILE = "data/ids.txt"

# Output directory to save the downloaded receipts
OUTPUT_DIR = "data/receipts"

# Delay in seconds between each request (after successful download or max retries)
# This is for politeness, the retry mechanism handles delays for failed attempts.
REQUEST_DELAY_SECONDS = 2.0

# --- Headers for mimicking a browser request ---
COMMON_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "monitoring.e-kassa.gov.az",
    # Updated Referer to the new URL requested by the user
    "Referer": "https://monitoring.e-kassa.gov.az/#/index",
    "Sec-Ch-Ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "User-Lang": "az",
    "User-Time-Zone": "Asia/Baku",
    "X-Csrf-Token": "", # This will be dynamically updated
}

# --- Script Logic ---

def create_output_directory(directory):
    """Creates the output directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    print(f"Ensured output directory '{directory}' exists.")

def setup_session():
    """Configures a requests Session with retries and common headers."""
    session = requests.Session()

    # Define retry strategy
    # Increased total retries to 10
    retries = Retry(
        total=10, # Increased total retries
        backoff_factor=1, # 1 second initial delay, then 2, 4, 8, 16 seconds etc.
        status_forcelist=[500, 502, 503, 504, 429],
        allowed_methods=frozenset(['GET']), # Only retry GET requests
        raise_on_status=False # Do not raise exception for status codes in status_forcelist
    )

    # Mount the retry strategy to the session
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # Set default timeout for all requests in this session (connect, read)
    session.timeout = (30, 90)

    # Update session headers
    session.headers.update(COMMON_HEADERS)

    return session

def get_csrf_token(session, main_url):
    """
    Attempts to fetch the CSRF token from the main website using the session.
    This simulates visiting the page to get a valid token.
    """
    # Add a small initial delay before attempting to fetch the CSRF token
    print(f"Waiting {REQUEST_DELAY_SECONDS} seconds before fetching CSRF token...")
    time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Attempting to fetch CSRF token from: {main_url}")
    try:
        # Use the session for the GET request. Session's timeout and retry apply.
        # Use the updated main_url from COMMON_HEADERS["Referer"]
        response = session.get(main_url)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for a meta tag or input field that might contain the CSRF token
        csrf_meta = soup.find('meta', attrs={'name': 'csrf-token'})
        if csrf_meta and 'content' in csrf_meta.attrs:
            token = csrf_meta['content']
            print(f"Found CSRF token from meta tag: {token[:10]}...")
            return token

        csrf_input = soup.find('input', attrs={'name': '_csrf'})
        if csrf_input and 'value' in csrf_input.attrs:
            token = csrf_input['value']
            print(f"Found CSRF token from input field: {token[:10]}...")
            return token

        print("CSRF token not found on the main page. Proceeding without it.")
        return "" # Return empty if not found
    except requests.exceptions.Timeout:
        print(f"Timeout occurred while fetching CSRF token from {main_url}. (Session retries exhausted)")
        return ""
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error while fetching CSRF token from {main_url}: {e}. (Session retries exhausted)")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"An unhandled request error occurred while fetching CSRF token from {main_url}: {e}")
        return ""

def read_fiscal_ids(file_path):
    """Reads fiscal IDs from a text file, one per line."""
    fiscal_ids = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                fiscal_ids.append(line.strip())
        print(f"Successfully read {len(fiscal_ids)} fiscal IDs from '{file_path}'.")
    except FileNotFoundError:
        print(f"Error: Fiscal IDs file '{file_path}' not found. Please create it.")
        exit()
    return fiscal_ids

def download_receipt(session, fiscal_id, output_dir, delay):
    """Downloads a single receipt and saves it to the specified directory."""
    url = f"{BASE_URL}{fiscal_id}"
    file_path = os.path.join(output_dir, f"{fiscal_id}.jpeg")

    if os.path.exists(file_path):
        print(f"Skipping {fiscal_id}: File already exists at '{file_path}'.")
        return True

    try:
        print(f"Attempting to download: {url}")
        # Use the session for the GET request. Session's timeout and retry apply.
        response = session.get(url, stream=True)

        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded: {fiscal_id} to '{file_path}'")
            return True
        else:
            print(f"Failed to download {fiscal_id}: Status Code {response.status_code}, URL: {url}")
            return False
    except requests.exceptions.Timeout:
        print(f"Timeout occurred while downloading {fiscal_id} from {url}. (Session retries handled)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error while downloading {fiscal_id} from {url}: {e}. (Session retries handled)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"An unhandled request error occurred while downloading {fiscal_id} from {url}: {e}")
        return False
    finally:
        # This delay is for politeness between different fiscal ID downloads,
        # not for retries of the same fiscal ID.
        time.sleep(delay)

def main():
    create_output_directory(OUTPUT_DIR)
    session = setup_session()

    # Get CSRF token before starting downloads
    # Pass the updated Referer URL as the main_url for CSRF token fetch
    csrf_token = get_csrf_token(session, COMMON_HEADERS["Referer"])
    if csrf_token:
        session.headers["X-Csrf-Token"] = csrf_token
        print(f"Updated session with CSRF token: {csrf_token[:10]}...")
    else:
        print("Warning: Could not obtain CSRF token. Downloads might still fail if it's strictly required.")

    fiscal_ids = read_fiscal_ids(FISCAL_IDS_FILE)

    if not fiscal_ids:
        print("No fiscal IDs found to process. Exiting.")
        return

    successful_downloads = 0
    failed_downloads = 0

    for i, fiscal_id in enumerate(fiscal_ids):
        print(f"\nProcessing {i+1}/{len(fiscal_ids)}: {fiscal_id}")
        if download_receipt(session, fiscal_id, OUTPUT_DIR, REQUEST_DELAY_SECONDS):
            successful_downloads += 1
        else:
            failed_downloads += 1

    print("\n--- Download Summary ---")
    print(f"Total IDs processed: {len(fiscal_ids)}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Receipts saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()

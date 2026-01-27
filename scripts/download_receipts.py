#!/usr/bin/env python3
"""
Crash-proof fiscal receipts downloader.

Features:
- Reads fiscal IDs from CSV (configurable column) or txt fallback.
- Resumable: maintains a `completed_ids` file and skips already downloaded IDs.
- Atomic file writes: downloads to `.part` temporary file then renames.
- Graceful shutdown on SIGINT / SIGTERM, saving progress.
- Configurable limit to process first N IDs (e.g., 1000 or 2000).
- Robust retries via requests + urllib3 Retry.
- Helpful logging to console and optional log file.
"""

import requests
import os
import time
import signal
import sys
import csv
import json
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Set

# Optional pandas usage (better CSV handling). Falls back to csv module if not present.
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

# --------------------- CONFIGURATION --------------------- #
BASE_URL = "https://monitoring.e-kassa.gov.az/pks-monitoring/2.0.0/documents/"

# Input file (CSV preferred). You can also point to a .txt (one id per line).
FISCAL_IDS_FILE = "../data/fiscal_ids_from_api.txt"

# CSV settings
FISCAL_ID_COLUMN = "FISCAL_ID"  # column name (str) or integer index (0-based) if CSV_HAS_HEADER=False
CSV_DELIMITER = ","
CSV_ENCODING = "utf-8"
CSV_HAS_HEADER = True

# Output directory to save the downloaded receipts
OUTPUT_DIR = "../data/receipts"

# Where we store list of completed ids (one per line) for resumability
COMPLETED_IDS_FILE = "../data/receipts_completed.txt"

# Optional checkpoint file (json) that stores summary counts and last index
CHECKPOINT_FILE = "../data/download_checkpoint.json"

# Politeness and request settings
REQUEST_DELAY_SECONDS = 2.0
COMMON_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "monitoring.e-kassa.gov.az",
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
    "X-Csrf-Token": "",
}

# Crash-proof/resume options
# Set MAX_IDS to None to process all ids. Or set to 1000 or 2000 etc.
MAX_IDS = None  # Process all IDs from API

# File extension for saved receipts
OUTPUT_EXT = ".jpeg"

# Retry configuration for requests
RETRY_TOTAL = 10
RETRY_BACKOFF_FACTOR = 1
RETRY_STATUS_FORCELIST = [500, 502, 503, 504, 429]

# Timeout used on requests.get (connect, read)
REQUEST_TIMEOUT = (30, 90)

# --------------------------------------------------------- #

_stop_requested = False  # set by signal handler to stop gracefully


def _signal_handler(signum, frame):
    global _stop_requested
    print(f"\nReceived signal {signum}. Requesting graceful shutdown...")
    _stop_requested = True


# register for common termination signals
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def create_output_directory(directory: str):
    os.makedirs(directory, exist_ok=True)
    print(f"Ensured output directory '{directory}' exists.")


def setup_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=frozenset(['GET']),
        raise_on_status=False
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # store in attribute for later use when calling session.get(...)
    session.request_timeout = REQUEST_TIMEOUT
    session.headers.update(COMMON_HEADERS)
    return session


def get_csrf_token(session: requests.Session, main_url: str) -> str:
    """Try to extract CSRF token from the provided main URL. Non-fatal failure returns empty string."""
    print(f"Waiting {REQUEST_DELAY_SECONDS} seconds before fetching CSRF token...")
    time.sleep(REQUEST_DELAY_SECONDS)
    print(f"Attempting to fetch CSRF token from: {main_url}")
    try:
        resp = session.get(main_url, timeout=(10, 20))
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        csrf_meta = soup.find('meta', attrs={'name': 'csrf-token'})
        if csrf_meta and 'content' in csrf_meta.attrs:
            token = csrf_meta['content']
            print("Found CSRF token from meta tag.")
            return token
        csrf_input = soup.find('input', attrs={'name': '_csrf'})
        if csrf_input and 'value' in csrf_input.attrs:
            token = csrf_input['value']
            print("Found CSRF token from input field.")
            return token
        print("CSRF token not found on the main page. Proceeding without it.")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"CSRF token fetch failed: {e}. Proceeding without CSRF token.")
        return ""


def read_completed_ids(path: str) -> Set[str]:
    """Read completed ids from file, return a set. If file missing, returns empty set."""
    completed = set()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    v = line.strip()
                    if v:
                        completed.add(v)
            print(f"Loaded {len(completed)} completed IDs from '{path}'.")
        except Exception as e:
            print(f"Warning: failed to read completed ids file '{path}': {e}")
    return completed


def append_completed_id_atomic(path: str, fiscal_id: str):
    """Append a single completed id to file and fsync to make it durable."""
    try:
        # open with append, write line, flush & fsync
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"{fiscal_id}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Warning: failed to persist completed id '{fiscal_id}' to '{path}': {e}")


def read_fiscal_ids(file_path: str,
                    column=FISCAL_ID_COLUMN,
                    delimiter=CSV_DELIMITER,
                    encoding=CSV_ENCODING,
                    has_header=CSV_HAS_HEADER) -> List[str]:
    """Reads fiscal ids from CSV or txt. Returns list of strings (cleaned)."""
    fiscal_ids = []

    if not os.path.exists(file_path):
        print(f"Error: Fiscal IDs file '{file_path}' not found. Please create it.")
        return fiscal_ids

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        # fallback: simple text file, one ID per line
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                val = line.strip()
                if val:
                    fiscal_ids.append(val)
        print(f"Read {len(fiscal_ids)} fiscal IDs from text file '{file_path}'.")
        return fiscal_ids

    # CSV handling
    if _HAS_PANDAS:
        try:
            if has_header:
                df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, dtype=str)
                if isinstance(column, int):
                    colname = df.columns[column]
                    raw = df[colname]
                else:
                    if column not in df.columns:
                        raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")
                    raw = df[column]
            else:
                df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, header=None, dtype=str)
                if not isinstance(column, int):
                    raise ValueError("CSV_HAS_HEADER=False but FISCAL_ID column is not an integer index.")
                raw = df.iloc[:, column]
            raw = raw.dropna().astype(str).str.strip()
            fiscal_ids = [v for v in raw.tolist() if v != ""]
            print(f"Read {len(fiscal_ids)} fiscal IDs from CSV '{file_path}' using pandas.")
            return fiscal_ids
        except Exception as e:
            print(f"Pandas read failed ({e}), falling back to csv module.")

    # fallback to csv module
    try:
        with open(file_path, newline='', encoding=encoding) as csvfile:
            if has_header:
                reader = csv.DictReader(csvfile, delimiter=delimiter)
                if isinstance(column, int):
                    raise ValueError("CSV_HAS_HEADER=True but column provided as integer index. Provide a column name.")
                if column not in reader.fieldnames:
                    raise KeyError(f"Column '{column}' not found in CSV. Available columns: {reader.fieldnames}")
                for row in reader:
                    val = row.get(column, "")
                    if val is not None:
                        val = val.strip()
                        if val:
                            fiscal_ids.append(val)
            else:
                reader = csv.reader(csvfile, delimiter=delimiter)
                if not isinstance(column, int):
                    raise ValueError("CSV_HAS_HEADER=False but column is not an integer index.")
                for row in reader:
                    if len(row) > column:
                        val = row[column].strip()
                        if val:
                            fiscal_ids.append(val)
        print(f"Read {len(fiscal_ids)} fiscal IDs from CSV '{file_path}' using csv module.")
    except Exception as e:
        print(f"Failed to read fiscal IDs from CSV: {e}")

    return fiscal_ids


def _download_to_temp_and_move(response: requests.Response, final_path: str):
    """Write stream to a .part temp file and atomically replace final path."""
    temp_path = final_path + ".part"
    try:
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
        # atomic rename/replace
        os.replace(temp_path, final_path)
    finally:
        # if temp file remains for some reason and final does not exist, leave it for inspection.
        if os.path.exists(temp_path) and not os.path.exists(final_path):
            print(f"Notice: temporary file left at '{temp_path}' (download may have been interrupted).")


def download_receipt(session: requests.Session, fiscal_id: str, output_dir: str, delay: float) -> bool:
    """
    Downloads receipt to output_dir/fiscal_id{OUTPUT_EXT}, using streaming and atomic write.
    Returns True on success, False on failure.
    """
    url = f"{BASE_URL}{fiscal_id}"
    safe_filename = f"{fiscal_id}{OUTPUT_EXT}"
    file_path = os.path.join(output_dir, safe_filename)

    if os.path.exists(file_path):
        # Already exists — assume completed
        return True

    try:
        print(f"Downloading {fiscal_id} from {url} ...")
        timeout = getattr(session, "request_timeout", REQUEST_TIMEOUT)
        resp = session.get(url, stream=True, timeout=timeout)
        if resp.status_code == 200:
            _download_to_temp_and_move(resp, file_path)
            print(f"Saved: {file_path}")
            return True
        else:
            print(f"Failed to download {fiscal_id}: HTTP {resp.status_code} ({url})")
            return False
    except requests.exceptions.Timeout:
        print(f"Timeout while downloading {fiscal_id}.")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error while downloading {fiscal_id}: {e}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Request error while downloading {fiscal_id}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error while saving {fiscal_id}: {e}")
        return False
    finally:
        # politeness delay - always executed
        time.sleep(delay)


def persist_checkpoint(path: str, summary: dict):
    """Save a small json checkpoint (best-effort)."""
    try:
        tmp = f"{path}.part"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception as e:
        print(f"Warning: failed to persist checkpoint to '{path}': {e}")


def main():
    global _stop_requested

    create_output_directory(OUTPUT_DIR)
    session = setup_session()

    # Get CSRF token non-fatal
    csrf_token = get_csrf_token(session, COMMON_HEADERS["Referer"])
    if csrf_token:
        session.headers["X-Csrf-Token"] = csrf_token

    # Read completed IDs
    completed = read_completed_ids(COMPLETED_IDS_FILE)

    # Read all fiscal ids
    all_ids = read_fiscal_ids(
        FISCAL_IDS_FILE,
        column=FISCAL_ID_COLUMN,
        delimiter=CSV_DELIMITER,
        encoding=CSV_ENCODING,
        has_header=CSV_HAS_HEADER
    )

    if not all_ids:
        print("No fiscal IDs found. Exiting.")
        return

    # Deduplicate while preserving original order
    seen = set()
    deduped = []
    for v in all_ids:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    all_ids = deduped

    # Apply MAX_IDS limit if configured
    if MAX_IDS is not None:
        if not isinstance(MAX_IDS, int) or MAX_IDS <= 0:
            print("Warning: MAX_IDS is set but not a positive integer. Ignoring MAX_IDS.")
        else:
            print(f"Processing only first {MAX_IDS} fiscal IDs (configured).")
            all_ids = all_ids[:MAX_IDS]

    # Filter out already completed
    to_process = [fid for fid in all_ids if fid not in completed]
    total_to_process = len(to_process)
    print(f"Total IDs: {len(all_ids)} | Already completed: {len(completed & set(all_ids))} | To process now: {total_to_process}")

    if total_to_process == 0:
        print("Nothing to do. All configured IDs already processed.")
        return

    successful = 0
    failed = 0
    start_index = 0

    try:
        for idx, fiscal_id in enumerate(to_process, start=1):
            # Stop if signal requested
            if _stop_requested:
                print("Shutdown requested: breaking processing loop.")
                break

            print(f"\nProcessing {idx}/{total_to_process}: {fiscal_id}")
            ok = download_receipt(session, fiscal_id, OUTPUT_DIR, REQUEST_DELAY_SECONDS)
            if ok:
                # Persist completed id immediately (durable append)
                append_completed_id_atomic(COMPLETED_IDS_FILE, fiscal_id)
                successful += 1
            else:
                print(f"Download failed for {fiscal_id}. Will mark as failed and continue.")
                failed += 1

            # Persist checkpoint summary after each item (best-effort)
            summary = {
                "timestamp": time.time(),
                "processed": idx,
                "total_to_process": total_to_process,
                "successful_since_start": successful,
                "failed_since_start": failed,
                "remaining": total_to_process - idx,
            }
            persist_checkpoint(CHECKPOINT_FILE, summary)

            # optional: short sleep already done in download_receipt finally.
            start_index = idx

        # final summary save
        final_summary = {
            "timestamp": time.time(),
            "total_ids_configured": len(all_ids),
            "total_to_process_initial": total_to_process,
            "processed": start_index,
            "successful": successful,
            "failed": failed,
        }
        persist_checkpoint(CHECKPOINT_FILE, final_summary)
        print("\n--- Run Summary ---")
        print(f"Configured IDs: {len(all_ids)}")
        print(f"Processed in this run: {start_index}")
        print(f"Successful in this run: {successful}")
        print(f"Failed in this run: {failed}")
        print(f"Receipts saved to: {os.path.abspath(OUTPUT_DIR)}")
        print(f"Completed IDs file: {os.path.abspath(COMPLETED_IDS_FILE)}")
        print(f"Checkpoint file: {os.path.abspath(CHECKPOINT_FILE)}")

    except Exception as e:
        # Ensure we persist a checkpoint on unexpected error
        print(f"Unhandled exception occurred: {e}")
        summary = {
            "timestamp": time.time(),
            "processed_so_far": start_index,
            "successful_so_far": successful,
            "failed_so_far": failed,
            "exception": str(e)
        }
        persist_checkpoint(CHECKPOINT_FILE, summary)
        print("Saved checkpoint after exception. Exiting with error.")
        raise  # re-raise for visibility if you want the stacktrace


if __name__ == "__main__":
    main()

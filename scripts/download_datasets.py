"""Dataset Download Script for PhysioNet MIT-BIH Arrhythmia Database.

Source: PhysioNet MIT-BIH Arrhythmia Database (mitdb) v1.0.0
Citation: Goldberger, A., et al. PhysioBank, PhysioToolkit, and PhysioNet:
          Components of a new research resource for complex physiologic signals.
          Circulation 101(23):e215-e220 (2000).
URL: https://physionet.org/content/mitdb/1.0.0/

Disclaimer:
    This dataset contains historical de-identified research records.
    The records do NOT represent the current user or operator of this system.
    Research prototype — not intended for medical diagnosis.

Usage:
    python scripts/download_datasets.py [--records 100 101 106 119] [--target-dir data/ecg]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import requests
import wfdb

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = ROOT_DIR / "data" / "ecg"

# Representative MIT-BIH records covering normal sinus rhythm and common arrhythmias:
# 100: Normal Sinus Rhythm (NSR) baseline with normal QRS
# 101: NSR with minor conduction variations
# 106: Ventricular Trigeminy and Premature Ventricular Contractions (PVC / V)
# 119: Frequent Ventricular Ectopy (PVC couplets and bigeminy)
DEFAULT_RECORDS = ["100", "101", "106", "119"]

# Required extensions for each PhysioNet record
EXTENSIONS = [".hea", ".atr", ".dat"]
PHYSIONET_BASE_URL = "https://physionet.org/files/mitdb/1.0.0/"


def ensure_directory(target_dir: Path) -> None:
    """Ensure target directory exists."""
    target_dir.mkdir(parents=True, exist_ok=True)


def verify_record_files(record_id: str, target_dir: Path) -> bool:
    """Check if all required record files (.hea, .dat, .atr) exist and are non-empty."""
    for ext in EXTENSIONS:
        f_path = target_dir / f"{record_id}{ext}"
        if not f_path.exists() or f_path.stat().st_size == 0:
            return False
    return True


def download_file_stream(url: str, dest_path: Path, timeout: int = 45) -> bool:
    """Download a file with streaming and chunked disk writes."""
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        headers = {"User-Agent": "BioQuest-CardiacMonitoring/1.0"}
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                print(f"      [HTTP {r.status_code} from {url}]", flush=True)
                return False
            with open(temp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        if temp_path.exists() and temp_path.stat().st_size > 0:
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            return True
        return False
    except Exception as e:
        print(f"      [Download error from {url}: {e}]", flush=True)
        if temp_path.exists():
            temp_path.unlink()
        return False


def download_record(record_id: str, target_dir: Path) -> Tuple[bool, str]:
    """Download a single record using streaming PhysioNet download with WFDB fallback."""
    if verify_record_files(record_id, target_dir):
        try:
            rec = wfdb.rdrecord(str(target_dir / record_id))
            ann = wfdb.rdann(str(target_dir / record_id), "atr")
            return True, f"Verified OK ({rec.sig_name}, {rec.fs} Hz, {len(ann.sample)} annotations)"
        except Exception as e:
            print(f"      [Existing files for {record_id} invalid, re-downloading: {e}]", flush=True)

    print(f"  --> Downloading Record {record_id} from PhysioNet MIT-BIH...", flush=True)
    success_all = True

    # 1. Download .hea, .atr, and .dat via streaming HTTP
    for ext in EXTENSIONS:
        filename = f"{record_id}{ext}"
        dest_path = target_dir / filename
        if dest_path.exists() and dest_path.stat().st_size > 0:
            continue

        file_url = f"{PHYSIONET_BASE_URL}{filename}"
        print(f"      Downloading {filename}...", flush=True)
        ok = download_file_stream(file_url, dest_path)
        if not ok:
            success_all = False
            break

    # 2. Fallback to wfdb.dl_database if needed
    if not success_all:
        print(f"      [Attempting WFDB download fallback for {record_id}...]", flush=True)
        try:
            wfdb.dl_database("mitdb", dl_dir=str(target_dir), records=[record_id], annotators=["atr"])
        except Exception as wfdb_err:
            return False, f"Download failed: {wfdb_err}"

    # 3. Verify download via WFDB parser
    if verify_record_files(record_id, target_dir):
        try:
            rec = wfdb.rdrecord(str(target_dir / record_id))
            ann = wfdb.rdann(str(target_dir / record_id), "atr")
            return True, f"Verified OK ({rec.sig_name}, {rec.fs} Hz, {len(ann.sample)} annotations)"
        except Exception as v_err:
            return False, f"Verification failed after download: {v_err}"

    return False, "Missing required files after download"


def download_all_records(records: List[str], target_dir: Path) -> Dict[str, Tuple[bool, str]]:
    """Download and verify all requested MIT-BIH records."""
    ensure_directory(target_dir)
    results = {}

    print("=" * 70, flush=True)
    print("  PHYSIONET MIT-BIH ARRHYTHMIA DATABASE DOWNLOADER", flush=True)
    print(f"  Target Directory: {target_dir}", flush=True)
    print(f"  Selected Records: {', '.join(records)}", flush=True)
    print("=" * 70, flush=True)

    for rec_id in records:
        ok, msg = download_record(rec_id, target_dir)
        results[rec_id] = (ok, msg)
        status_tag = "SUCCESS" if ok else "FAILED"
        print(f"  [{status_tag}] Record {rec_id:4s} : {msg}", flush=True)

    print("-" * 70, flush=True)
    succeeded = sum(1 for ok, _ in results.values() if ok)
    print(f"  Summary: {succeeded}/{len(records)} records ready in {target_dir}", flush=True)
    print("=" * 70, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Download PhysioNet MIT-BIH ECG records.")
    parser.add_argument(
        "--records",
        nargs="+",
        default=DEFAULT_RECORDS,
        help="Space-separated list of record IDs to download (e.g. 100 101 106 119)"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Target folder to save downloaded ECG records"
    )
    args = parser.parse_args()

    results = download_all_records(args.records, args.target_dir)
    all_ok = all(ok for ok, _ in results.values())
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

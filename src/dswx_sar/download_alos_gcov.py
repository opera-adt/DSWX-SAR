#!/usr/bin/env python3
"""
download_gcov_hls.py
----------------------------------
Command‑line tool for
  1. **Setting up Earthdata authentication files** (`.netrc`, `.urs_cookies`, `.dodsrc`, optional `.edl_token`).
  2. Querying (and optionally downloading) **DSWx‑HLS GCOV** granules from NASA CMR within a bounding box and time range.

Authentication priority
-----------------------
1. **Bearer token** – `--token` flag or `$EARTHDATA_TOKEN` env‑var.
2. **Basic auth via .netrc** – looks in `--netrc`, `$NETRC`, or *~/.netrc*.

If neither is found, you can generate the prerequisite files on the fly with
`--setup-auth` (interactive prompts).

Examples
--------
Set up auth files (only needs to be done once per machine):

```bash
./download_gcov_hls.py --setup-auth --generate-token
```

Query granules and download the files:

```bash
./download_gcov_hls.py \
  --bbox -170 -80 170 80 \
  --start 2000-10-01T00:00:00Z \
  --end   2025-10-13T00:00:00Z \
  --download-folder ./data \
  --download
```

If `--download` is omitted the script only lists the matching granule IDs and
file URLs.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import netrc
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
# import earthaccess
import requests

CMR_OPS = "https://cmr.earthdata.nasa.gov/search/granules"
CONCEPT_ID = "C2850262927-ASF"  # DSWx‑HLS GCOV collection
URS_HOST = "urs.earthdata.nasa.gov"
TOKEN_API = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"

# -----------------------------------------------------------------------------
# Authentication helpers
# -----------------------------------------------------------------------------

def _get_netrc_auth(host: str, netrc_path: Path | None = None) -> Tuple[str, str] | None:
    """Return (user, password) tuple for *host* from a .netrc file."""
    try:
        nrc = netrc.netrc(str(netrc_path) if netrc_path else None)
        auth = nrc.authenticators(host)
        return auth[:2] if auth else None
    except (FileNotFoundError, netrc.NetrcParseError):
        return None


def _init_session(token: str | None, netrc_path: Path | None) -> requests.Session:
    """Create a requests.Session with either Bearer‑token or basic‑auth."""
    sess = requests.Session()
    sess.headers.update({"Accept": "application/json"})

    if token:
        sess.headers["Authorization"] = f"Bearer {token}"
    else:
        auth = _get_netrc_auth(URS_HOST, netrc_path)
        if auth:
            sess.auth = auth
        else:
            print(
                "Warning: No token provided and no credentials found in ~/.netrc; "
                "downloads from protected hosts may fail.",
                file=sys.stderr,
            )
    return sess


# -----------------------------------------------------------------------------
# Earthdata prerequisite file creation
# -----------------------------------------------------------------------------

def _chmod_600(path: Path) -> None:
    if platform.system() != "Windows":
        try:
            path.chmod(0o600)
        except Exception as exc:
            print(f"Warning: failed to chmod 600 {path}: {exc}")


def setup_auth_files(netrc_path: Path, generate_token: bool = False) -> None:
    """Interactively create .netrc, .urs_cookies, .dodsrc (and optional token)."""

    home = Path.home()

    # prompt for credentials
    user = getpass.getpass(
        "Enter NASA Earthdata Login Username (or create at urs.earthdata.nasa.gov): "
    )
    pwd = getpass.getpass("Enter NASA Earthdata Login Password: ")

    # .netrc
    netrc_content = f"machine {URS_HOST} login {user} password {pwd}\n"
    netrc_path = netrc_path.expanduser()
    netrc_path.write_text(netrc_content)
    _chmod_600(netrc_path)
    print(f"Saved .netrc -> {netrc_path}")

    # .urs_cookies (blank)
    urs_cookies = home / ".urs_cookies"
    urs_cookies.touch(exist_ok=True)
    print(f"Saved .urs_cookies -> {urs_cookies}")

    # .dodsrc
    dodsrc = home / ".dodsrc"
    dodsrc.write_text(
        f"HTTP.COOKIEJAR={urs_cookies}\nHTTP.NETRC={netrc_path}\n"
    )
    print(f"Saved .dodsrc -> {dodsrc}")

    # Windows: copy .dodsrc to cwd for OPeNDAP tools
    if platform.system() == "Windows":
        shutil.copy2(dodsrc, Path.cwd())
        print(f"Copied .dodsrc -> {Path.cwd()}")

    # optional token generation
    if generate_token:
        creds = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        headers = {"Authorization": f"Basic {creds}"}
        resp = requests.post(TOKEN_API, headers=headers, timeout=30)
        if resp.ok:
            token = resp.json().get("access_token")
            if token:
                token_file = home / ".edl_token"
                token_file.write_text(token)
                print(f"Token saved -> {token_file}")
            else:
                print("Token response did not contain 'access_token'.")
        else:
            print(f"Token request failed ({resp.status_code}): {resp.text[:200]}...")


# -----------------------------------------------------------------------------
# CMR query & download
# -----------------------------------------------------------------------------

def download_gcov_hls(
    bbox: List[float],
    start_time: str,
    end_time: str,
    session: requests.Session,
    download_folder: Path,
    download: bool = False,
) -> Tuple[int, List[Path]]:
    """Query CMR for GCOV granules intersecting *bbox* and temporal window."""

    params = {
        "concept_id": CONCEPT_ID,
        "bounding_box": ",".join(map(str, bbox)),
        "temporal": f"{start_time},{end_time}",
        "page_size": 400,
    }

    r = session.get(CMR_OPS, params=params, timeout=60)
    if not r.ok:
        raise RuntimeError(
            f"CMR request failed with {r.status_code}: {r.text[:200]}..."
        )

    hits = r.headers.get("CMR-Hits", "0")
    print(f"CMR hits: {hits}")

    feed = r.json().get("feed", {})
    entries = feed.get("entry", [])
    if not entries:
        print("No granules found.")
        return 0, []

    download_folder.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    urls: List[str] = []

    # if download:
    #     earthaccess.login(persist=True)
    #     # Get requests https Session using Earthdata Login Info
    #     fs = earthaccess.get_requests_https_session()
    for entry in entries:
        gran_id = entry.get("producer_granule_id")
        print(f"Granule: {gran_id}")

        href = next(
            (
                l["href"]
                for l in entry.get("links", [])
                if l.get("href", "").lower().endswith(".h5")
            ),
            None,
        )
        if not href:
            print("  No .h5 link found; skipping.")
            continue

        print(f"  URL: {href}")
        urls.append(href)
        if not download:
            continue

        target = download_folder / Path(href).name
        if target.exists():
            print("  Already exists; skipping.")
            continue

        with session.get(href, stream=True, timeout=120) as resp:
            if not resp.ok:
                print(f"  Download failed ({resp.status_code}).")
                continue

            with open(target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        downloaded.append(target)
        print(f"  Saved -> {target}")
    if not download and urls:
        url_file = download_folder / "gcov_urls.txt"
        url_file.write_text(" ".join(urls) + " ")
        print(f"Saved {len(urls)} URLs to {url_file}")

    return len(entries), downloaded
    return len(entries), downloaded


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Set up Earthdata auth files and/or query+download DSWx‑HLS GCOV granules."
    )

    p.add_argument(
        "--setup-auth",
        action="store_true",
        help="Interactively create .netrc, .urs_cookies, .dodsrc (and optional token) then exit.",
    )
    p.add_argument(
        "--generate-token",
        action="store_true",
        help="With --setup-auth, also request an Earthdata Login token and save to ~/.edl_token.",
    )

    # Query / download options
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box coordinates in degrees.",
    )
    p.add_argument(
        "--start",
        help="Start datetime ISO‑8601 (e.g., 2020-01-01T00:00:00Z).",
    )
    p.add_argument(
        "--end",
        help="End datetime ISO‑8601 (e.g., 2020-12-31T23:59:59Z).",
    )
    p.add_argument(
        "--download-folder",
        type=Path,
        default=Path("."),
        help="Folder to store .h5 files (default: current directory).",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Actually download the .h5 files (otherwise just list).",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("EARTHDATA_TOKEN"),
        help="Earthdata bearer token. If omitted, falls back to ~/.netrc.",
    )
    p.add_argument(
        "--netrc",
        type=Path,
        default=os.environ.get("NETRC"),
        help="Path to .netrc (default: ~/.netrc).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------
    # Setup‑auth mode
    # ------------------------------------------------------
    if args.setup_auth:
        netrc_path = args.netrc or Path.home() / ".netrc"
        setup_auth_files(netrc_path, generate_token=args.generate_token)
        return

    # ------------------------------------------------------
    # Query / download mode
    # ------------------------------------------------------
    required = [args.bbox, args.start, args.end]
    if not all(required):
        print("Error: --bbox, --start and --end are required unless --setup-auth is used.")
        sys.exit(1)

    sess = _init_session(args.token, args.netrc)
    count, files = download_gcov_hls(
        bbox=args.bbox,
        start_time=args.start,
        end_time=args.end,
        session=sess,
        download_folder=args.download_folder,
        download=args.download,
    )

    print(f"\nMatched granules: {count}")
    if args.download:
        print(f"Downloaded files: {len(files)}")


if __name__ == "__main__":
    main()

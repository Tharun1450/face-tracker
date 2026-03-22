"""
download_video.py - Download video(s) from a Google Drive link.

Usage:
    python download_video.py
        → downloads all videos from the default folder URL

    python download_video.py "https://drive.google.com/drive/folders/FOLDER_ID"
        → downloads ALL video files from that folder into data/

    python download_video.py "https://drive.google.com/file/d/FILE_ID/view"
        → downloads a single specific video

Requires `gdown` (pip install gdown).
"""

import os
import sys

# Default Google Drive FOLDER URL containing your videos
DEFAULT_FOLDER_URL = "https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ"

OUTPUT_DIR = "data"

# File types to download from a folder
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def is_folder_url(url: str) -> bool:
    """Return True if the URL points to a folder rather than a file."""
    return "/drive/folders/" in url


def download_folder(url: str, dest: str) -> None:
    """Download all video files from a Google Drive folder."""
    import gdown

    print(f"\n📂  Downloading ALL videos from folder:")
    print(f"    {url}")
    print(f"    → saving to: {os.path.abspath(dest)}\n")

    os.makedirs(dest, exist_ok=True)

    try:
        # gdown.download_folder downloads everything in the folder
        gdown.download_folder(
            url=url,
            output=dest,
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:
        print(f"\n⚠  gdown folder download failed: {exc}")
        print("Trying fallback: listing files manually...\n")
        _fallback_folder_download(url, dest)


def _fallback_folder_download(url: str, dest: str) -> None:
    """Fallback: parse folder page and download each video individually."""
    import re
    import gdown

    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run:  pip install requests")
        sys.exit(1)

    # Extract folder ID from URL
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not match:
        print("ERROR: Could not extract folder ID from URL.")
        sys.exit(1)

    folder_id = match.group(1)

    # Use Google Drive API (no auth) to list files
    api_url = f"https://drive.google.com/drive/folders/{folder_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(api_url, headers=headers, timeout=20)

    # Find all file IDs in the page source
    file_ids = re.findall(r'"([a-zA-Z0-9_-]{33})"', resp.text)
    file_ids = list(dict.fromkeys(file_ids))  # deduplicate, preserve order

    if not file_ids:
        print("ERROR: Could not find any file IDs in the folder.")
        print("Make sure the folder is shared as 'Anyone with the link'.")
        sys.exit(1)

    print(f"Found {len(file_ids)} potential file(s). Attempting downloads...\n")
    downloaded = 0

    for fid in file_ids:
        dl_url = f"https://drive.google.com/uc?id={fid}"
        # Use a temp name, rename if it's a video
        tmp_path = os.path.join(dest, f"_tmp_{fid}.part")
        try:
            gdown.download(url=dl_url, output=tmp_path, quiet=True, fuzzy=True)
            if os.path.exists(tmp_path):
                size = os.path.getsize(tmp_path)
                if size < 50_000:
                    os.remove(tmp_path)
                    continue
                final_path = os.path.join(dest, f"video_{downloaded + 1:02d}.mp4")
                os.rename(tmp_path, final_path)
                print(f"  ✅  {os.path.basename(final_path)}  ({size / 1_048_576:.1f} MB)")
                downloaded += 1
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue

    if downloaded == 0:
        print("\nNo videos downloaded. Check folder permissions.")
    else:
        print(f"\n✅  Downloaded {downloaded} video(s) to: {os.path.abspath(dest)}")


def download_single(url: str, dest_dir: str) -> None:
    """Download a single video file."""
    import gdown

    os.makedirs(dest_dir, exist_ok=True)

    # Derive a filename from the URL if possible
    import re
    file_id_match = re.search(r"(?:id=|/d/)([a-zA-Z0-9_-]+)", url)
    if file_id_match:
        filename = f"video_{file_id_match.group(1)[:8]}.mp4"
    else:
        filename = "sample_video.mp4"

    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 50_000:
        print(f"✅  Already exists: {os.path.abspath(dest_path)}")
        return

    print(f"\n⬇   Downloading single video...")
    print(f"    URL : {url}")
    print(f"    Dest: {os.path.abspath(dest_path)}\n")

    try:
        gdown.download(url=url, output=dest_path, quiet=False, fuzzy=True)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("Try:  pip install --upgrade gdown")
        sys.exit(1)

    size_mb = os.path.getsize(dest_path) / 1_048_576
    print(f"\n✅  Saved: {filename}  ({size_mb:.1f} MB)")
    print(f"Update config.json → \"video_source\": \"data/{filename}\"")


def main():
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown not installed. Run:  pip install gdown")
        sys.exit(1)

    url = sys.argv[1].strip() if len(sys.argv) > 1 else DEFAULT_FOLDER_URL

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if is_folder_url(url):
        download_folder(url, OUTPUT_DIR)
        print("\nNow run:  python select_video.py   (to pick which video to use)")
    else:
        download_single(url, OUTPUT_DIR)
        print("\nNow run:  python main.py")


if __name__ == "__main__":
    main()

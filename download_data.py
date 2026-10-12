"""Download the QuickDraw bitmap files for every category in config.py.

Files are saved to data/<category>.npy. Categories that are already
downloaded are skipped, so the script is safe to re-run.
"""
import os
import urllib.parse
import urllib.request
from config import categories
os.makedirs("data", exist_ok=True)
URL= "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

# Some servers reject Python's default user agent, so identify as a browser
openerr = urllib.request.build_opener()
openerr.addheaders = [("User-Agent", "Mozilla/5.0")]
urllib.request.install_opener(openerr)


def show_progress(a, b, total):
    """Print the download percentage on a single, updating line."""
    downloaded = a * b
    if total > 0:
        percent = min(100, downloaded * 100 // total)
        print(f"\r  {percent}%", end="", flush=True)


for category in categories:
    filename = f"{category}.npy"
    filepath = os.path.join("data", filename)

    # Skip files that were already downloaded completely
    if os.path.exists(filepath):
        print(f"{filename} already exists")
        continue
    print(f"Downloading {filename}...")
    # Category names may contain spaces, which are not valid in URLs
    url_filename = urllib.parse.quote(filename)
    temp_path = filepath + ".part"
    try:
        urllib.request.urlretrieve(URL + url_filename, temp_path, show_progress)
        # Rename only after the download finished, so an interrupted
        # download can never leave a corrupt .npy file behind
        os.replace(temp_path, filepath)
        print(f"\n{filename} downloaded!")
    except OSError as error:
        print(f"\nfailed to download {filename} because of {error}")
        # Remove the partial file so the next run starts clean
        if os.path.exists(temp_path):
            os.remove(temp_path)

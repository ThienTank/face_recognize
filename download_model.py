import os
import requests
import zipfile
import time

url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
dest_dir = r"E:\mohinh\face_attendance\models"
os.makedirs(dest_dir, exist_ok=True)
dest_path = os.path.join(dest_dir, "buffalo_l.zip")

print(f"Downloading from {url}...")
print(f"Destination: {dest_path}")

max_retries = 30
retry_delay = 5

for attempt in range(1, max_retries + 1):
    file_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    headers = {}
    if file_size > 0:
        headers['Range'] = f'bytes={file_size}-'
        print(f"Attempt {attempt}/{max_retries}: Resuming from {file_size} bytes...")
    else:
        print(f"Attempt {attempt}/{max_retries}: Downloading from scratch...")

    try:
        r = requests.get(url, headers=headers, stream=True, timeout=20)
        
        status = r.status_code
        if status not in (200, 206):
            print(f"Received status code {status}, retrying...")
            time.sleep(retry_delay)
            continue
            
        mode = 'ab' if status == 206 else 'wb'
        if status == 206:
            total_size = int(r.headers.get('content-range', '').split('/')[-1])
        else:
            total_size = int(r.headers.get('content-length', 0))
            
        print(f"Total file size: {total_size} bytes")
        
        with open(dest_path, mode) as f:
            downloaded = file_size if status == 206 else 0
            for chunk in r.iter_content(chunk_size=1024 * 128): # 128KB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = (downloaded / total_size) * 100 if total_size else 0
                    print(f"\rProgress: {downloaded}/{total_size} bytes ({percent:.2f}%)", end="", flush=True)
        print()
        
        current_size = os.path.getsize(dest_path)
        if total_size and current_size >= total_size:
            print("Download completed successfully!")
            break
        else:
            print(f"Download ended prematurely ({current_size}/{total_size} bytes). Retrying...")
            
    except (requests.exceptions.RequestException, Exception) as e:
        print(f"\nError during attempt {attempt}: {e}")
        time.sleep(retry_delay)
else:
    print("Failed to download after maximum retries.")
    exit(1)

print("Extracting...")
extract_dir = os.path.join(dest_dir, "buffalo_l")
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(dest_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Successfully extracted to {extract_dir}")

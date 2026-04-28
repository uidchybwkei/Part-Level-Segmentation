import requests
import os
import threading
from tqdm import tqdm

URL = "https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth"
OUTPUT = "/root/autodl-tmp/Semantic-SAM/swinl_only_sam_many2many.pth"
NUM_THREADS = 16

def get_file_size(url):
    r = requests.head(url, allow_redirects=True)
    return int(r.headers.get('content-length', 0))

def download_chunk(url, start, end, part_file):
    headers = {'Range': f'bytes={start}-{end}'}
    r = requests.get(url, headers=headers, stream=True)
    with open(part_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)

def main():
    total = get_file_size(URL)
    print(f"File size: {total / 1024**2:.1f} MB")

    chunk_size = total // NUM_THREADS
    threads = []
    part_files = []

    for i in range(NUM_THREADS):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < NUM_THREADS - 1 else total - 1
        part_file = OUTPUT + f".part{i}"
        part_files.append(part_file)
        t = threading.Thread(target=download_chunk, args=(URL, start, end, part_file))
        threads.append(t)

    print(f"Starting {NUM_THREADS} parallel threads...")
    for t in threads:
        t.start()

    with tqdm(total=total, unit='B', unit_scale=True, desc="Total") as pbar:
        done = [False] * NUM_THREADS
        while not all(done):
            total_downloaded = sum(os.path.getsize(p) for p in part_files if os.path.exists(p))
            pbar.n = total_downloaded
            pbar.refresh()
            import time; time.sleep(1)
            done = [not t.is_alive() for t in threads]

    for t in threads:
        t.join()

    print("Merging parts...")
    with open(OUTPUT, 'wb') as out:
        for part_file in part_files:
            with open(part_file, 'rb') as f:
                out.write(f.read())
            os.remove(part_file)

    print(f"Done! Saved to {OUTPUT}")

if __name__ == '__main__':
    main()

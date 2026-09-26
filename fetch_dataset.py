"""Resolve the existing repository's Git LFS dataset and verify its checksum."""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import os

ROOT = Path(__file__).resolve().parent
URL = 'https://media.githubusercontent.com/media/GubbalaSaiJithin/Movie-Review-Sentiment-Analyzer/7741c95f105d6bb4b4cdc065e1e41e9a70b40957/IMDB_reviews_dataset.csv'
EXPECTED_SHA256 = '064d754f25565ffdf3d2f4ebab67809b759ebf90eb52c0ac101ca034db1c045d'
EXPECTED_BYTES = 384061983

def download():
    destination = ROOT / 'data/IMDB_reviews_dataset.csv'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists() and destination.stat().st_size == EXPECTED_BYTES:
        with destination.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else checksum(source)
        if digest == EXPECTED_SHA256:
            print('Verified cached dataset:', destination)
            return destination
    temporary = destination.with_suffix('.download')
    digest = hashlib.sha256()
    total = 0
    with urlopen(Request(URL, headers={'User-Agent': 'portfolio-project-audit'}), timeout=90) as response, temporary.open('wb') as target:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk); digest.update(chunk); total += len(chunk)
    if total != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError(f'Dataset verification failed: {total} bytes; partial file retained for inspection.')
    os.replace(temporary, destination)
    print('Verified dataset download:', total, 'bytes')
    return destination

def checksum(source):
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()

if __name__ == '__main__':
    download()

import os
import urllib.request

BASE_URL = 'https://hf-mirror.com/bert-base-uncased/resolve/main'
SAVE_DIR = r'k:\CLIP\mmdet\pretrained\bert-base-uncased'

FILES = [
    'config.json',
    'vocab.txt',
    'tokenizer_config.json',
    'pytorch_model.bin',
]

os.makedirs(SAVE_DIR, exist_ok=True)

for filename in FILES:
    url = f'{BASE_URL}/{filename}'
    filepath = os.path.join(SAVE_DIR, filename)
    print(f'[INFO] Downloading {filename} ...')
    urllib.request.urlretrieve(url, filepath)
    size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f'[OK] {filename}  ({size_mb:.1f} MB)')

print(f'\n[DONE] All files saved to {SAVE_DIR}')

print('\nContents:')
for f in os.listdir(SAVE_DIR):
    size_mb = os.path.getsize(os.path.join(SAVE_DIR, f)) / 1024 / 1024
    print(f'  {f}  ({size_mb:.1f} MB)')
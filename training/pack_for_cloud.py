"""Build the folder you upload to Kaggle as a private dataset: the trainer + train.jsonl split into parts under 8 MB + dev.jsonl (+ bench.jsonl if present).

A browser upload takes at most about 10 MB per file, so train.jsonl is split into train1.jsonl, train2.jsonl, ... (the trainer reads them all).
    python training/pack_for_cloud.py <folder with train.jsonl and dev.jsonl> <output folder> [--trainer train_multi.py]
The output folder is emptied first. Look inside before you upload: it must contain only data you are allowed to send to a cloud service.
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_BYTES = 8 * 1024 * 1024


def split_lines(lines, max_bytes=MAX_BYTES):
    """Group lines into parts whose UTF-8 size stays under max_bytes (a single longer line gets its own part)."""
    parts, size = [], 0
    for line in lines:
        b = len(line.encode('utf-8'))
        if not parts or size + b > max_bytes:
            parts.append([])
            size = 0
        parts[-1].append(line)
        size += b
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('out')
    ap.add_argument('--trainer', default='train_multi.py')
    a = ap.parse_args()
    if os.path.isdir(a.out):
        for f in os.listdir(a.out):
            os.remove(os.path.join(a.out, f))
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(os.path.join(HERE, a.trainer), a.out)
    shutil.copy(os.path.join(a.data, 'dev.jsonl'), a.out)
    if os.path.exists(os.path.join(a.data, 'bench.jsonl')):   # optional held-out bench, measured by the trainer
        shutil.copy(os.path.join(a.data, 'bench.jsonl'), a.out)
    lines = open(os.path.join(a.data, 'train.jsonl'), encoding='utf-8').readlines()
    for k, part in enumerate(split_lines(lines), 1):
        with open(os.path.join(a.out, f'train{k}.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(part)
    for n in sorted(os.listdir(a.out)):
        print(n, os.path.getsize(os.path.join(a.out, n)))


if __name__ == '__main__':
    main()

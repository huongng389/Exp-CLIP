"""Unlabelled AffectNet training subset for EXP-010 (train the head on AffectNet faces instead of CAER-S).

Draws N random images (fixed seed) from AffectNet+ `human_annotated/train_set/images` and copies them
into a flat ImageFolder (`<out>/all/`), so no label is used. File names in AffectNet+ are plain indices
(0.jpg, ...) and repeat between train and validation, so overlap with the test sets is checked by
content hash (MD5), not by name.

    python tools/prepare_affectnet_subset.py --src <AffectNet+>/human_annotated/train_set/images \
        --test-dirs <static_test_sets>/AffectNet7/test <static_test_sets>/AffectNet8/test \
        --out <AffectNet_train49k> --n 48938 --zip <AffectNet_train49k.zip>
"""
import argparse
import hashlib
import os
import random
import shutil
import zipfile

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Directory of AffectNet training images.')
    parser.add_argument('--test-dirs', nargs='+', default=[], help='Test ImageFolders that must not overlap.')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--n', type=int, default=48938, help='Number of images (default: CAER-S train size).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--zip', type=str, default=None, help='Also write <out> as this zip (stored, no compression).')
    args = parser.parse_args()

    names = sorted(f for f in os.listdir(args.src) if f.lower().endswith(IMG_EXTS))
    subset = sorted(random.Random(args.seed).sample(names, args.n))
    print(f'{len(names)} images in {args.src}, sampled {len(subset)} (seed {args.seed})')

    test_hashes = {md5(os.path.join(d, f)) for t in args.test_dirs for d, _, fs in os.walk(t) for f in fs
                   if f.lower().endswith(IMG_EXTS)}
    print(f'{len(test_hashes)} distinct test images in {len(args.test_dirs)} dirs')

    out_dir = os.path.join(args.out, 'all')
    os.makedirs(out_dir, exist_ok=True)
    overlap = []
    for k, f in enumerate(subset):
        src = os.path.join(args.src, f)
        if md5(src) in test_hashes:
            overlap.append(f)
            continue
        shutil.copy2(src, os.path.join(out_dir, f))
        if k % 10000 == 0:
            print(f'{k}/{len(subset)}')
    print(f'copied {len(subset) - len(overlap)} images to {out_dir}; {len(overlap)} skipped (same content as a test image): {overlap[:10]}')
    with open(os.path.join(args.out, 'subset.txt'), 'w') as fh:
        fh.write('\n'.join(f for f in subset if f not in overlap))

    if args.zip:
        with zipfile.ZipFile(args.zip, 'w', zipfile.ZIP_STORED) as z:
            for d, _, fs in os.walk(args.out):
                for f in sorted(fs):
                    p = os.path.join(d, f)
                    z.write(p, os.path.relpath(p, os.path.dirname(os.path.abspath(args.out))))
        print(f'wrote {args.zip} ({os.path.getsize(args.zip) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()

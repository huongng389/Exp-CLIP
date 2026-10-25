"""RAF-DB compound-expression test set for EXP-011 (zero-shot on 11 unseen compound classes).

Reads the aligned 100x100 images straight from `compound/Image/aligned.zip` (same image type as our
RAFDB basic test set) and the labels from `compound/EmoLabel/list_patition_label.txt`, and writes an
ImageFolder per split: `<out>/<split>/<NN>_<Class_Name>/`. Class folders are zero-padded so that
ImageFolder's alphabetical order equals the RAF-DB label order (1..11 -> 0..10).

    python tools/prepare_rafdb_compound.py --src <RAF-DB>/compound --out <RAFDB_compound> \
        --zip <RAFDB_compound.zip>
"""
import argparse
import os
import zipfile
from collections import Counter

# RAF-DB compound label order (compound/README.txt, labels 1..11)
COMPOUND_CLASSES = ['Happily_Surprised', 'Happily_Disgusted', 'Sadly_Fearful', 'Sadly_Angry',
                    'Sadly_Surprised', 'Sadly_Disgusted', 'Fearfully_Angry', 'Fearfully_Surprised',
                    'Angrily_Surprised', 'Angrily_Disgusted', 'Disgustedly_Surprised']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='RAF-DB `compound` directory.')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--splits', nargs='+', default=['test', 'train'])
    parser.add_argument('--zip', type=str, default=None, help='Also write <out> as this zip (stored, no compression).')
    args = parser.parse_args()

    labels = {}
    with open(os.path.join(args.src, 'EmoLabel', 'list_patition_label.txt')) as fh:
        for line in fh:
            name, lab = line.split()
            labels[name.replace('.jpg', '_aligned.jpg')] = int(lab) - 1

    counts = Counter()
    with zipfile.ZipFile(os.path.join(args.src, 'Image', 'aligned.zip')) as zf:
        for member in zf.namelist():
            name = os.path.basename(member)
            split = name.split('_')[0]
            if name not in labels or split not in args.splits:
                continue
            lab = labels[name]
            cls_dir = os.path.join(args.out, split, f'{lab:02d}_{COMPOUND_CLASSES[lab]}')
            os.makedirs(cls_dir, exist_ok=True)
            with open(os.path.join(cls_dir, name), 'wb') as f:
                f.write(zf.read(member))
            counts[(split, lab)] += 1

    for split in args.splits:
        n = [counts[(split, k)] for k in range(len(COMPOUND_CLASSES))]
        print(f'{split}: {sum(n)} images, per class {n}')
    missing = len(labels) - sum(counts[(s, k)] for s in ['train', 'test'] for k in range(len(COMPOUND_CLASSES)))
    if set(args.splits) == {'train', 'test'} and missing:
        print(f'WARNING: {missing} labelled images not found in aligned.zip')

    if args.zip:
        with zipfile.ZipFile(args.zip, 'w', zipfile.ZIP_STORED) as zf:
            for d, _, fs in os.walk(args.out):
                for f in sorted(fs):
                    path = os.path.join(d, f)
                    zf.write(path, os.path.relpath(path, os.path.dirname(args.out.rstrip('/'))))
        print(f'wrote {args.zip}')


if __name__ == '__main__':
    main()

"""Crop faces from an ImageFolder-style dataset with RetinaFace (via the batch-face package).

Usage:
    pip install batch-face
    python tools/crop_faces.py --src /content/CAER-S/train --dst /content/CAER-S_Face/train

For each image, the largest detected face is cropped as a square box (optionally enlarged by
--margin) and saved under the same class/file name in --dst. Images without a detected face
are skipped. Already-cropped outputs are skipped, so the script can be re-run after a Colab
disconnect.
"""
import argparse
import os

import cv2
from tqdm import tqdm
from batch_face import RetinaFace

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.9, help='RetinaFace confidence threshold.')
    parser.add_argument('--margin', type=float, default=0.0,
                        help='Enlarge the square face box by this fraction (0.2 = 20%% larger).')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id, -1 for CPU.')
    args = parser.parse_args()

    detector = RetinaFace(gpu_id=args.gpu, network='resnet50')
    total, saved, no_face, unreadable = 0, 0, 0, 0

    for cls in sorted(os.listdir(args.src)):
        src_dir = os.path.join(args.src, cls)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(args.dst, cls)
        os.makedirs(dst_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS))

        for name in tqdm(files, desc=cls):
            total += 1
            out_path = os.path.join(dst_dir, name)
            if os.path.exists(out_path):
                saved += 1
                continue
            img = cv2.imread(os.path.join(src_dir, name))
            if img is None:
                unreadable += 1
                continue
            faces = detector(img, threshold=args.threshold, cv=True)
            if not faces:
                no_face += 1
                continue

            # keep the largest face, as a square box around its centre
            box, _, _ = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = max(x2 - x1, y2 - y1) * (1 + args.margin) / 2
            h, w = img.shape[:2]
            x1, y1 = max(int(cx - half), 0), max(int(cy - half), 0)
            x2, y2 = min(int(cx + half), w), min(int(cy + half), h)
            cv2.imwrite(out_path, img[y1:y2, x1:x2])
            saved += 1

    print(f'total={total} saved={saved} no_face={no_face} unreadable={unreadable}')


if __name__ == '__main__':
    main()

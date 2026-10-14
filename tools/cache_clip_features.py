"""Cache frozen CLIP image features for train.py --cache-clip (EXP-003).

train.py feeds images (Resize 224 + ToTensor, no augmentation) through a frozen CLIP, so each
image's feature is the same in every epoch. Saving it once lets train.py skip CLIP and BLIP-2
(targets come from tools/precompute_au_weights.py stage `target`, pool `all`).

    python tools/cache_clip_features.py --train-data <ImageFolder root> --out <clip_feats.npz>

Output: npz with `paths` (relative to --train-data, as in the target chunks) and `feats`
(fp16 = what CLIP's encode_image returns on GPU; train.py casts it to float before the head).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from models.clip import clip  # noqa: E402

CLIP_NAMES = {'CLIP_B32': 'ViT-B/32', 'CLIP_B16': 'ViT-B/16', 'CLIP_L14': 'ViT-L/14'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--load-model', type=str, default='CLIP_L14', choices=list(CLIP_NAMES))
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load(CLIP_NAMES[args.load_model], device)
    model.eval()

    # same transform as train.py
    dataset = datasets.ImageFolder(args.train_data, transforms.Compose([transforms.Resize((224, 224)),
                                                                        transforms.ToTensor()]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
    feats = []
    with torch.no_grad():
        for k, (images, _) in enumerate(loader):
            feats.append(model.encode_image(images.to(device)).half().cpu())
            if k % 20 == 0:
                print(f'{min((k + 1) * args.batch_size, len(dataset))}/{len(dataset)}')
    paths = [os.path.relpath(p, args.train_data) for p, _ in dataset.samples]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, paths=np.array(paths), feats=torch.cat(feats).numpy())
    print(f'saved {len(paths)} features {tuple(feats[0].shape[1:])} to {args.out}')


if __name__ == '__main__':
    main()

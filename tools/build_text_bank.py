"""Build and encode the text bank for train.py --text-bank (EXP-004 text-anchor loss).

The anchor loss keeps CLIP's image-text similarities to these sentences after the projection head,
so the head does not drift on the text side. Variant A describes facial actions only and avoids
the test class names and the test prompt templates (models/Text.py), so no label information
leaks into training. Variant B adds emotion words (ablation only: overlaps the test classes).

    python tools/build_text_bank.py --variant A --out <text_bank_A.npz>

Output: npz with `texts` and `feats` (CLIP encode_text, fp16 as on GPU; train.py casts to float).
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from models.clip import clip  # noqa: E402

CLIP_NAMES = {'CLIP_B32': 'ViT-B/32', 'CLIP_B16': 'ViT-B/16', 'CLIP_L14': 'ViT-L/14'}

ACTIONS = [
    'raised inner eyebrows', 'raised outer eyebrows', 'raised eyebrows', 'lowered eyebrows',
    'eyebrows drawn together', 'furrowed brows', 'a wrinkled forehead', 'a smooth forehead',
    'raised upper eyelids', 'wide open eyes', 'tightened eyelids', 'narrowed eyes', 'squinting eyes',
    'closed eyes', 'half-closed eyes', 'eyes looking down', 'eyes looking to the side', 'raised cheeks',
    'a wrinkled nose', 'flared nostrils', 'a raised upper lip', 'lip corners pulled up',
    'lip corners pulled down', 'a raised chin', 'dimpled lip corners', 'lips pressed together',
    'tightened lips', 'pursed lips', 'puckered lips', 'stretched lips', 'parted lips', 'an open mouth',
    'a dropped jaw', 'a wide open mouth', 'clenched teeth', 'visible teeth', 'a relaxed mouth',
    'a closed mouth', 'a lowered lower lip', 'a tilted head', 'a head turned to the side', 'a lowered head',
]
ACTION_TEMPLATES = [
    'a face with {}.', 'a person with {}.', 'a close-up of someone with {}.', 'a portrait of a person with {}.',
    'this person has {}.', 'a man with {}.', 'a woman with {}.', 'a child with {}.',
]
# variant B only
EMOTIONS = [
    'happy', 'sad', 'angry', 'surprised', 'scared', 'disgusted', 'neutral', 'contemptuous', 'calm',
    'joyful', 'upset', 'furious', 'shocked', 'anxious', 'bored', 'confused', 'excited', 'tired',
    'proud', 'ashamed', 'nervous', 'relaxed', 'annoyed', 'amused',
]
EMOTION_TEMPLATES = ['a {} face.', 'a person who looks {}.', 'someone feeling {}.', 'a {} man.', 'a {} woman.']


def build(variant):
    texts = [t.format(a) for t in ACTION_TEMPLATES for a in ACTIONS]
    if variant == 'B':
        texts += [t.format(e) for t in EMOTION_TEMPLATES for e in EMOTIONS]
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['A', 'B'], default='A')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--load-model', type=str, default='CLIP_L14', choices=list(CLIP_NAMES))
    args = parser.parse_args()

    texts = build(args.variant)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load(CLIP_NAMES[args.load_model], device)
    model.eval()
    feats = []
    with torch.no_grad():
        for k in range(0, len(texts), 256):
            tokens = clip.tokenize(texts[k:k + 256], context_length=77, truncate=True).to(device)
            feats.append(model.encode_text(tokens).half().cpu())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, texts=np.array(texts), feats=torch.cat(feats).numpy())
    print(f'saved {len(texts)} sentences (variant {args.variant}) to {args.out}, e.g. {texts[:3]}')


if __name__ == '__main__':
    main()

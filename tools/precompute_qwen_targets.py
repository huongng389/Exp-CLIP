"""EXP-012: training targets from Qwen2.5-VL instead of BLIP-2 (FlanT5-XXL).

The original target is the T5 *encoder* state for (32 Q-Former image tokens + instruction),
mean-pooled and L2-normalised. Qwen2.5-VL is decoder-only, so we run one forward pass (no
generation) on the chat prompt (image + instruction) and pool the LLM hidden states:

    all     final-norm output, mean over the user turn (<|vision_start|> .. end of prompt)
    img     final-norm output, mean over the image tokens only
    last    final-norm output of the last prompt token (where the answer would start)
    mid_all hidden state of layer --mid-layer (default 2/3 depth), mean as in `all`

Output has the same layout as `precompute_au_weights.py target`, so train.py --cache-targets reads
it directly (pick the pool with --cache-pool):

    python tools/precompute_qwen_targets.py --train-data /content/CAER-S_Face/train --out-dir OUT

writes OUT/targets/chunk_*.npz with `paths` (relative to --train-data) and one fp16 (N, D) array
per pool. Resumable: images already in a chunk are skipped. Needs a recent transformers
(Qwen2_5_VLForConditionalGeneration), not the lavis environment.
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_INSTRUCTION = ('Please play the role of a facial action describer. '
                       'Objectively describe the detailed facial actions of the person in the image.')
POOLS = ('all', 'img', 'last', 'mid_all')


def list_images(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(paths)


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--instruction', type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument('--image-size', type=int, default=224,
                        help='Images are resized to this square size (multiple of 28) -> (size/28)^2 visual tokens.')
    parser.add_argument('--mid-layer', type=int, default=None, help='Layer for pool mid_all (default: 2/3 depth).')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='Encode only a fixed random subset of N images.')
    args = parser.parse_args()
    assert args.image_size % 28 == 0, '--image-size must be a multiple of 28'

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    tdir = os.path.join(args.out_dir, 'targets')
    os.makedirs(tdir, exist_ok=True)
    chunks = sorted(glob.glob(os.path.join(tdir, 'chunk_*.npz')))
    done = {p for c in chunks for p in np.load(c)['paths']}
    paths = list_images(args.train_data)
    random.Random(0).shuffle(paths)
    if args.limit:
        paths = paths[:args.limit]
    paths = [p for p in paths if p not in done]
    print(f'{len(done)} done, {len(paths)} to encode')
    if not paths:
        return

    pixels = args.image_size * args.image_size
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=pixels, max_pixels=pixels)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda').eval()
    n_layers = model.config.get_text_config().num_hidden_layers
    mid = args.mid_layer or round(n_layers * 2 / 3)
    image_id = model.config.image_token_id
    vision_start_id = model.config.vision_start_token_id

    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': args.instruction}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print('prompt:', repr(prompt), '| layers', n_layers, '| mid layer', mid)

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            im = Image.open(os.path.join(args.train_data, paths[i])).convert('RGB')
            return im.resize((args.image_size, args.image_size), Image.BICUBIC), paths[i]

    loader = torch.utils.data.DataLoader(Images(), batch_size=args.batch_size, num_workers=args.workers,
                                         collate_fn=lambda b: tuple(zip(*b)))
    n = 0
    for k, (images, batch_paths) in enumerate(loader, start=len(chunks)):
        inputs = processor(text=[prompt] * len(images), images=list(images), return_tensors='pt',
                           padding=True).to('cuda')
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
        ids, att = inputs['input_ids'], inputs['attention_mask'].bool()
        # user turn = from <|vision_start|> to the end (left padding, so the end is the last column)
        user = (ids == vision_start_id).long().cumsum(1).bool() & att
        img = ids == image_id
        h_last = out.hidden_states[-1].float()  # after the final RMSNorm
        h_mid = out.hidden_states[mid].float()
        pooled = {'all': masked_mean(h_last, user), 'img': masked_mean(h_last, img),
                  'last': h_last[:, -1], 'mid_all': masked_mean(h_mid, user)}
        np.savez(os.path.join(tdir, f'chunk_{k:05d}.npz'), paths=np.array(batch_paths),
                 **{p: F.normalize(v, dim=-1).half().cpu().numpy() for p, v in pooled.items()})
        n += len(batch_paths)
        if k == len(chunks):
            print(f'tokens/prompt {ids.shape[1]} | image tokens {int(img[0].sum())} | user tokens {int(user[0].sum())} '
                  f'| dim {h_last.shape[-1]}')
        print(f'{n}/{len(paths)}')


if __name__ == '__main__':
    main()

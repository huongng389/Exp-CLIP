"""EXP-001: per-image confidence weights from LLM captions vs. a real AU detector.

The training target of Exp-CLIP is the T5 encoder state of BLIP-2 (FlanT5-XXL) for
(image, instruction). The patched `generate` in models/BLIP2_T5.py never runs the decoder,
so here we load an *unpatched* BLIP-2 and decode the caption that this same encoder state
produces. The caption's facial actions are then checked against an AU detector (py-feat);
images whose caption claims actions the detector does not see get a low weight.

Three resumable stages (run in order; each writes to --out-dir):

    # 1) captions.jsonl  (needs lavis + GPU, same model/instruction as training)
    python tools/precompute_au_weights.py caption --train-data /content/CAER-S_Face/train \
        --out-dir /content/drive/MyDrive/exp001 --instruction "Please play the role of ..."
    # 2) aus.csv  (pip install py-feat)
    python tools/precompute_au_weights.py au --train-data /content/CAER-S_Face/train \
        --out-dir /content/drive/MyDrive/exp001
    # 3) au_weights.json + au_weights_debug.csv
    python tools/precompute_au_weights.py weight --out-dir /content/drive/MyDrive/exp001

Use --limit N on stages 1-2 to try a few images first.
Keys in all outputs are image paths relative to --train-data (as in ImageFolder).
"""
import argparse
import csv
import json
import os
import re

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

# Concept groups: a caption mentions a group via keywords; the detector "sees" the group if
# any of its AUs is active. Matching on groups (not single AUs) because captions are coarse
# ("raised eyebrows" = AU1 and/or AU2).
AU_GROUPS = {
    'brow_raise':     (['AU01', 'AU02'], [
        r'\b(rais|lift|arch)\w*\s+(\w+\s+){0,2}(eye)?brows?\b',
        r'\b(eye)?brows?\s+(are\s+|is\s+)?(slightly\s+)?(rais|lift|arch)\w*']),
    'brow_lower':     (['AU04'], [
        r'\bfrown\w*', r'\bfurrow\w*', r'\bscowl\w*', r'\bknit\w*\s+(\w+\s+)?brows?',
        r'\bbrows?\s+(are\s+)?(drawn|pulled|knitted)\s+together', r'\blower\w*\s+(\w+\s+)?(eye)?brows?']),
    'upper_lid_raise': (['AU05'], [
        r'\bwide[\s-]*(open\s+)?eyes', r'\beyes\s+(are\s+)?(wide|widened)', r'\bwiden\w*\s+(\w+\s+)?eyes',
        r'\bstar(e|ing)\b']),
    'cheek_raise':    (['AU06'], [
        r'\bcheeks?\s+(are\s+|is\s+)?rais\w*', r'\brais\w*\s+(\w+\s+)?cheeks?', r"\bcrow'?s[\s-]feet"]),
    'lid_tight':      (['AU07'], [
        r'\bsquint\w*', r'\bnarrow\w*\s+(\w+\s+)?eyes', r'\beyes\s+(are\s+)?narrow\w*']),
    'nose_wrinkle':   (['AU09', 'AU10'], [
        r'\bwrinkl\w*\s+(\w+\s+)?nose', r'\bnose\s+(is\s+)?wrinkl\w*', r'\bsneer\w*',
        r'\bupper\s+lip\s+(is\s+)?rais\w*', r'\brais\w*\s+(\w+\s+)?upper\s+lip']),
    'lip_corner_pull': (['AU12'], [
        r'\bsmil\w*', r'\bgrin\w*', r'\blaugh\w*',
        r'\b(lip\s+)?corners?\s+(of\s+(the|his|her|their)\s+(mouth|lips)\s+)?(are\s+|is\s+)?(pulled|turned|raised|curved)\s+up']),
    'dimpler':        (['AU14'], [r'\bdimpl\w*', r'\bsmirk\w*']),
    'lip_corner_depress': (['AU15', 'AU17'], [
        r'\b(lip\s+)?corners?\s+(of\s+(the|his|her|their)\s+(mouth|lips)\s+)?(are\s+|is\s+)?(pulled|turned|drawn)\s+down',
        r'\bdownturned', r'\bpout\w*', r'\bchin\s+(is\s+)?(rais|push|wrinkl)\w*']),
    'lip_stretch':    (['AU20'], [r'\blips?\s+(are\s+|is\s+)?stretch\w*', r'\bgrimac\w*']),
    'lip_press':      (['AU23', 'AU24', 'AU28'], [
        r'\b(lips?|mouth)\s+(are\s+|is\s+)?(tightly\s+)?(pressed|tightened|pursed|compressed|clenched)',
        r'\b(pursed|pressed|tight\w*|compressed)\s+lips', r'\bpurs\w*\s+(\w+\s+)?lips']),
    'mouth_open':     (['AU25', 'AU26'], [
        r'\bmouth\s+(is\s+)?(slightly\s+|wide\s+|partially\s+)?open', r'\bopen\w*\s+(\w+\s+)?mouth',
        r'\bparted\s+lips', r'\blips\s+(are\s+)?(slightly\s+)?parted', r'\bjaw\s+(is\s+)?drop\w*',
        r'\bgap\w*', r'\byawn\w*']),
    'eyes_closed':    (['AU43'], [r'\beyes\s+(are\s+)?(closed|shut)', r'\bclos\w*\s+(\w+\s+)?eyes', r'\bblink\w*']),
}
NEGATION = re.compile(r"\b(not|no|without|never|n't)\b(\s+\w+){0,2}\s*$")


def list_images(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(paths)


def todo(root, done, limit):
    paths = [p for p in list_images(root) if p not in done]
    return paths[:limit] if limit else paths


# ---------------------------------------------------------------- stage 1: captions
def stage_caption(args):
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    from lavis.models import load_model_and_preprocess

    out_path = os.path.join(args.out_dir, 'captions.jsonl')
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)['path'] for line in f}
    paths = todo(args.train_data, done, args.limit)
    print(f'{len(done)} done, {len(paths)} to caption')

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    # unpatched model: its generate() runs the decoder (models/BLIP2_T5.get_blip2t5_model does not)
    model, _, _ = load_model_and_preprocess(name='blip2_t5', model_type='pretrain_flant5xxl',
                                            is_eval=True, device=device)
    # same preprocessing as train.py, so the caption is decoded from the same encoder state
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    with open(out_path, 'a') as f:
        for i in range(0, len(paths), args.batch_size):
            batch = paths[i:i + args.batch_size]
            images = torch.stack([tf(Image.open(os.path.join(args.train_data, p)).convert('RGB'))
                                  for p in batch]).to(device)
            captions = model.generate({'image': images, 'prompt': args.instruction},
                                      num_beams=args.num_beams, max_length=args.max_length)
            for p, c in zip(batch, captions):
                f.write(json.dumps({'path': p, 'caption': c}) + '\n')
            f.flush()
            print(f'{i + len(batch)}/{len(paths)}  e.g. {batch[0]}: {captions[0]}')


# ---------------------------------------------------------------- stage 2: AUs
def stage_au(args):
    from feat import Detector

    out_path = os.path.join(args.out_dir, 'aus.csv')
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {row['path'] for row in csv.DictReader(f)}
    paths = todo(args.train_data, done, args.limit)
    print(f'{len(done)} done, {len(paths)} to detect')

    detector = Detector(au_model='xgb', device='cuda' if args.gpu else 'cpu')
    au_cols = None
    write_header = not os.path.exists(out_path)
    with open(out_path, 'a', newline='') as f:
        writer = None
        for i in range(0, len(paths), args.batch_size):
            batch = paths[i:i + args.batch_size]
            full = [os.path.join(args.train_data, p) for p in batch]
            fex = detector.detect_image(full, batch_size=len(full), output_size=256)
            if au_cols is None:
                au_cols = list(fex.au_columns)
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['path', 'face_found'] + au_cols)
            for p, fp in zip(batch, full):
                rows = fex[fex['input'] == fp]
                rows = rows.dropna(subset=au_cols)
                if len(rows) == 0:
                    writer.writerow([p, 0] + [''] * len(au_cols))
                    continue
                # several faces → keep the largest
                area = rows['FaceRectWidth'] * rows['FaceRectHeight']
                row = rows.loc[area.idxmax()]
                writer.writerow([p, 1] + [f'{row[c]:.4f}' for c in au_cols])
            f.flush()
            print(f'{i + len(batch)}/{len(paths)}')


# ---------------------------------------------------------------- stage 3: weights
def text_groups(caption):
    text = caption.lower()
    found = set()
    for group, (_, patterns) in AU_GROUPS.items():
        for pat in patterns:
            for m in re.finditer(pat, text):
                if not NEGATION.search(text[:m.start()]):
                    found.add(group)
                    break
            if group in found:
                break
    return found


def detected_groups(aus, threshold):
    return {g for g, (au_list, _) in AU_GROUPS.items()
            if any(aus.get(a, 0.0) >= threshold for a in au_list)}


def stage_weight(args):
    with open(os.path.join(args.out_dir, 'captions.jsonl')) as f:
        captions = {d['path']: d['caption'] for d in map(json.loads, f)}
    aus = {}
    with open(os.path.join(args.out_dir, 'aus.csv')) as f:
        for row in csv.DictReader(f):
            if row['face_found'] == '1':
                aus[row['path']] = {k: float(v) for k, v in row.items() if k.startswith('AU')}

    weights, n_missing, scores = {}, 0, []
    with open(os.path.join(args.out_dir, 'au_weights_debug.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'weight', 'text_groups', 'detected_groups', 'caption'])
        for path, caption in sorted(captions.items()):
            T = text_groups(caption)
            if path not in aus:
                w, D = args.w_missing, None
                n_missing += 1
            else:
                D = detected_groups(aus[path], args.threshold)
                if args.score == 'precision':
                    s = len(T & D) / len(T) if T else 1.0
                else:  # jaccard
                    s = len(T & D) / len(T | D) if (T | D) else 1.0
                w = max(args.w_min, s)
                scores.append(s)
            weights[path] = round(w, 4)
            writer.writerow([path, f'{w:.4f}', ' '.join(sorted(T)),
                             '' if D is None else ' '.join(sorted(D)), caption])

    with open(os.path.join(args.out_dir, 'au_weights.json'), 'w') as f:
        json.dump(weights, f)
    ws = sorted(weights.values())
    print(f'{len(weights)} images, {n_missing} without detected face (w={args.w_missing})')
    if ws:
        print(f'weight mean {sum(ws) / len(ws):.3f}, median {ws[len(ws) // 2]:.3f}, '
              f'== w_min: {sum(w <= args.w_min for w in ws)}')
    if scores:
        print(f'score == 0 (caption fully contradicts detector): {sum(s == 0 for s in scores)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['caption', 'au', 'weight'])
    parser.add_argument('--train-data', type=str, help='ImageFolder root (same as train.py --train-data).')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Only process the first N new images (0 = all).')
    parser.add_argument('--batch-size', type=int, default=32)
    # caption
    parser.add_argument('--instruction', type=str,
                        default='Please play the role of a facial action describer. '
                                'Objectively describe the detailed facial actions of the person in the image.')
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--max-length', type=int, default=60)
    # au
    parser.add_argument('--gpu', type=int, default=1, help='1 = run py-feat on CUDA.')
    # weight
    parser.add_argument('--threshold', type=float, default=0.5, help='AU probability threshold.')
    parser.add_argument('--score', choices=['jaccard', 'precision'], default='jaccard',
                        help='jaccard penalises both hallucinated and missed actions; '
                             'precision penalises only hallucinated ones.')
    parser.add_argument('--w-min', type=float, default=0.1)
    parser.add_argument('--w-missing', type=float, default=1.0,
                        help='Weight for images where py-feat finds no face.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    {'caption': stage_caption, 'au': stage_au, 'weight': stage_weight}[args.stage](args)


if __name__ == '__main__':
    main()

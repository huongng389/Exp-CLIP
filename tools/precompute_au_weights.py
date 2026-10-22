"""EXP-001: per-image confidence weights from the LLM target vector vs. a real AU detector.

The training target of Exp-CLIP is the BLIP-2 (FlanT5-XXL) T5 *encoder* state for
(32 Q-Former image tokens + instruction), mean-pooled and L2-normalised; no text is ever
decoded. So we read the facial actions straight from that vector: reference sentences for
each AU group ("the eyebrows are lowered and drawn together", ...) are encoded with the same
T5 encoder, and the cosine between target and these prototypes gives the image's
"LLM AU profile". It is compared with an AU detector (py-feat); images whose profile
disagrees with the detector get a low weight.

Resumable stages, run in order (each writes to --out-dir):

    # 1) targets/*.npz : target vectors, exactly as in training (lavis + GPU)
    python tools/precompute_au_weights.py target --train-data /content/CAER-S_Face/train --out-dir OUT
    # 2) protos.npz : AU prototype vectors (lavis + GPU)
    python tools/precompute_au_weights.py proto --out-dir OUT
    # 3) aus.csv : py-feat AU probabilities (separate venv with py-feat)
    python tools/precompute_au_weights.py au --train-data /content/CAER-S_Face/train --out-dir OUT
    # 4) au_weights_<tag>.json + sanity tables (numpy only)
    python tools/precompute_au_weights.py weight --out-dir OUT --pool all --proto plain

Use --limit N on stages 1 and 3 to try a random subset (same subset in both stages).
Keys in all outputs are image paths relative to --train-data (as in ImageFolder).
"""
import argparse
import csv
import glob
import json
import os
import random
import sys

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INSTRUCTION = ('Please play the role of a facial action describer. '
                       'Objectively describe the detailed facial actions of the person in the image.')

# AU group -> (AUs that the detector reports for it, reference sentences)
AU_GROUPS = {
    'brow_raise': (['AU01', 'AU02'], [
        'The eyebrows are raised.',
        'The person raises their eyebrows.',
        'The inner and outer brows are lifted upward.',
        'The eyebrows are lifted high, wrinkling the forehead.',
        'The brows are arched upward.']),
    'brow_lower': (['AU04'], [
        'The eyebrows are lowered and drawn together.',
        'The person is frowning.',
        'The brows are furrowed.',
        'The eyebrows are knitted together, creating vertical lines between them.',
        'The person scowls with lowered brows.']),
    'upper_lid_raise': (['AU05'], [
        'The eyes are wide open.',
        'The upper eyelids are raised.',
        'The person stares with widened eyes.',
        'The eyes are opened wide, showing the whites above the iris.',
        'The person widens their eyes.']),
    'cheek_raise': (['AU06'], [
        'The cheeks are raised.',
        'The cheeks are pushed up, narrowing the eyes from below.',
        "Crow's feet wrinkles appear at the corners of the eyes.",
        'The person raises their cheeks.',
        'The skin below the eyes is pushed upward by the cheeks.']),
    'lid_tight': (['AU07'], [
        'The eyelids are tightened.',
        'The person squints.',
        'The eyes are narrowed.',
        'The lower eyelids are raised and tense.',
        'The person narrows their eyes.']),
    'nose_wrinkle': (['AU09', 'AU10'], [
        'The nose is wrinkled.',
        'The upper lip is raised.',
        'The person wrinkles their nose.',
        'The person sneers, raising the upper lip.',
        'The nose is scrunched up and the upper lip is lifted.']),
    'lip_corner_pull': (['AU12'], [
        'The person is smiling.',
        'The corners of the lips are pulled up.',
        'The person grins.',
        'The mouth corners are raised in a smile.',
        'The lips are stretched upward into a smile.']),
    'dimpler': (['AU14'], [
        'Dimples appear at the corners of the mouth.',
        'The person smirks.',
        'The lip corners are tightened and pulled inward.',
        'The person has a one-sided smirk.',
        'The corners of the mouth are dimpled.']),
    'lip_corner_depress': (['AU15', 'AU17'], [
        'The corners of the mouth are pulled down.',
        'The person pouts.',
        'The chin is raised and the lower lip is pushed up.',
        'The mouth is turned down at the corners.',
        'The lip corners are depressed.']),
    'lip_stretch': (['AU20'], [
        'The lips are stretched horizontally.',
        'The person grimaces.',
        'The mouth is pulled wide sideways.',
        'The lips are stretched tight toward the ears.',
        'The person stretches their lips.']),
    'lip_press': (['AU23', 'AU24', 'AU28'], [
        'The lips are pressed together.',
        'The lips are tightened.',
        'The person purses their lips.',
        'The lips are pressed tightly into a thin line.',
        'The mouth is closed with tense lips.']),
    'mouth_open': (['AU25', 'AU26'], [
        'The mouth is open.',
        'The lips are parted.',
        'The jaw drops.',
        'The mouth hangs open.',
        'The person opens their mouth.']),
    'eyes_closed': (['AU43'], [
        'The eyes are closed.',
        'The person shuts their eyes.',
        'The eyelids are closed.',
        'The person blinks.',
        'The eyes are shut tightly.']),
}
GROUPS = list(AU_GROUPS)


def list_images(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(paths)


def todo(root, done, limit):
    # fixed shuffle so --limit gives a sample spread over classes/clips (not the first N files
    # of Angry/), and the target and au stages pick the same images
    paths = list_images(root)
    random.Random(0).shuffle(paths)
    if limit:
        paths = paths[:limit]
    return [p for p in paths if p not in done]


def load_blip2(device):
    sys.path.insert(0, REPO_ROOT)
    from models.BLIP2_T5 import get_blip2t5_model, get_tokens
    return get_blip2t5_model(device), get_tokens


# ---------------------------------------------------------------- stage 1: targets
def stage_target(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    tdir = os.path.join(args.out_dir, 'targets')
    os.makedirs(tdir, exist_ok=True)
    chunks = sorted(glob.glob(os.path.join(tdir, 'chunk_*.npz')))
    done = {p for c in chunks for p in np.load(c)['paths']}
    paths = todo(args.train_data, done, args.limit)
    print(f'{len(done)} done, {len(paths)} to encode')
    if not paths:
        return

    # same preprocessing as train.py, so the vectors are exactly the training targets
    sys.path.insert(0, REPO_ROOT)
    from data_loader.image_transform import image_transform
    tf = image_transform(args.preprocess)

    class Images(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return tf(Image.open(os.path.join(args.train_data, paths[i])).convert('RGB')), paths[i]

    loader = torch.utils.data.DataLoader(Images(), batch_size=args.batch_size, num_workers=args.workers)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model, get_tokens = load_blip2(device)
    n_query = model.query_tokens.shape[1]

    n = 0
    for k, (images, batch_paths) in enumerate(loader, start=len(chunks)):
        tokens = get_tokens(args.instruction, model.t5_tokenizer, len(batch_paths), device)
        _, h = model.generate({'image': images.to(device), 'tokens': tokens})
        h = h.float()  # (B, n_query + instruction length, 4096)
        np.savez(os.path.join(tdir, f'chunk_{k:05d}.npz'), paths=np.array(batch_paths),
                 all=F.normalize(h.mean(1), dim=-1).half().cpu().numpy(),             # = training target
                 img=F.normalize(h[:, :n_query].mean(1), dim=-1).half().cpu().numpy())  # image tokens only
        n += len(batch_paths)
        print(f'{n}/{len(paths)}')


# ---------------------------------------------------------------- stage 2: prototypes
def stage_proto(args):
    import numpy as np
    import torch
    import torch.nn.functional as F

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model, _ = load_blip2(device)

    def encode(texts):
        tok = model.t5_tokenizer(texts, padding='longest', return_tensors='pt').to(device)
        with torch.no_grad(), model.maybe_autocast(dtype=torch.bfloat16):
            h = model.t5_model.encoder(input_ids=tok.input_ids, attention_mask=tok.attention_mask,
                                       return_dict=True).last_hidden_state
        m = tok.attention_mask.unsqueeze(-1).float()
        return F.normalize((h.float() * m).sum(1) / m.sum(1), dim=-1)

    out = {'groups': np.array(GROUPS)}
    # plain: the sentence alone; templ: sentence after the training instruction
    for name, fmt in [('plain', '{}'), ('templ', args.instruction + ' {}')]:
        vecs = [F.normalize(encode([fmt.format(s) for s in AU_GROUPS[g][1]]).mean(0), dim=-1) for g in GROUPS]
        out[name] = torch.stack(vecs).cpu().numpy()
        sim = out[name] @ out[name].T
        print(f'{name}: mean cosine between different groups {sim[~np.eye(len(GROUPS), dtype=bool)].mean():.3f}')
    np.savez(os.path.join(args.out_dir, 'protos.npz'), **out)
    print(f'saved {len(GROUPS)} prototypes')


# ---------------------------------------------------------------- stage 3: AUs
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


# ---------------------------------------------------------------- stage 4: weights
def stage_weight(args):
    import numpy as np

    def zscore(m):
        return (m - m.mean(0)) / (m.std(0) + 1e-8)

    def ranks(v):
        r = np.empty(len(v))
        r[np.argsort(v)] = np.arange(len(v))
        return r

    chunks = sorted(glob.glob(os.path.join(args.out_dir, 'targets', 'chunk_*.npz')))
    paths = [p for c in chunks for p in np.load(c)['paths']]
    X = np.concatenate([np.load(c)[args.pool] for c in chunks]).astype(np.float32)
    protos = np.load(os.path.join(args.out_dir, 'protos.npz'))
    assert list(protos['groups']) == GROUPS, 'protos.npz is from another AU group list, rerun stage proto'
    P = protos[args.proto].astype(np.float32)

    aus, no_face = {}, set()
    with open(os.path.join(args.out_dir, 'aus.csv')) as f:
        for row in csv.DictReader(f):
            if row['face_found'] == '1':
                aus[row['path']] = {k: float(v) for k, v in row.items() if k.startswith('AU')}
            else:
                no_face.add(row['path'])

    # LLM AU profile. Centering removes what every target shares (same instruction tokens,
    # "a face"); per-group z-score removes the fixed offset between text-only prototypes
    # and image+instruction targets.
    Xc = X - X.mean(0)
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
    Pc = P - P.mean(0)
    Pc /= np.linalg.norm(Pc, axis=1, keepdims=True)
    L = zscore(Xc @ Pc.T)                                        # (N, G)

    idx = [i for i, p in enumerate(paths) if p in aus]
    A = np.array([[max(aus[paths[i]].get(a, 0.0) for a in AU_GROUPS[g][0]) for g in GROUPS] for i in idx])
    La, Za = L[idx], zscore(A)
    # agreement per image: Pearson across the selected AU groups of the two z-scored profiles
    # (test03: brow/lid groups agree at chance for both LLM and py-feat, so they only add noise)
    sel = [GROUPS.index(g) for g in args.groups.split(',')] if args.groups else list(range(len(GROUPS)))
    a = La[:, sel] - La[:, sel].mean(1, keepdims=True)
    b = Za[:, sel] - Za[:, sel].mean(1, keepdims=True)
    s = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)
    u = ranks(s) / max(len(s) - 1, 1) if args.map == 'rank' else np.clip((s + 1) / 2, 0, 1)
    w = args.w_min + (1 - args.w_min) * u

    cls = lambda p: p.split('/')[0]
    # test03: clear smiles agree best, so smiling images get higher weights. Rescale each bin
    # to the global mean weight: 'smile' bins by detector smile probability (label-free),
    # 'class' by the CAER-S folder (uses labels -> analysis only, not for the zero-shot claim).
    if args.balance == 'smile':
        smile = A[:, GROUPS.index('lip_corner_pull')]
        bins = np.searchsorted(np.quantile(smile, [0.2, 0.4, 0.6, 0.8]), smile, side='right')
    elif args.balance == 'class':
        bins = np.array([cls(paths[i]) for i in idx])
    else:
        bins = np.zeros(len(idx), dtype=int)
    if args.drop > 0:
        # EXP-001 seed1: soft weights (mostly 0.45-0.85) changed nothing, so drop the images
        # that agree least (w=0) and keep the rest at w=1. Drop the same fraction in every bin.
        w = np.ones(len(idx))
        for bval in np.unique(bins):
            m = np.where(bins == bval)[0]
            w[m[np.argsort(s[m])[:int(round(args.drop * len(m)))]]] = 0.0
    elif args.balance != 'none':
        target = w.mean()
        for bval in np.unique(bins):
            m = bins == bval
            w[m] *= target / w[m].mean()
        w = np.clip(w, args.w_min, 1.0)

    tag = args.tag or (f'{args.pool}-{args.proto}' + ('-sel' if args.groups else '')
                       + ('' if args.balance == 'none' else f'-{args.balance}')
                       + (f'-drop{round(args.drop * 100)}' if args.drop > 0 else ''))
    weights = {paths[i]: round(float(wi), 4) for i, wi in zip(idx, w)}
    weights.update({p: args.w_missing for p in paths if p in no_face})
    with open(os.path.join(args.out_dir, f'au_weights_{tag}.json'), 'w') as f:
        json.dump(weights, f)

    top = lambda v: ' '.join(GROUPS[j] for j in np.argsort(-v)[:3])
    with open(os.path.join(args.out_dir, f'au_weights_debug_{tag}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'class', 'weight', 'agreement', 'llm_top3', 'detector_top3'])
        for k, i in enumerate(idx):
            writer.writerow([paths[i], cls(paths[i]), f'{w[k]:.4f}', f'{s[k]:.4f}', top(La[k]), top(Za[k])])

    # sanity 1: mean profile per CAER-S class (LLM and detector side by side)
    classes = sorted({cls(paths[i]) for i in idx})
    with open(os.path.join(args.out_dir, f'sanity_class_profile_{tag}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'class'] + GROUPS)
        for source, M in [('llm', La), ('detector', Za)]:
            for c in classes:
                rows = [k for k, i in enumerate(idx) if cls(paths[i]) == c]
                writer.writerow([source, c] + [f'{v:.3f}' for v in M[rows].mean(0)])

    # sanity 2: per-group Spearman between LLM score and detector probability, over images
    rho = [np.corrcoef(ranks(La[:, g]), ranks(A[:, g]))[0, 1] for g in range(len(GROUPS))]
    with open(os.path.join(args.out_dir, f'sanity_spearman_{tag}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'spearman'])
        writer.writerows([g, f'{r:.3f}'] for g, r in zip(GROUPS, rho))

    print(f'[{tag}] {len(paths)} targets | {len(idx)} with AUs | {len(no_face)} no face (w={args.w_missing}) | '
          f'{len(paths) - len(idx) - len(no_face)} not yet AU-detected (skipped)')
    print(f'agreement over {len(sel)} groups: mean {s.mean():.3f} std {s.std():.3f} | weight mean {w.mean():.3f} '
          f'std {w.std():.3f} (map {args.map}, balance {args.balance}, w_min {args.w_min})')
    if args.drop > 0:
        kept = s[w > 0]
        print(f'drop {args.drop:.0%}: {int((w == 0).sum())} images w=0 | agreement kept min {kept.min():.3f} '
              f'mean {kept.mean():.3f} | dropped per class: '
              + ', '.join(f'{c} {np.mean([w[k] == 0 for k, i in enumerate(idx) if cls(paths[i]) == c]):.0%}'
                          for c in sorted({cls(p) for p in paths})))
    print('spearman per group: ' + ', '.join(f'{g} {r:+.2f}' for g, r in zip(GROUPS, rho)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['target', 'proto', 'au', 'weight'])
    parser.add_argument('--train-data', type=str, help='ImageFolder root (same as train.py --train-data).')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Only the first N images of a fixed shuffle (0 = all).')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--instruction', type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument('--preprocess', choices=['raw', 'clip'], default='raw',
                        help='target: raw = original Resize + ToTensor; clip = + bicubic and CLIP mean/std (EXP-009).')
    # au
    parser.add_argument('--gpu', type=int, default=1, help='1 = run py-feat on CUDA.')
    # weight
    parser.add_argument('--pool', choices=['all', 'img'], default='all',
                        help='all = training target (all tokens); img = image tokens only.')
    parser.add_argument('--proto', choices=['plain', 'templ'], default='plain',
                        help='plain = AU sentence alone; templ = AU sentence after the instruction.')
    parser.add_argument('--map', choices=['linear', 'rank'], default='linear',
                        help='agreement -> weight: linear (s+1)/2, or rank percentile.')
    parser.add_argument('--groups', type=str, default='',
                        help='Comma-separated AU groups used for the agreement (default: all 13).')
    parser.add_argument('--balance', choices=['none', 'smile', 'class'], default='none',
                        help='Equalise mean weight across smile-probability quintiles (label-free) '
                             'or CAER-S classes (uses labels, analysis only).')
    parser.add_argument('--tag', type=str, default='', help='Output name suffix (default: built from options).')
    parser.add_argument('--w-min', type=float, default=0.1)
    parser.add_argument('--drop', type=float, default=0.0,
                        help='Hard filter: w=0 for this fraction of lowest-agreement images (per --balance bin), '
                             'w=1 for the rest. Overrides --map/--w-min.')
    parser.add_argument('--w-missing', type=float, default=1.0,
                        help='Weight for images where py-feat finds no face.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    {'target': stage_target, 'proto': stage_proto, 'au': stage_au, 'weight': stage_weight}[args.stage](args)


if __name__ == '__main__':
    main()

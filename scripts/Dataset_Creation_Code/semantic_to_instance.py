#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Instance Dataset Converter (from semantic masks)
===============================================

Works directly with Falak's dataset layout and the semantic-composed output.

Expected layout (configurable via flags):
  IN_ROOT/
    <images_dirname>/<split>/<...>/*.jpg|*.png   (e.g., input/train/<scene>/visible/*.jpg)
    <masks_dirname>/<split>/<...>/*.png          (e.g., labels/train/<scene>/visible/*_mask.png or *_multiK.png)

You can alias a split on the image side to a different split on the mask side
(e.g., images -> labels for the flat RGB case) with:
  --mask_split_aliases images:labels

Examples:
1) Convert the semantic-composed FLAT RGB split produced earlier
   /home/falak/Full_revise_dataset/input/images  ↔  /home/falak/Full_revise_dataset/labels/labels

python3 instance_builder_falak.py \
  --in_root  /home/falak/Full_revise_dataset \
  --out_root /home/falak/Full_revise_dataset_instance \
  --images_dirname input --masks_dirname labels \
  --mask_split_aliases images:labels \
  --splits images \
  --recursive \
  --copy_images \
  --min_area 20

2) Convert train/val/test modality folders (visible/infrared):
python3 instance_builder_falak.py \
  --in_root  /home/falak/Full_revise_dataset \
  --out_root /home/falak/Full_revise_dataset_instance \
  --images_dirname input --masks_dirname labels \
  --splits train,val,test \
  --recursive \
  --copy_images \
  --min_area 20

Notes:
- Finds masks robustly (supports *_mask, -mask, _label, etc.; or numeric token fallback).
- Saves instances as 16-bit PNG (ids 1..K).
- Mirrors the input folder structure under OUT_ROOT using the same images/masks dir names.
"""

import os, re, cv2, numpy as np, argparse, shutil
from glob import glob
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP"}
MASK_EXTS= {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

MASK_CAND_TEMPLATES = [
    "{base}.png",
    "{base}_mask.png", "{base}-mask.png",
    "{base}_label.png", "{base}_gt.png", "{base}_seg.png",
    "{base}.bmp", "{base}_mask.bmp", "{base}-mask.bmp",
    "{base}_label.bmp", "{base}.tif", "{base}_mask.tif", "{base}-mask.tif",
    "{base}_label.tif", "{base}.tiff", "{base}_mask.tiff", "{base}-mask.tiff",
    "{base}_label.tiff", "{base}.jpg", "{base}_mask.jpg", "{base}-mask.jpg",
    "{base}_label.jpg", "{base}.jpeg", "{base}_mask.jpeg", "{base}-mask.jpeg",
    "{base}_label.jpeg",
]

def is_img(p: str) -> bool:
    return os.path.splitext(p)[1] in IMG_EXTS

# --------- helpers copied from semantic composer for robust matching ----------

def list_images_recursive(folder: str) -> List[str]:
    out=[]
    for root,_,files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1] in IMG_EXTS:
                out.append(os.path.relpath(os.path.join(root,f), folder))
    return sorted(out)

def list_images_flat(folder: str) -> List[str]:
    if not os.path.isdir(folder): return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1] in IMG_EXTS
    )

def find_mask(mask_dir: str, img_rel: str) -> Optional[str]:
    rel_dir  = os.path.dirname(img_rel)
    img_base = os.path.splitext(os.path.basename(img_rel))[0]

    # 1) Exact template matches
    for t in MASK_CAND_TEMPLATES:
        cand = os.path.join(mask_dir, rel_dir, t.format(base=img_base))
        if os.path.isfile(cand):
            return cand

    # 2) Same basename with any mask extension
    for e in MASK_EXTS:
        cand = os.path.join(mask_dir, rel_dir, img_base + e)
        if os.path.isfile(cand):
            return cand

    # 3) Fallback: match by the largest numeric token (e.g., frame index)
    digits = re.findall(r"\d+", img_base)
    if digits:
        key = digits[-1]
        search_dir = os.path.join(mask_dir, rel_dir)
        if os.path.isdir(search_dir):
            candidates = []
            for e in MASK_EXTS:
                candidates.extend(glob(os.path.join(search_dir, f"*{key}*{e}")))
            if candidates:
                def score(p):
                    name = os.path.basename(p).lower()
                    s = 0
                    if "mask" in name:  s += 3
                    if "label" in name: s += 2
                    if "gt" in name:    s += 1
                    if img_base.lower() in name: s += 4
                    return -s
                candidates.sort(key=score)
                return candidates[0]
    return None

# ----------------------------------------------------------------------------

def walk_pairs(in_root: str, images_dirname: str, masks_dirname: str,
               split: str, recursive: bool, mask_split_aliases: Dict[str,str]):
    img_split_dir = os.path.join(in_root, images_dirname, split)
    mask_split_name = mask_split_aliases.get(split, split)
    msk_split_dir = os.path.join(in_root, masks_dirname, mask_split_name)

    if not (os.path.isdir(img_split_dir) and os.path.isdir(msk_split_dir)):
        return

    rel_paths = list_images_recursive(img_split_dir) if recursive else list_images_flat(img_split_dir)
    for rel in rel_paths:
        img_abs = os.path.join(img_split_dir, rel)
        msk_abs = find_mask(msk_split_dir, rel)
        base, _ = os.path.splitext(os.path.basename(rel))
        rel_dir = os.path.dirname(rel)
        yield split, rel_dir, base, img_abs, msk_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root",  required=True, help="Semantic dataset root (contains <images_dirname>/ and <masks_dirname>/)")
    ap.add_argument("--out_root", required=True, help="Output root for instance dataset")
    ap.add_argument("--splits", default="train,val,test", help="Comma-separated list, e.g., train,val,test,images")
    ap.add_argument("--images_dirname", default="input", help="Images top-level dir name under in_root")
    ap.add_argument("--masks_dirname",  default="labels", help="Masks  top-level dir name under in_root")
    ap.add_argument("--mask_split_aliases", default="", help="Comma-separated mapping like 'images:labels' if mask split differs")
    ap.add_argument("--recursive", action="store_true", help="Recurse into nested scene/modality folders")
    ap.add_argument("--min_area", type=int, default=20, help="Remove components with area < min_area")
    ap.add_argument("--copy_images", action="store_true", help="Copy images instead of creating symlinks")
    args = ap.parse_args()

    # Parse aliases
    mask_alias: Dict[str,str] = {}
    if args.mask_split_aliases:
        for kv in args.mask_split_aliases.split(","):
            if ":" in kv:
                k,v = kv.split(":",1)
                mask_alias[k.strip()] = v.strip()

    # Prepare output roots mirroring input top-level dirnames
    out_img_root = os.path.join(args.out_root, args.images_dirname)
    out_msk_root = os.path.join(args.out_root, args.masks_dirname)
    os.makedirs(out_img_root, exist_ok=True)
    os.makedirs(out_msk_root, exist_ok=True)

    total=0; ok=0; skipped=0

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    all_pairs = []
    for sp in splits:
        for tup in walk_pairs(args.in_root, args.images_dirname, args.masks_dirname, sp, args.recursive, mask_alias):
            all_pairs.append(tup)

    for split, rel_dir, base, ip, mp in tqdm(all_pairs, desc="[inst]"):
        total += 1
        if not mp or not os.path.isfile(mp):
            skipped += 1
            continue

        I = cv2.imread(ip, cv2.IMREAD_COLOR)
        M = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if I is None or M is None:
            skipped += 1
            continue
        if M.ndim == 3:
            M = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)
        if I.shape[:2] != M.shape[:2]:
            M = cv2.resize(M, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)

        # binary foreground
        fg = (M > 0).astype(np.uint8)
        if fg.sum() == 0:
            skipped += 1
            continue

        # connected components -> instance ids 1..N
        num, labels = cv2.connectedComponents(fg, connectivity=8)
        if args.min_area > 1 and num > 1:
            areas = np.bincount(labels.ravel())
            keep = np.zeros(num, dtype=bool)
            keep[0] = False
            for lab in range(1, num):
                keep[lab] = areas[lab] >= args.min_area
            inst = np.zeros_like(labels, dtype=np.uint16)
            new_id = 1
            for lab in range(1, num):
                if keep[lab]:
                    inst[labels == lab] = new_id
                    new_id += 1
        else:
            inst = labels.astype(np.uint16)

        # write image + instance mask under OUT_ROOT mirroring the relative path
        out_img_dir = os.path.join(out_img_root, split, rel_dir)
        out_msk_dir = os.path.join(out_msk_root, mask_alias.get(split, split), rel_dir)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_msk_dir, exist_ok=True)

        # copy/symlink image
        ext = os.path.splitext(ip)[1].lower()
        out_img_path = os.path.join(out_img_dir, base + ext)
        if not os.path.exists(out_img_path):
            try:
                if args.copy_images:
                    shutil.copy2(ip, out_img_path)
                else:
                    os.symlink(ip, out_img_path)
            except FileExistsError:
                pass
            except OSError:
                shutil.copy2(ip, out_img_path)

        out_msk_path = os.path.join(out_msk_dir, base + ".png")
        cv2.imwrite(out_msk_path, inst, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        ok += 1

    print(f"\n[DONE] Instance conversion")
    print(f"  total images seen: {total}")
    print(f"  written: {ok}")
    print(f"  skipped (missing/empty): {skipped}")
    print(f"  out_root: {args.out_root}")

if __name__ == "__main__":
    main()

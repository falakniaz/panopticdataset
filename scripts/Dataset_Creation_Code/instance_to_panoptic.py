import os, re, cv2, json, argparse
import numpy as np
from glob import glob
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def is_img(path: str) -> bool:
    return os.path.splitext(path)[1] in IMG_EXTS

def list_images_recursive(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1] in IMG_EXTS:
                out.append(os.path.relpath(os.path.join(root, f), folder))
    return sorted(out)

# ---- mask finding (for cases when instance masks mirror image names) ----

def find_mask(mask_root: str, img_rel: str) -> Optional[str]:
    rel_dir = os.path.dirname(img_rel)
    img_base = os.path.splitext(os.path.basename(img_rel))[0]

    for t in MASK_CAND_TEMPLATES:
        cand = os.path.join(mask_root, rel_dir, t.format(base=img_base))
        if os.path.isfile(cand):
            return cand
    for e in MASK_EXTS:
        cand = os.path.join(mask_root, rel_dir, img_base + e)
        if os.path.isfile(cand):
            return cand
    digits = re.findall(r"\d+", img_base)
    if digits:
        key = digits[-1]
        search_dir = os.path.join(mask_root, rel_dir)
        if os.path.isdir(search_dir):
            cands = []
            for e in MASK_EXTS:
                cands.extend(glob(os.path.join(search_dir, f"*{key}*{e}")))
            if cands:
                def score(p):
                    name = os.path.basename(p).lower()
                    s = 0
                    if "mask" in name: s += 3
                    if "label" in name: s += 2
                    if "gt" in name: s += 1
                    if img_base.lower() in name: s += 4
                    return -s
                cands.sort(key=score)
                return cands[0]
    return None

# ---- panoptic helpers ----

def rgb_encode_id(seg_id: np.ndarray) -> np.ndarray:
    seg_id = seg_id.astype(np.uint32)
    r = (seg_id & 255).astype(np.uint8)
    g = ((seg_id >> 8) & 255).astype(np.uint8)
    b = ((seg_id >> 16) & 255).astype(np.uint8)
    return np.dstack([r, g, b])


def bbox_from_mask(mask_bin: np.ndarray):
    ys, xs = np.where(mask_bin)
    if xs.size == 0:
        return [0, 0, 0, 0], 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0 + 1, y1 - y0 + 1
    area = int(mask_bin.sum())
    return [x0, y0, w, h], area


def connected_components_from_binary(bin_mask: np.ndarray) -> np.ndarray:
    M = (bin_mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(M.astype(np.uint8))
    return labels.astype(np.uint16)

# ---- scanning ----

def scan_split(image_root: str, images_dir: str,
               inst_root: str, inst_masks_dir: str,
               split: str, mask_split_aliases: Dict[str, str], recursive: bool) -> List[Tuple[str, str, str]]:
    img_split = os.path.join(image_root, images_dir, split)
    mask_split_name = mask_split_aliases.get(split, split)
    msk_split = os.path.join(inst_root, inst_masks_dir, mask_split_name)

    if not os.path.isdir(img_split) or not os.path.isdir(msk_split):
        return []

    rel_imgs = list_images_recursive(img_split) if recursive else [f for f in sorted(os.listdir(img_split)) if is_img(os.path.join(img_split, f))]

    items: List[Tuple[str, str, str]] = []
    for rel in rel_imgs:
        img_abs = os.path.join(img_split, rel)
        msk_abs = find_mask(msk_split, rel)
        if msk_abs is None:
            base, _ = os.path.splitext(os.path.basename(rel))
            cand = os.path.join(msk_split, os.path.dirname(rel), base + ".png")
            if os.path.isfile(cand):
                msk_abs = cand
        items.append((rel, img_abs, msk_abs if msk_abs else ""))
    return items

# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True, help="Semantic dataset root (images source)")
    ap.add_argument("--instance_root", required=True, help="Instance dataset root (instance masks source)")
    ap.add_argument("--out_root", required=True, help="Output root for panoptic dataset")
    ap.add_argument("--images_dirname", default="input", help="Top‑level images dir name under image_root")
    ap.add_argument("--instance_masks_dirname", default="labels", help="Top‑level masks dir name under instance_root")
    ap.add_argument("--mask_split_aliases", default="", help="Comma‑sep mapping like 'images:labels' if split names differ")
    ap.add_argument("--splits", default="train,val,test", help="Comma‑sep list, e.g., 'images' or 'train,val,test'")
    ap.add_argument("--recursive", action="store_true", help="Recurse into scene/modality dirs")
    ap.add_argument("--from_binary_components", action="store_true", help="If instance masks are binary, derive instances via CCs")
    ap.add_argument("--min_area", type=int, default=20)
    ap.add_argument("--id_multiplier", type=int, default=1000)
    ap.add_argument("--category_id", type=int, default=1)
    ap.add_argument("--category_name", default="uav_object")
    ap.add_argument("--add_background_stuff", action="store_true", help="Add a background stuff segment to cover uncovered pixels")
    ap.add_argument("--background_category_id", type=int, default=0)
    ap.add_argument("--background_category_name", default="background")
    ap.add_argument("--fail_if_empty", action="store_true")
    args = ap.parse_args()

    alias: Dict[str, str] = {}
    if args.mask_split_aliases:
        for kv in args.mask_split_aliases.split(','):
            if ':' in kv:
                k, v = kv.split(':', 1)
                alias[k.strip()] = v.strip()

    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    # Ensure output dirs
    ann_dir = os.path.join(args.out_root, "annotations")
    ensure_dir(ann_dir)
    panoptic_dirs: Dict[str, str] = {}
    for sp in splits:
        pdir = os.path.join(args.out_root, f"panoptic_{sp}")
        ensure_dir(pdir)
        panoptic_dirs[sp] = pdir

    # Scan
    per_split = {
        sp: scan_split(args.image_root, args.images_dirname,
                       args.instance_root, args.instance_masks_dirname,
                       sp, alias, args.recursive)
        for sp in splits
    }
    total_found = sum(len(per_split[sp]) for sp in splits)
    for sp in splits:
        print(f"[scan] {sp}: found {len(per_split[sp])} image entries under {args.images_dirname}/{sp}")
    if total_found == 0:
        msg = ("[error] No images found in any split. Check paths:\n"
               f"  images:   {os.path.join(args.image_root, args.images_dirname, '<split>')}\n"
               f"  instances:{os.path.join(args.instance_root, args.instance_masks_dirname, '<split or alias>')}")
        if args.fail_if_empty:
            raise SystemExit(msg)
        else:
            print(msg)

    # categories
    categories = [{
        "id": int(args.category_id),
        "name": args.category_name,
        "supercategory": "object",
        "isthing": 1,
        "color": [220, 20, 60],
    }]
    if args.add_background_stuff:
        categories.append({
            "id": int(args.background_category_id),
            "name": args.background_category_name,
            "supercategory": "background",
            "isthing": 0,
            "color": [100, 100, 100],
        })

    images = []
    annotations = []

    image_id = 1
    processed = 0
    skipped = 0
    total_instances = 0

    for sp in splits:
        items = per_split[sp]
        if not items:
            continue
        print(f"[convert] {sp}: processing {len(items)} images …")
        for rel, img_path, msk_path in tqdm(items, desc=f"[{sp}]"):
            if not msk_path or not os.path.isfile(msk_path):
                skipped += 1
                continue

            I = cv2.imread(img_path, cv2.IMREAD_COLOR)
            M = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
            if I is None or M is None:
                skipped += 1
                continue
            if M.ndim == 3:
                M = cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)

            H, W = I.shape[:2]
            if (M.shape[0], M.shape[1]) != (H, W):
                M = cv2.resize(M, (W, H), interpolation=cv2.INTER_NEAREST)

            if args.from_binary_components:
                M = connected_components_from_binary(M)

            seg_global = np.zeros((H, W), dtype=np.uint32)
            seg_infos = []

            local_ids = np.unique(M)
            local_ids = local_ids[local_ids > 0]

            inst_count = 0
            for lid in local_ids:
                mask_bin = (M == lid)
                area = int(mask_bin.sum())
                if area < args.min_area:
                    continue
                bbox, _ = bbox_from_mask(mask_bin)
                gid = image_id * args.id_multiplier + int(lid)
                seg_global[mask_bin] = gid
                seg_infos.append({
                    "id": int(gid),
                    "category_id": int(args.category_id),
                    "area": int(area),
                    "bbox": [int(x) for x in bbox],
                    "iscrowd": 0,
                })
                inst_count += 1

            # optional background stuff to cover the rest
            if args.add_background_stuff:
                bg_mask = seg_global == 0
                if bg_mask.any():
                    bbox_bg, area_bg = bbox_from_mask(bg_mask)
                    if area_bg > 0:
                        # In COCO panoptic, background stuff gets its own segment id as well
                        gid_bg = image_id * args.id_multiplier  # reserve 0 local id for bg
                        seg_global[bg_mask] = gid_bg
                        seg_infos.append({
                            "id": int(gid_bg),
                            "category_id": int(args.background_category_id),
                            "area": int(area_bg),
                            "bbox": [int(x) for x in bbox_bg],
                            "iscrowd": 0,
                        })

            # write RGB-encoded panoptic PNG
            rel_dir = os.path.dirname(rel)
            base = os.path.splitext(os.path.basename(rel))[0]
            out_dir_nested = os.path.join(panoptic_dirs[sp], rel_dir)
            ensure_dir(out_dir_nested)
            out_png = os.path.join(out_dir_nested, f"{base}.png")
            rgb = rgb_encode_id(seg_global)
            cv2.imwrite(out_png, rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            rel_img_for_json = rel.replace('\\', '/')
            images.append({
                "id": int(image_id),
                "file_name": rel_img_for_json,
                "width": int(W),
                "height": int(H),
                "split": sp,
            })
            annotations.append({
                "image_id": int(image_id),
                "file_name": f"panoptic_{sp}/{rel_dir}/{base}.png",
                "segments_info": seg_infos,
            })

            processed += 1
            total_instances += inst_count
            image_id += 1

        print(f"[convert] {sp}: done. processed so far: {processed}, skipped: {skipped}")

    out_json = os.path.join(ann_dir, "panoptic.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }, f, ensure_ascii=False, indent=2)

    print("\n[DONE] COCO-Panoptic build")
    print(f"  images scanned:   {total_found}")
    print(f"  images processed: {processed}")
    print(f"  images skipped:   {skipped}")
    print(f"  total instances:  {total_instances}")
    print(f"  out_root:         {args.out_root}")
    print(f"  panoptic.json:    {out_json}")

if __name__ == "__main__":
    main()

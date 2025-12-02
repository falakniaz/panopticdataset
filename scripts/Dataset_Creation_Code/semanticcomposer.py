import os, re, cv2, argparse, random
import numpy as np
from glob import glob
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

# ---------- Configurable patterns ----------
IMG_EXTS  = {".jpg",".jpeg",".png",".bmp",".JPG",".JPEG",".PNG",".BMP"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

# Include your hyphen style here: {base}-mask.png
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

# ---------- Utilities ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder): return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1] in IMG_EXTS
    )

def list_images_recursive(folder: str) -> List[str]:
    out=[]
    for root,_,files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1] in IMG_EXTS:
                out.append(os.path.relpath(os.path.join(root,f), folder))
    return sorted(out)

def read_mask_as_id(path: str) -> Optional[np.ndarray]:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None: return None
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return (m > 0).astype(np.uint8) * 255

def find_mask(mask_dir: str, img_rel: str) -> Optional[str]:
    """
    Try exact template matches (with many extensions).
    If not found, try same base with any mask ext.
    If still not found, fall back to number-based matching (e.g., 10904).
    """
    rel_dir  = os.path.dirname(img_rel)  # e.g., 'visible' or 'scene/visible'
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

    # 3) Number-based fallback (match by the largest numeric token)
    digits = re.findall(r"\d+", img_base)
    if digits:
        key = digits[-1]  # usually the frame index
        search_dir = os.path.join(mask_dir, rel_dir)
        if os.path.isdir(search_dir):
            candidates = []
            for e in MASK_EXTS:
                candidates.extend(glob(os.path.join(search_dir, f"*{key}*{e}")))
            if candidates:
                # prefer names with mask-ish hints, then closer to img_base
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

def tight_bbox(mask_bin: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0: return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def paste_rgba(dst_bgr: np.ndarray, src_bgr: np.ndarray, src_a: np.ndarray, top: int, left: int):
    H,W = dst_bgr.shape[:2]
    h,w = src_bgr.shape[:2]
    y0,y1 = max(0, top), min(H, top+h)
    x0,x1 = max(0, left), min(W, left+w)
    if y1 <= y0 or x1 <= x0: return
    sy0,sy1 = y0-top, y1-top
    sx0,sx1 = x0-left, x1-left
    roi = dst_bgr[y0:y1, x0:x1]
    src_crop = src_bgr[sy0:sy1, sx0:sx1]
    a = (src_a[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[..., None]
    roi[:] = (src_crop.astype(np.float32)*a + roi.astype(np.float32)*(1.0-a)).astype(np.uint8)

def paste_mask(dst_mask: np.ndarray, src_mask: np.ndarray, top: int, left: int, val: int):
    H,W = dst_mask.shape[:2]
    h,w = src_mask.shape[:2]
    y0,y1 = max(0, top), min(H, top+h)
    x0,x1 = max(0, left), min(W, left+w)
    if y1 <= y0 or x1 <= x0: return
    sy0,sy1 = y0-top, y1-top
    sx0,sx1 = x0-left, x1-left
    sub = dst_mask[y0:y1, x0:x1]
    sc  = src_mask[sy0:sy1, sx0:sx1]
    sub[sc > 0] = val

def transform_cutout(c_bgr: np.ndarray, c_m: np.ndarray,
                     scale: float, angle: float, flip: bool) -> Tuple[np.ndarray,np.ndarray]:
    h,w = c_bgr.shape[:2]
    nw = max(1, int(round(w*scale)))
    nh = max(1, int(round(h*scale)))
    b = cv2.resize(c_bgr, (nw,nh), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(c_m,   (nw,nh), interpolation=cv2.INTER_NEAREST)

    ctr = (nw/2.0, nh/2.0)
    M = cv2.getRotationMatrix2D(ctr, angle, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    bw = int(nh*sin + nw*cos)
    bh = int(nh*cos + nw*sin)
    M[0,2] += bw/2.0 - ctr[0]
    M[1,2] += bh/2.0 - ctr[1]

    b = cv2.warpAffine(b, M, (bw,bh), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    m = cv2.warpAffine(m, M, (bw,bh), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if flip:
        b = cv2.flip(b, 1)
        m = cv2.flip(m, 1)
    return b, m

def iou_overlap(occ: np.ndarray, prop: np.ndarray, top: int, left: int) -> float:
    H,W = occ.shape[:2]
    h,w = prop.shape[:2]
    y0,y1 = max(0, top), min(H, top+h)
    x0,x1 = max(0, left), min(W, left+w)
    if y1 <= y0 or x1 <= x0:
        return 1.0
    sy0,sy1 = y0-top, y1-top
    sx0,sx1 = x0-left, x1-left
    A = occ[y0:y1, x0:x1] > 0
    B = prop[sy0:sy1, sx0:sx1] > 0
    inter = np.logical_and(A,B).sum()
    uni   = np.logical_or(A,B).sum()
    return 0.0 if uni == 0 else inter/float(uni)

# ---------- Core processing ----------
def process_flat(img_dir: str, msk_dir: str, out_img_dir: str, out_msk_dir: str, args) -> int:
    ensure_dir(out_img_dir); ensure_dir(out_msk_dir)
    names = list_images_recursive(img_dir) if args.recursive else list_images(img_dir)
    total = 0
    for rel in tqdm(names, desc=f"[compose flat] {os.path.basename(os.path.dirname(img_dir)) or os.path.basename(img_dir)}"):
        img_path = os.path.join(img_dir, rel)
        msk_path = find_mask(msk_dir, rel)
        if msk_path is None:
            if args.verbose: print(f"[flat] mask missing for {rel}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        m   = read_mask_as_id(msk_path)
        if img is None or m is None:
            if args.verbose: print(f"[flat] unreadable img/mask for {rel}")
            continue

        H,W = img.shape[:2]
        if (m.shape[0], m.shape[1]) != (H,W):
            m = cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)

        bbox = tight_bbox(m)
        if bbox is None:
            if args.verbose: print(f"[flat] empty mask for {rel}")
            continue

        x1,y1,x2,y2 = bbox
        cut_bgr = img[y1:y2+1, x1:x2+1].copy()
        cut_msk = m[y1:y2+1, x1:x2+1].copy()

        canvas   = img.copy()
        out_mask = (m > 0).astype(np.uint8) * args.object_id
        occ      = (out_mask > 0).astype(np.uint8)

        # decide total objects K in final (original + extras)
        K_total = random.randint(args.min_copies, args.max_copies)
        extras  = max(0, K_total - 1)
        placed  = 0

        for _ in range(extras):
            success = False
            for _try in range(args.attempts_per_copy):
                sc  = np.random.uniform(args.scale_min, args.scale_max)
                ang = np.random.uniform(-args.rot_deg, args.rot_deg)
                fl  = args.allow_flip and (np.random.rand() < 0.5)
                tb, tm = transform_cutout(cut_bgr, cut_msk, sc, ang, fl)

                th, tw = tm.shape[:2]
                top  = np.random.randint(-th//2, H-1)
                left = np.random.randint(-tw//2, W-1)

                if iou_overlap(occ, tm, top, left) <= 0.35:
                    paste_rgba(canvas, tb, tm, top, left)
                    paste_mask(out_mask, tm, top, left, args.object_id)
                    paste_mask(occ, tm, top, left, 1)
                    success = True
                    placed += 1
                    break
            if not success and args.verbose:
                print(f"[flat] could not place extra copy for {rel}")

        # write outputs (mirror relative path)
        base, ext = os.path.splitext(os.path.basename(rel))
        rel_dir   = os.path.dirname(rel)

        out_img_sub = os.path.join(out_img_dir, rel_dir)
        out_msk_sub = os.path.join(out_msk_dir, rel_dir)
        ensure_dir(out_img_sub); ensure_dir(out_msk_sub)

        outI = os.path.join(out_img_sub, f"{base}_multi{1+placed}{ext if ext else '.jpg'}")
        outM = os.path.join(out_msk_sub, f"{base}_multi{1+placed}.png")

        if ext.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(outI, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(outI, canvas)
        cv2.imwrite(outM, out_mask, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        total += 1
    return total

def process_split(root_in: str, root_out: str, split: str, args) -> int:
    # allow mapping image split name -> mask split name (e.g., images -> labels)
    m_alias: Dict[str,str] = {}
    if args.mask_split_aliases:
        for kv in args.mask_split_aliases.split(","):
            if ":" in kv:
                k,v = kv.split(":",1)
                m_alias[k.strip()] = v.strip()

    img_root = os.path.join(root_in, args.images_dirname)
    msk_root = os.path.join(root_in, args.masks_dirname)

    img_split = os.path.join(img_root, split)

    mask_split_name = m_alias.get(split, split)
    msk_split = os.path.join(msk_root, mask_split_name)

    if not (os.path.isdir(img_split) and os.path.isdir(msk_split)):
        print(f"[{split}] missing split dirs: img({img_split}) mask({msk_split}), skip")
        return 0

    out_img = os.path.join(root_out, args.images_dirname, split)
    out_msk = os.path.join(root_out, args.masks_dirname, mask_split_name)

    # If flat flag set, process directly; else try subfolders, fall back to flat.
    if args.flat:
        return process_flat(img_split, msk_split, out_img, out_msk, args)

    subs = sorted([d for d in os.listdir(img_split) if os.path.isdir(os.path.join(img_split, d))])
    if not subs:
        # No subfolders -> treat as flat
        return process_flat(img_split, msk_split, out_img, out_msk, args)

    total = 0
    for sub in tqdm(subs, desc=f"[compose] {split}"):
        sub_img = os.path.join(img_split, sub)
        # For mask, prefer matching subdir name; fall back if missing
        sub_msk_candidate = os.path.join(msk_split, sub)
        sub_msk = sub_msk_candidate if os.path.isdir(sub_msk_candidate) else msk_split
        if not os.path.isdir(sub_msk):
            if args.verbose: print(f"[{split}/{sub}] no matching mask subdir")
            continue
        total += process_flat(sub_img, sub_msk,
                              os.path.join(out_img, sub),
                              os.path.join(out_msk, sub if os.path.isdir(sub_msk_candidate) else ""),
                              args)
    return total

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_root", required=True, help="Dataset root (e.g., /home/falak/UAV_dataset/combined_uav)")
    ap.add_argument("--out_root", required=True, help="Output root; mirrors input/mask structure inside")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--images_dirname", default="UAV-Image", help="Top-level images dir name (e.g., 'input')")
    ap.add_argument("--masks_dirname", default="UAV-Mask", help="Top-level masks dir name (e.g., 'labels')")
    ap.add_argument("--mask_split_aliases", default="", help="Comma-separated mapping like 'images:labels,val:val' if mask split name differs")
    ap.add_argument("--min_copies", type=int, default=2, help="Min total objects (including original)")
    ap.add_argument("--max_copies", type=int, default=5, help="Max total objects (including original)")
    ap.add_argument("--attempts_per_copy", type=int, default=60, help="Placement attempts per extra object")
    ap.add_argument("--scale_min", type=float, default=0.7)
    ap.add_argument("--scale_max", type=float, default=1.3)
    ap.add_argument("--rot_deg", type=float, default=18.0)
    ap.add_argument("--allow_flip", action="store_true")
    ap.add_argument("--object_id", type=int, default=1, help="Semantic class id written in output mask")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--flat", action="store_true", help="Treat each split dir as flat (no subfolders)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into modality dirs like 'visible/' and 'infrared/'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    ensure_dir(args.out_root)

    total = 0
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        total += process_split(args.original_root, args.out_root, sp, args)

    print(f"[OK] Multi-object composition complete. Wrote {total} samples to {args.out_root}")

if __name__ == "__main__":
    main()

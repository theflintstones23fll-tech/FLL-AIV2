"""
Archaeological Fragment Reassembly via Arc-Based Fracture Matching
==================================================================
Designed for ceramic/glass sherds photographed on dark backgrounds.

Core approach:
1. Segment artifact from background using Otsu threshold
2. Classify each contour point as smooth-arc vs fracture via perpendicular deviation
3. Extract smooth arc segments and fit circles to them (vessel wall geometry)
4. Match arcs that share the same circle radius → same vessel
5. Score by angular adjacency (small gap = adjacent pieces)
6. Visualise edge classification + assembled reconstruction
"""

import cv2
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# SEGMENTATION
# ──────────────────────────────────────────────

def get_artifact_contour(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract the main artifact contour from an image with dark background.
    Masks out the bottom 38% (meter stick area) before detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove meter stick (bottom portion)
    h, w = thresh.shape
    thresh[int(h * 0.62):, :] = 0

    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


# ──────────────────────────────────────────────
# EDGE CLASSIFICATION
# ──────────────────────────────────────────────

def compute_deviation(pts: np.ndarray, window: int = 25) -> np.ndarray:
    """
    For each contour point, measure perpendicular deviation from the chord
    connecting its neighbours ±window steps away, normalised by chord length.

    Low value  → smooth curve or straight edge (vessel wall / rim)
    High value → jagged/irregular (fracture break)
    """
    n = len(pts)
    devs = []
    for i in range(n):
        prev = pts[(i - window) % n]
        curr = pts[i]
        nxt  = pts[(i + window) % n]
        chord     = nxt - prev
        chord_len = np.linalg.norm(chord) + 1e-6
        chord_dir = chord / chord_len
        v    = curr - prev
        perp = abs(v[0] * chord_dir[1] - v[1] * chord_dir[0])
        devs.append(perp / chord_len)
    return np.array(devs)


def extract_smooth_segments(pts: np.ndarray, devs: np.ndarray,
                             smooth_thresh: float = 0.08,
                             min_len: int = 30) -> list[dict]:
    """
    Find contiguous runs of low-deviation points (smooth arcs).
    Returns list of dicts with 'points' and 'length'.
    """
    n = len(pts)
    is_smooth = devs < smooth_thresh
    segments = []
    i = 0
    while i < n:
        if is_smooth[i]:
            j = i
            while j < n and is_smooth[j]:
                j += 1
            if j - i >= min_len:
                segments.append({
                    'points': pts[i:j],
                    'length': j - i
                })
            i = j
        else:
            i += 1
    return segments


# ──────────────────────────────────────────────
# CIRCLE FITTING
# ──────────────────────────────────────────────

def resample(pts: np.ndarray, n: int = 150) -> np.ndarray:
    """Resample a polyline to n equally-spaced points."""
    dists = np.zeros(len(pts))
    for i in range(1, len(pts)):
        dists[i] = dists[i - 1] + np.linalg.norm(pts[i] - pts[i - 1])
    total = dists[-1]
    if total < 1e-6:
        return pts
    t = np.linspace(0, total, n)
    return np.column_stack([
        np.interp(t, dists, pts[:, 0]),
        np.interp(t, dists, pts[:, 1])
    ])


def fit_circle(pts: np.ndarray) -> Optional[tuple]:
    """
    Algebraic circle fit. Returns (cx, cy, radius, mean_residual) or None.
    Low residual means the points genuinely lie on a circular arc.
    """
    if len(pts) < 5:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx = result[0] / 2
        cy = result[1] / 2
        r  = np.sqrt(result[2] + cx ** 2 + cy ** 2)
        residual = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r).mean()
        return cx, cy, r, residual
    except Exception:
        return None


# ──────────────────────────────────────────────
# ARC MATCHING
# ──────────────────────────────────────────────

def match_arc_segments(seg1: dict, seg2: dict) -> tuple[float, Optional[dict]]:
    """
    Test whether two arc segments belong to the same vessel circle.

    Scoring:
    - radius_score : radii must be similar (same vessel)
    - gap_score    : after aligning to same circle, arcs should be angularly adjacent
    - overlap_score: arcs should not overlap

    Returns (score 0-100, transform_info or None)
    """
    r1 = resample(seg1['points'], 150)
    r2 = resample(seg2['points'], 150)

    c1 = fit_circle(r1)
    c2 = fit_circle(r2)

    if c1 is None or c2 is None:
        return 0.0, None

    cx1, cy1, rad1, res1 = c1
    cx2, cy2, rad2, res2 = c2

    # Both arcs must actually be arc-shaped (low residual)
    if res1 > 5.0 or res2 > 5.0:
        return 0.0, None

    # Radii must be similar → same vessel
    ratio = min(rad1, rad2) / (max(rad1, rad2) + 1e-6)
    if ratio < 0.70:
        return 0.0, None
    radius_score = ratio

    # Translate seg2's circle center onto seg1's
    dx = cx1 - cx2
    dy = cy1 - cy2
    r2_shifted = r2 + np.array([dx, dy])

    # Angular positions on the shared circle
    def arc_angles(pts, cx, cy):
        return np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)

    a1 = arc_angles(r1, cx1, cy1)
    a2 = arc_angles(r2_shifted, cx1, cy1)

    a1_lo, a1_hi = a1.min(), a1.max()
    a2_lo, a2_hi = a2.min(), a2.max()

    # Angular gap between the two arcs (minimum of the two possible gaps)
    gap_fwd = (a2_lo - a1_hi) % (2 * np.pi)
    gap_bck = (a1_lo - a2_hi) % (2 * np.pi)
    angular_gap_rad = min(gap_fwd, gap_bck)
    angular_gap_deg = np.degrees(angular_gap_rad)

    # Adjacent pieces → small gap. Score falls off with gap size.
    gap_score = float(np.exp(-angular_gap_deg / 30.0))

    # Overlap penalty
    overlap = max(0.0, min(a1_hi, a2_hi) - max(a1_lo, a2_lo))
    overlap_score = 1.0 - min(overlap / (np.pi / 4), 1.0)

    total = (radius_score * 0.40 + gap_score * 0.40 + overlap_score * 0.20) * 100.0

    r_shared = (rad1 + rad2) / 2.0

    transform = {
        'dx': float(dx),
        'dy': float(dy),
        'rotation_deg': 0.0,
        'shared_circle': (float(cx1), float(cy1), float(r_shared)),
        'angular_gap_deg': float(angular_gap_deg),
        'arc1_angles_deg': (float(np.degrees(a1_lo)), float(np.degrees(a1_hi))),
        'arc2_angles_deg': (float(np.degrees(a2_lo)), float(np.degrees(a2_hi))),
        'radius_ratio': float(ratio),
    }

    return round(total, 2), transform


# ──────────────────────────────────────────────
# MAIN MATCHING PIPELINE
# ──────────────────────────────────────────────

def find_connection(img1_path: str, img2_path: str) -> dict:
    """
    Full pipeline: load images, segment, classify edges, match arcs.

    Returns a result dict with:
        connects          bool
        score             float 0-100
        message           str
        alignment         dict  (dx, dy, rotation_deg)
        metrics           dict
        internal          dict  (contours, segments, etc. for visualisation)
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return {'error': 'Could not read one or both images', 'connects': False, 'score': 0}

    c1 = get_artifact_contour(img1)
    c2 = get_artifact_contour(img2)

    if c1 is None or c2 is None:
        return {'error': 'Could not detect artifact contour', 'connects': False, 'score': 0}

    pts1 = c1.squeeze().astype(np.float32)
    pts2 = c2.squeeze().astype(np.float32)

    dev1 = compute_deviation(pts1)
    dev2 = compute_deviation(pts2)

    segs1 = extract_smooth_segments(pts1, dev1)
    segs2 = extract_smooth_segments(pts2, dev2)

    # Try every combination of smooth segments
    best_score = 0.0
    best_transform = None
    best_pair = (0, 0)

    for i, s1 in enumerate(segs1):
        for j, s2 in enumerate(segs2):
            score, transform = match_arc_segments(s1, s2)
            if score > best_score:
                best_score = score
                best_transform = transform
                best_pair = (i, j)

    connects = best_score >= 60.0

    result = {
        'connects': connects,
        'score': best_score,
        'message': '',
        'alignment': None,
        'metrics': {},
        'internal': {
            'img1': img1, 'img2': img2,
            'pts1': pts1, 'pts2': pts2,
            'dev1': dev1, 'dev2': dev2,
            'segs1': segs1, 'segs2': segs2,
            'best_pair': best_pair,
            'contour1': c1, 'contour2': c2,
        }
    }

    if best_transform is None:
        result['message'] = 'No matching arc segments found'
        return result

    si, sj = best_pair
    cx, cy, r = best_transform['shared_circle']
    gap = best_transform['angular_gap_deg']

    result['message'] = (
        f"{'MATCH' if connects else 'Possible match'}: "
        f"Piece 1 arc {si} connects with Piece 2 arc {sj} "
        f"(shared vessel radius {r:.0f}px, angular gap {gap:.1f}°)"
    )
    result['alignment'] = {
        'dx': best_transform['dx'],
        'dy': best_transform['dy'],
        'rotation_deg': best_transform['rotation_deg'],
    }
    result['metrics'] = {
        'shared_radius_px': r,
        'angular_gap_deg': gap,
        'radius_ratio': best_transform['radius_ratio'],
        'arc1_angles_deg': best_transform['arc1_angles_deg'],
        'arc2_angles_deg': best_transform['arc2_angles_deg'],
    }
    result['internal']['transform'] = best_transform

    return result


# ──────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────

def visualise_edges(img: np.ndarray, pts: np.ndarray, devs: np.ndarray,
                    segs: list[dict], title: str = '') -> np.ndarray:
    """
    Draw contour colour-coded by deviation:
      Blue  → smooth arc (vessel wall)
      Red   → fracture break
    Cyan overlay on the detected smooth arc segments.
    """
    out = img.copy()
    for pt, d in zip(pts, devs):
        t     = min(d / 0.12, 1.0)
        color = (int(255 * (1 - t)), 50, int(255 * t))   # blue→red
        cv2.circle(out, (int(pt[0]), int(pt[1])), 2, color, -1)

    for si, seg in enumerate(segs):
        sp = seg['points'].astype(np.int32)
        for k in range(len(sp) - 1):
            cv2.line(out, tuple(sp[k]), tuple(sp[k + 1]), (0, 230, 230), 3)
        mid = sp[len(sp) // 2]
        cv2.putText(out, f'Arc {si}', (mid[0] + 5, mid[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 230), 1)

    if title:
        cv2.putText(out, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(out, 'CYAN=vessel arc  RED=fracture', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
    return out


def visualise_assembly(result: dict) -> Optional[np.ndarray]:
    """
    Place both pieces on a shared canvas aligned to their common vessel circle.
    Highlights the matched arc segments and marks the join point.
    """
    if not result.get('internal') or result['internal'].get('transform') is None:
        return None

    intern = result['internal']
    img1   = intern['img1']
    img2   = intern['img2']
    pts1   = intern['pts1']
    pts2   = intern['pts2']
    segs1  = intern['segs1']
    segs2  = intern['segs2']
    si, sj = intern['best_pair']
    tf     = intern['transform']

    dx, dy = tf['dx'], tf['dy']
    pts2_t = pts2 + np.array([dx, dy])

    # Combined bounding box
    all_pts = np.vstack([pts1, pts2_t])
    margin  = 40
    x_min, y_min = all_pts.min(axis=0) - margin
    x_max, y_max = all_pts.max(axis=0) + margin

    scale = min(860 / (x_max - x_min), 540 / (y_max - y_min), 1.8)
    CW = int((x_max - x_min) * scale) + 60
    CH = int((y_max - y_min) * scale) + 130
    canvas = np.full((CH, CW, 3), 18, dtype=np.uint8)
    ox = int(-x_min * scale + 30)
    oy = int(-y_min * scale + 80)

    def to_canvas(pts):
        return (pts * scale + np.array([ox, oy])).astype(np.int32)

    def blit(img_src, pts_orig, pts_c, outline_col):
        sx, sy, sw, sh = cv2.boundingRect(pts_orig.astype(np.int32))
        sx1, sy1 = max(0, sx - 2), max(0, sy - 2)
        sx2, sy2 = min(img_src.shape[1], sx + sw + 2), min(img_src.shape[0], sy + sh + 2)
        src = img_src[sy1:sy2, sx1:sx2]
        if src.size == 0:
            return
        sw_s, sh_s = int((sx2 - sx1) * scale), int((sy2 - sy1) * scale)
        if sw_s < 1 or sh_s < 1:
            return
        src_r = cv2.resize(src, (sw_s, sh_s))
        mask  = np.zeros((sh_s, sw_s), dtype=np.uint8)
        lpts  = ((pts_orig - np.array([sx1, sy1])) * scale).astype(np.int32)
        cv2.fillPoly(mask, [lpts], 255)

        bx, by, _, _ = cv2.boundingRect(pts_c)
        dst_x, dst_y = bx, by
        d_x2 = min(canvas.shape[1], dst_x + sw_s)
        d_y2 = min(canvas.shape[0], dst_y + sh_s)
        s_x2, s_y2 = min(sw_s, d_x2 - dst_x), min(sh_s, d_y2 - dst_y)
        if s_x2 <= 0 or s_y2 <= 0:
            return
        patch = src_r[:s_y2, :s_x2]
        msk   = mask[:s_y2, :s_x2]
        reg   = canvas[dst_y:dst_y + s_y2, dst_x:dst_x + s_x2]
        if patch.shape != reg.shape:
            return
        canvas[dst_y:dst_y + s_y2, dst_x:dst_x + s_x2] = np.where(
            msk[:, :, None] > 0, patch, reg)
        cv2.drawContours(canvas, [pts_c.reshape(-1, 1, 2)], -1, outline_col, 2)

    blit(img1, pts1,   to_canvas(pts1),   (80, 200, 255))   # cyan  = piece 1
    blit(img2, pts2,   to_canvas(pts2_t), (80, 255, 150))   # green = piece 2

    # Draw shared circle guide
    cx_c = int(tf['shared_circle'][0] * scale + ox)
    cy_c = int(tf['shared_circle'][1] * scale + oy)
    r_c  = int(tf['shared_circle'][2] * scale)
    cv2.ellipse(canvas, (cx_c, cy_c), (r_c, r_c), 0, 0, 360, (50, 50, 75), 1)

    # Highlight matched arcs
    if si < len(segs1):
        arc1 = to_canvas(segs1[si]['points'])
        for k in range(len(arc1) - 1):
            cv2.line(canvas, tuple(arc1[k]), tuple(arc1[k + 1]), (0, 240, 240), 3)

    if sj < len(segs2):
        arc2 = to_canvas(segs2[sj]['points'] + np.array([dx, dy]))
        for k in range(len(arc2) - 1):
            cv2.line(canvas, tuple(arc2[k]), tuple(arc2[k + 1]), (0, 240, 160), 3)

        # Join arrow between nearest endpoints
        if si < len(segs1):
            ep1 = [arc1[0], arc1[-1]]
            ep2 = [arc2[0], arc2[-1]]
            best_d = 9999
            bp1, bp2 = ep1[0], ep2[0]
            for p1 in ep1:
                for p2 in ep2:
                    d = np.linalg.norm(np.array(p1, float) - np.array(p2, float))
                    if d < best_d:
                        best_d, bp1, bp2 = d, p1, p2
            cv2.arrowedLine(canvas, tuple(bp1), tuple(bp2), (255, 220, 0), 2, tipLength=0.25)
            mid = ((np.array(bp1) + np.array(bp2)) // 2)
            cv2.putText(canvas, 'JOIN HERE', (mid[0] + 8, mid[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 0), 2)

    # Labels
    score = result.get('score', 0)
    r_shared = tf['shared_circle'][2]
    gap      = tf['angular_gap_deg']
    cv2.putText(canvas, 'RECONSTRUCTED ARTIFACT', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2)
    cv2.putText(canvas,
                f'Score: {score:.1f}/100  |  Vessel radius: {r_shared:.0f}px  |  Arc gap: {gap:.1f}deg',
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1)
    cv2.putText(canvas, 'CYAN = Piece 1   GREEN = Piece 2   YELLOW = join point',
                (20, CH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    return canvas


# ──────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────

def analyse_fragment_pair(img1_path: str, img2_path: str,
                           output_prefix: str = 'output') -> dict:
    """
    Main entry point. Pass two image paths; get back a result dict and
    three saved diagnostic images:
        {prefix}_edges.jpg    — edge classification for each piece
        {prefix}_assembly.jpg — reconstructed artifact with join marked

    Example:
        result = analyse_fragment_pair('shard_A.jpg', 'shard_B.jpg',
                                       output_prefix='site_001')
        print(result['connects'], result['score'])
        print(result['message'])
    """
    result = find_connection(img1_path, img2_path)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return result

    intern = result.get('internal', {})
    img1   = intern.get('img1')
    img2   = intern.get('img2')

    # ── Edge classification panels ──
    if img1 is not None and img2 is not None:
        e1 = visualise_edges(img1, intern['pts1'], intern['dev1'],
                             intern['segs1'], 'Piece 1 – edge classification')
        e2 = visualise_edges(img2, intern['pts2'], intern['dev2'],
                             intern['segs2'], 'Piece 2 – edge classification')
        h_max = max(e1.shape[0], e2.shape[0])

        def pad_h(img, h):
            if img.shape[0] < h:
                pad = np.full((h - img.shape[0], img.shape[1], 3), 20, dtype=np.uint8)
                return np.vstack([img, pad])
            return img

        cv2.imwrite(f'{output_prefix}_edges.jpg',
                    np.hstack([pad_h(e1, h_max), pad_h(e2, h_max)]))

    # ── Assembly ──
    assembly = visualise_assembly(result)
    if assembly is not None:
        cv2.imwrite(f'{output_prefix}_assembly.jpg', assembly)

    # Strip internal data before returning
    clean = {k: v for k, v in result.items() if k != 'internal'}
    return clean


# ──────────────────────────────────────────────
# LEGACY BRIDGE  (keeps existing test.py working)
# ──────────────────────────────────────────────

def from_existing_pipeline(img1_path: str, img2_path: str,
                            poly1_px=None, poly2_px=None,
                            output_prefix: str = 'output') -> dict:
    """
    Drop-in replacement for the old from_existing_pipeline().
    poly1_px / poly2_px are accepted but ignored — the new algorithm
    re-segments directly from the images for better accuracy.
    """
    return analyse_fragment_pair(img1_path, img2_path, output_prefix)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Usage: python fracture_matching.py <img1> <img2> [output_prefix]')
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    prefix    = sys.argv[3] if len(sys.argv) > 3 else 'output'

    result = analyse_fragment_pair(img1_path, img2_path, prefix)

    print('\n' + '=' * 60)
    print('FRACTURE MATCH RESULT')
    print('=' * 60)
    print(f"  Connects:  {'✓ YES' if result.get('connects') else '✗ NO'}")
    print(f"  Score:     {result.get('score', 0):.1f} / 100")
    print(f"  {result.get('message', '')}")
    if result.get('metrics'):
        m = result['metrics']
        print(f"  Vessel radius: {m.get('shared_radius_px', 0):.0f} px")
        print(f"  Angular gap:   {m.get('angular_gap_deg', 0):.1f}°")
        print(f"  Radius ratio:  {m.get('radius_ratio', 0):.3f}")
    print()
    print(f"  Saved: {prefix}_edges.jpg")
    if result.get('connects'):
        print(f"  Saved: {prefix}_assembly.jpg")
    print('=' * 60)

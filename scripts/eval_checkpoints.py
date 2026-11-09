#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
checkpoint 定量评估：色彩贴合度 / 纹理能量 / 风格泄漏率 / 可选 CLIP 风格相似度。

对应方案 §7.2。核心原则：不看 loss，看固定 prompt 网格 + 这些指标 + 人工盲评。

目录约定：
  --real      真实作品集（参考分布）
  --gen       生成图根目录，其每个【子目录】= 一个 checkpoint 的网格图；
              若该目录直接放图片，则视为单个 checkpoint
  --base      可选，底模（不挂 LoRA）出图，作为泄漏基线
  --leakage   可选，不带触发词的出图，用于计算风格泄漏率

仅依赖 Pillow + 标准库；--clip 额外需要 torch + open_clip。

用法：
  python scripts/eval_checkpoints.py --real dataset/train/10_mdtr --gen output/grids
  python scripts/eval_checkpoints.py --real dataset/train/10_mdtr --gen output/grids \
      --base output/base --leakage output/no_trigger --clip
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_dataset import EXTS, image_metrics  # noqa: E402


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS) if d.is_dir() else []


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def aggregate(d: Path) -> dict | None:
    """把一个目录下所有图聚合成一条风格指纹。"""
    files = list_images(d)
    if not files:
        return None
    metrics = []
    for p in files:
        try:
            metrics.append(image_metrics(p))
        except Exception as e:  # noqa: BLE001
            print(f"[!] 跳过 {p.name}: {e}")
    if not metrics:
        return None
    hists = [m["hue_hist"] for m in metrics]
    return {
        "n": len(metrics),
        "mean_s": _mean([m["mean_s"] for m in metrics]),
        "mean_v": _mean([m["mean_v"] for m in metrics]),
        "low_sat_ratio": _mean([m["low_sat_ratio"] for m in metrics]),
        "high_sat_ratio": _mean([m["high_sat_ratio"] for m in metrics]),
        "bright_ratio": _mean([m["bright_ratio"] for m in metrics]),
        "lap_var": statistics.median([m["lap_var"] for m in metrics]),
        "hue_hist": [_mean([h[i] for h in hists]) for i in range(12)],
        "per_image": metrics,
    }


def color_distance(a: dict, b: dict) -> float:
    """色板距离：色相分布 L1 + 饱和度/亮度/低饱和占比差。越小越贴合。"""
    hue_l1 = sum(abs(x - y) for x, y in zip(a["hue_hist"], b["hue_hist"])) / 2.0
    return (hue_l1
            + abs(a["mean_s"] - b["mean_s"])
            + 0.5 * abs(a["mean_v"] - b["mean_v"])
            + 0.5 * abs(a["low_sat_ratio"] - b["low_sat_ratio"]))


def texture_gap(a: dict, b: dict) -> float:
    """笔触/颗粒能量差（对数尺度），过大说明学成了平涂或过度噪点。"""
    return abs(math.log1p(max(a["lap_var"], 0.0)) - math.log1p(max(b["lap_var"], 0.0)))


def _as_fingerprint(m: dict) -> dict:
    return {k: m[k] for k in ("hue_hist", "mean_s", "mean_v", "low_sat_ratio")}


def clip_scores(real: Path, gen_dirs: list[tuple[str, Path]]) -> dict[str, float]:
    """可选：CLIP image embedding 与真实作品集质心的余弦相似度。缺依赖则跳过。"""
    try:
        import torch
        import open_clip
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        model = model.to(device).eval()
    except Exception as e:  # noqa: BLE001
        print(f"[i] 跳过 CLIP 指标（需 torch + open_clip，首次运行需下载权重）: {type(e).__name__}: {e}")
        return {}

    def embed(d: Path):
        feats = []
        for p in list_images(d):
            with Image.open(p) as im:
                x = preprocess(im.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f)
        return torch.cat(feats, dim=0) if feats else None

    rv = embed(real)
    if rv is None:
        return {}
    centroid = rv.mean(dim=0)
    centroid = centroid / max(centroid.norm().item(), 1e-8)
    out: dict[str, float] = {}
    for name, d in gen_dirs:
        gv = embed(d)
        if gv is not None:
            out[name] = float((gv @ centroid).mean().item())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA checkpoint 风格评估")
    ap.add_argument("--real", required=True, help="真实作品集目录")
    ap.add_argument("--gen", required=True, help="生成图根目录（子目录 = 各 checkpoint）")
    ap.add_argument("--base", default="", help="底模直出图目录（泄漏基线）")
    ap.add_argument("--leakage", default="", help="不带触发词的出图目录")
    ap.add_argument("--out", default="eval_report.csv")
    ap.add_argument("--clip", action="store_true", help="启用 CLIP 风格相似度")
    args = ap.parse_args()

    real_dir, gen_root = Path(args.real), Path(args.gen)
    real = aggregate(real_dir)
    if real is None:
        print(f"[!] 参考目录无有效图片：{real_dir}")
        return 1

    subdirs = sorted(d for d in gen_root.iterdir() if d.is_dir()) if gen_root.is_dir() else []
    targets = [(d.name, d) for d in subdirs if list_images(d)]
    if not targets and list_images(gen_root):
        targets = [(gen_root.name, gen_root)]

    rows = []
    for name, d in targets:
        g = aggregate(d)
        if g is None:
            continue
        cd = color_distance(g, real)
        tg = texture_gap(g, real)
        rows.append({
            "checkpoint": name,
            "n_images": g["n"],
            "color_distance": round(cd, 4),
            "texture_gap": round(tg, 4),
            "score": round(cd + 0.30 * tg, 4),
            "mean_s": round(g["mean_s"], 4),
            "mean_v": round(g["mean_v"], 4),
            "low_sat_ratio": round(g["low_sat_ratio"], 4),
            "high_sat_ratio": round(g["high_sat_ratio"], 4),
            "lap_var": round(g["lap_var"], 2),
        })

    if not rows:
        print(f"[!] 未在 {gen_root} 找到任何 checkpoint 网格图")
        return 1

    if args.clip:
        cs = clip_scores(real_dir, targets)
        for r in rows:
            if r["checkpoint"] in cs:
                r["clip_sim"] = round(cs[r["checkpoint"]], 4)
                r["score"] = round(r["score"] + 0.5 * (1.0 - cs[r["checkpoint"]]), 4)

    rows.sort(key=lambda r: r["score"])

    # 风格泄漏率（启发式）：无触发词图比底模基线更贴近目标色板的比例
    leak_rate = None
    if args.leakage and args.base:
        base_agg = aggregate(Path(args.base)) or {}
        leak_agg = aggregate(Path(args.leakage)) or {}
        per_base = [color_distance(_as_fingerprint(m), real) for m in base_agg.get("per_image", [])]
        per_leak = [color_distance(_as_fingerprint(m), real) for m in leak_agg.get("per_image", [])]
        if per_base and per_leak:
            thr = statistics.median(per_base) * 0.9
            leak_rate = _mean([1.0 if x < thr else 0.0 for x in per_leak])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["checkpoint", "n_images", "score", "color_distance", "texture_gap",
            "mean_s", "mean_v", "low_sat_ratio", "high_sat_ratio", "lap_var", "clip_sim"]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print("=" * 78)
    print(" checkpoint 评估（score 越小越贴合；仍需人工盲评确认）")
    print("=" * 78)
    print(f" 参考色板: mean_s={real['mean_s']:.3f}  mean_v={real['mean_v']:.3f}  "
          f"low_sat={real['low_sat_ratio']:.3f}  lap_var={real['lap_var']:.1f}")
    print("-" * 78)
    print(f" {'checkpoint':<26}{'score':>9}{'color':>9}{'texture':>10}{'mean_s':>9}{'lap_var':>10}{'clip':>8}")
    for r in rows:
        clip = r.get("clip_sim")
        print(f" {r['checkpoint']:<26}{r['score']:>9.4f}{r['color_distance']:>9.4f}"
              f"{r['texture_gap']:>10.4f}{r['mean_s']:>9.3f}{r['lap_var']:>10.1f}"
              f"{(f'{clip:8.3f}' if clip is not None else '       -')}")
    print("-" * 78)
    if leak_rate is not None:
        print(f" 风格泄漏率（无触发词）: {leak_rate * 100:.1f}%   "
              + ("偏高 -> 过拟合 / caption 泄漏，见方案 §7.6" if leak_rate > 0.3 else "可接受"))
    else:
        print(" 风格泄漏率: 未计算（需同时提供 --base 与 --leakage）")
    print(f" 推荐首选: {rows[0]['checkpoint']}   （再用固定网格图与人工盲评复核）")
    print(f" 报告已写出: {out.resolve()}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())

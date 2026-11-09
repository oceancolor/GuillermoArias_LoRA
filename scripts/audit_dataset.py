#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集审计：pHash 近重复检测 + 尺寸/长宽比/色彩统计 + caption 合规检查。

用途（对应方案 §3.2 / §3.3 / §3.5）：
  - 找出同一作品的多次压缩版本、九宫格、截图、疑似低质图
  - 体检色板分布（为 retro_muted / candy_bright 子标签提供依据）
  - 确保 caption 以触发词开头、且不含任何风格形容词

仅依赖 Pillow + Python 标准库（不需要 numpy / cv2 / imagehash）。
注意：本脚本【不做 AI 生成检测】——那类判断不可靠，必须人工复核。

用法：
  python scripts/audit_dataset.py --dir dataset/train/10_mdtr
  python scripts/audit_dataset.py --dir dataset/train/10_mdtr --trigger br70_print
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageFile, ImageFilter, ImageStat

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_TRIGGER = "mdtr_style"
SUBLABELS = {"retro_muted", "candy_bright", "br70_print", "ga_procreate"}

# 出现这些词说明 caption 把"风格"写进了文本，会让触发词失效（方案 §3.5）
# 与 configs/caption_rules_branchA.md §4 的禁用词表保持同步。
# 注意：完整子标签 token（retro_muted / candy_bright）会在检查前被剔除，不会误报。
STYLE_WORDS = [
    # 媒介 / 技法
    "oil painting", "painterly", "impasto", "impressionist", "brushstroke",
    "brush stroke", "thick paint", "textured paint", "watercolor", "gouache",
    "illustration", "art style", "in the style of",
    # 色彩 / 色板
    "muted", "low saturation", "desaturated", "retro", "vintage", "morandi",
    "pastel", "vibrant",
    # 氛围 / 情绪 / 评价
    "aesthetic", "cozy", "peaceful", "serene", "dreamy", "nostalgic",
    "healing", "soft", "gentle", "warm glow", "soft light", "beautiful",
    "stunning", "elegant", "masterpiece", "high quality",
    # 地域 / 署名
    "mediterranean", "guillermo arias", "inspired by", "by the artist",
    # 中文
    "油画", "厚涂", "笔触", "低饱和", "复古", "插画", "治愈", "松弛感",
    "氛围感", "高级感", "温柔", "慵懒", "静谧", "莫兰迪", "水彩", "水粉",
    "印象派", "颗粒感", "画风",
]

# 32 点 DCT-II 的前 8 个基函数，预计算供 pHash 使用
_N = 32
_COS = [[math.cos(math.pi * (2 * n + 1) * k / (2 * _N)) for n in range(_N)]
        for k in range(8)]
_ALPHA = [math.sqrt(1.0 / _N)] + [math.sqrt(2.0 / _N)] * 7

_LAP_KERNEL = ImageFilter.Kernel(
    (3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


def _pixels(im: Image.Image):
    """兼容 Pillow 新旧版本：11.3+ 用 get_flattened_data，旧版回退 getdata。"""
    if hasattr(im, "get_flattened_data"):
        return im.get_flattened_data()
    return im.getdata()


def phash(path: Path) -> int:
    """64-bit pHash：32x32 灰度 -> 可分离 2D DCT -> 8x8 低频 -> 中位数二值化。"""
    with Image.open(path) as im:
        pix = list(_pixels(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS)))
    rows = [pix[y * 32:(y + 1) * 32] for y in range(32)]

    # 行方向：每行取前 8 个 DCT 系数
    tmp = [[0.0] * 8 for _ in range(32)]
    for y in range(32):
        row = rows[y]
        for k in range(8):
            c = _COS[k]
            tmp[y][k] = sum(row[n] * c[n] for n in range(32)) * _ALPHA[k]

    # 列方向：对 8 个中间列各取前 8 个系数
    low = [[0.0] * 8 for _ in range(8)]
    for k in range(8):
        col = [tmp[y][k] for y in range(32)]
        for kk in range(8):
            c = _COS[kk]
            low[kk][k] = sum(col[n] * c[n] for n in range(32)) * _ALPHA[kk]

    med = statistics.median(
        [low[i][j] for i in range(8) for j in range(8) if not (i == 0 and j == 0)])
    bits = 0
    for i in range(8):
        for j in range(8):
            bits = (bits << 1) | (1 if low[i][j] > med else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def image_metrics(path: Path) -> dict:
    """返回尺寸、色彩分布、纹理能量、边框可疑度等可量化指标。"""
    with Image.open(path) as im:
        w, h = im.size
        hsv = im.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS).convert("HSV")
        gray = im.convert("L").resize((256, 256), Image.Resampling.LANCZOS)

    n = 0
    s_sum = v_sum = 0.0
    low_sat = high_sat = bright = hue_total = 0
    hue_counts = [0] * 12
    for hh, ss, vv in _pixels(hsv):
        s, v = ss / 255.0, vv / 255.0
        n += 1
        s_sum += s
        v_sum += v
        low_sat += s < 0.25
        high_sat += s > 0.60
        bright += v > 0.80
        if s > 0.15:
            hue_counts[min(11, hh * 12 // 256)] += 1
            hue_total += 1
    n = max(n, 1)

    # Laplacian 方差：粗粝厚涂应显著高于平涂/塑料感
    lap_var = ImageStat.Stat(gray.filter(_LAP_KERNEL)).stddev[0] ** 2

    # 四边窄条的方差：极低时可能是截图/拼图/带边框转载图（仅提示人工复核）
    bw = max(4, 256 // 25)
    strips = [
        gray.crop((0, 0, 256, bw)), gray.crop((0, 256 - bw, 256, 256)),
        gray.crop((0, 0, bw, 256)), gray.crop((256 - bw, 0, 256, 256)),
    ]
    border_std = sum(ImageStat.Stat(x).stddev[0] for x in strips) / 4.0

    return {
        "width": w,
        "height": h,
        "aspect": round(w / max(h, 1), 3),
        "mean_s": round(s_sum / n, 4),
        "mean_v": round(v_sum / n, 4),
        "low_sat_ratio": round(low_sat / n, 4),
        "high_sat_ratio": round(high_sat / n, 4),
        "bright_ratio": round(bright / n, 4),
        "lap_var": round(lap_var, 2),
        "border_std": round(border_std, 2),
        "hue_hist": [c / max(hue_total, 1) for c in hue_counts],
    }


def check_caption(txt: Path, trigger: str) -> list[str]:
    if not txt.exists():
        return ["缺 caption 文件"]
    text = txt.read_text(encoding="utf-8", errors="replace").strip()
    issues: list[str] = []
    if not text:
        return ["caption 为空"]
    if not text.lower().startswith(trigger.lower()):
        issues.append(f"触发词 {trigger} 未置于首位")
    # 去掉触发词与配色子标签后再查风格词
    rest = text
    for token in [trigger, *SUBLABELS]:
        rest = rest.replace(token, "")
    low = rest.lower()
    hits = [w for w in STYLE_WORDS if w.lower() in low]
    if hits:
        issues.append("含风格词: " + ", ".join(hits[:5]))
    if len(text.split()) < 8:
        issues.append("caption 过短，内容描述不足")
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA 风格数据集审计")
    ap.add_argument("--dir", default="dataset/train/10_mdtr", help="训练图目录（含同名 .txt caption）")
    ap.add_argument("--out", default="audit_report.csv", help="输出 CSV 路径")
    ap.add_argument("--trigger", default=DEFAULT_TRIGGER, help="触发词")
    ap.add_argument("--phash-threshold", type=int, default=6, help="pHash 汉明距离阈值，<= 视为近重复")
    args = ap.parse_args()

    root = Path(args.dir)
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in EXTS) if root.is_dir() else []
    if not files:
        print(f"[!] 未在 {root} 找到图片。请先按方案 §3.6 建立目录结构。")
        return 1

    rows, hashes = [], []
    for p in files:
        try:
            m = image_metrics(p)
            hv = phash(p)
        except Exception as e:  # noqa: BLE001
            print(f"[!] 无法读取 {p.name}: {e}")
            continue
        cap = check_caption(p.with_suffix(".txt"), args.trigger)
        flags = []
        if min(m["width"], m["height"]) < 768:
            flags.append("低分辨率")
        if m["aspect"] < 0.45 or m["aspect"] > 2.3:
            flags.append("极端长宽比")
        if m["border_std"] < 3.0:
            flags.append("疑似截图/边框(人工复核)")
        if p.stat().st_size < 40_000:
            flags.append("文件过小(可能强压缩)")
        flags.extend(cap)
        row = {k: v for k, v in m.items() if k != "hue_hist"}
        row.update({
            "file": p.name,
            "size_kb": round(p.stat().st_size / 1024, 1),
            "flags": "; ".join(flags) if flags else "OK",
            "hue_hist": "|".join(f"{x:.3f}" for x in m["hue_hist"]),
        })
        rows.append(row)
        hashes.append((p.name, hv))

    # 近重复分组（O(n^2)，n<=300 完全够用）
    dup_of: dict[str, str] = {}
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i][1], hashes[j][1]) <= args.phash_threshold:
                dup_of.setdefault(hashes[j][0], hashes[i][0])
    for r in rows:
        if r["file"] in dup_of:
            r["flags"] = ("" if r["flags"] == "OK" else r["flags"] + "; ") + "近重复->" + dup_of[r["file"]]
        r["near_duplicate_of"] = dup_of.get(r["file"], "")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["file", "width", "height", "aspect", "size_kb", "mean_s", "mean_v",
            "low_sat_ratio", "high_sat_ratio", "bright_ratio", "lap_var",
            "border_std", "near_duplicate_of", "hue_hist", "flags"]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if r["flags"] != "OK"]
    cap_issues = [r for r in rows
                  if "风格词" in r["flags"] or "触发词" in r["flags"] or "caption" in r["flags"].lower()]
    s_vals = sorted(r["mean_s"] for r in rows)
    lap_vals = sorted(r["lap_var"] for r in rows)

    print("=" * 64)
    print(f" 数据集审计报告  |  {root}")
    print("=" * 64)
    print(f" 图片总数            : {len(rows)}")
    print(f" 近重复              : {len(dup_of)}  (阈值 <= {args.phash_threshold})")
    print(f" 带风险标记          : {len(bad)}")
    print(f" caption 需修正      : {len(cap_issues)}   <- 必须为 0 才能开训")
    if s_vals:
        print(f" 平均饱和度 mean_s   : min={s_vals[0]:.3f}  中位={statistics.median(s_vals):.3f}  max={s_vals[-1]:.3f}")
        print(f" 低饱和支 / 明亮支   : {sum(v < 0.35 for v in s_vals)} / {sum(v >= 0.35 for v in s_vals)}  (子标签配额参考)")
        print(f" 纹理能量 lap_var    : 中位={statistics.median(lap_vals):.1f}  (后续与生成图对比，防平涂)")
    print(f" 报告已写出          : {out.resolve()}")
    if cap_issues:
        print(" [!] caption 违规示例：")
        for r in cap_issues[:5]:
            print(f"     - {r['file']}: {r['flags']}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())

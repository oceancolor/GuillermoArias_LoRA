#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S0 · 从 configs/sources.csv 构建训练目录（主方案 §2.4 / §3.6）

做的事：
  1. 读取采集台账，只处理 decision=keep 的行（EXAMPLE- 开头的模板示例行自动跳过）
  2. 把原图复制到 dataset/train/10_mdtr/<id>.<ext>
  3. 生成同名 .txt caption 骨架（含 TODO，强制人工填写）
  4. 生成 dataset/dataset_manifest.csv（含 sha256，便于下架追溯）

安全约定：
  * 默认【不覆盖】已存在的 .txt —— 人工标注不会被重复运行冲掉；确需重置加 --force
  * 只复制，不移动、不删除任何原图
  * 不做上采样、不做格式转换（保留原始笔触细节）

仅依赖 Python 标准库。

用法：
  python scripts/init_dataset.py
  python scripts/init_dataset.py --sources configs/sources.csv --root . --force
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VALID_PALETTE = {"retro_muted", "candy_bright", "undecided"}
DEFAULT_TRIGGER = "mdtr_style"

CAPTION_SKELETON = (
    "{trigger} {sublabel}, TODO subject and pose, TODO objects and clothing, "
    "TODO setting and background, TODO light and composition"
)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k}
                for r in reader]
    return cols, rows


def resolve_image(row: dict, root: Path, raw_root: Path) -> Path | None:
    """先按 local_file 找，再回退到 raw_root/<id>.<ext>。"""
    raw = row.get("local_file", "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        if p.is_file() and p.suffix.lower() in EXTS:
            return p
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        p = raw_root / (row.get("id", "").strip() + ext)
        if p.is_file():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="S0：从采集台账构建训练目录")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent),
                    help="项目根目录（默认取本脚本的上两级）")
    ap.add_argument("--sources", default="configs/sources.csv", help="采集台账 CSV")
    ap.add_argument("--raw-root", default="dataset/raw", help="原图目录（local_file 缺失时回退）")
    ap.add_argument("--train-dir", default="dataset/train/10_mdtr", help="训练图目录")
    ap.add_argument("--manifest", default="dataset/dataset_manifest.csv", help="输出素材台账")
    ap.add_argument("--trigger", default=DEFAULT_TRIGGER, help="触发词")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 caption（会抹掉人工标注，慎用）")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    def rp(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else root / q

    sources = rp(args.sources)
    raw_root = rp(args.raw_root)
    train_dir = rp(args.train_dir)
    manifest = rp(args.manifest)

    if not sources.is_file():
        print(f"[!] 找不到采集台账：{sources}")
        return 1

    cols, rows = read_rows(sources)
    if "id" not in cols or "decision" not in cols:
        print(f"[!] 台账缺少必需列 id / decision。当前列：{cols}")
        return 1

    templates = [r for r in rows if r.get("id", "").upper().startswith("EXAMPLE-")]
    real = [r for r in rows if not r.get("id", "").upper().startswith("EXAMPLE-")]
    keep = [r for r in real if r.get("decision", "").lower() == "keep"]

    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir.parent / "1_reg").mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []
    copied = skeleton_new = skeleton_kept = missing = 0
    missing_ids: list[str] = []
    bad_palette: list[str] = []

    for r in keep:
        rid = r.get("id", "").strip()
        src = resolve_image(r, root, raw_root)
        rec = dict(r)
        if not src:
            missing += 1
            missing_ids.append(rid)
            rec.update({"train_file": "", "caption_file": "", "sha256_16": "",
                        "status": "MISSING_IMAGE"})
            out_rows.append(rec)
            continue

        dst = train_dir / (rid + src.suffix.lower())
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        copied += 1

        branch = r.get("palette_branch", "").strip().lower()
        if branch not in VALID_PALETTE:
            bad_palette.append(rid)
            branch = "undecided"
        sublabel = branch if branch != "undecided" else "TODO_palette"
        cap = train_dir / (rid + ".txt")
        if cap.exists() and not args.force:
            skeleton_kept += 1
        else:
            cap.write_text(
                CAPTION_SKELETON.format(trigger=args.trigger, sublabel=sublabel),
                encoding="utf-8")
            skeleton_new += 1

        rec.update({
            "train_file": str(dst.relative_to(root)).replace("\\", "/"),
            "caption_file": str(cap.relative_to(root)).replace("\\", "/"),
            "sha256_16": sha16(dst),
            "status": "OK",
        })
        out_rows.append(rec)

    if out_rows:
        fieldnames = [c for c in cols if c not in
                      {"train_file", "caption_file", "sha256_16", "status"}]
        fieldnames += ["train_file", "caption_file", "sha256_16", "status"]
        with manifest.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows)

    print("=" * 66)
    print(" S0 · 训练目录构建报告")
    print("=" * 66)
    print(f" 台账                : {sources}")
    print(f" 模板示例行(已跳过)  : {len(templates)}")
    print(f" 有效记录 / 入选     : {len(real)} / {len(keep)}")
    print(f" 复制图片            : {copied}")
    print(f" 新建 caption 骨架   : {skeleton_new}")
    print(f" 保留已有 caption    : {skeleton_kept}  (默认不覆盖人工标注)")
    print(f" 找不到原图          : {missing}")
    print(f" 配色支取值异常      : {len(bad_palette)}")
    print(f" 训练目录            : {train_dir}")
    print(f" 素材台账            : {manifest}")
    if missing_ids:
        print(f" [!] 缺图 id（前 8）: {', '.join(missing_ids[:8])}")
        print("     请按台账 local_file 把原图放到 dataset/raw/ 后重跑本脚本。")
    if bad_palette:
        print(f" [!] palette_branch 非法: {', '.join(bad_palette[:8])}")
    if skeleton_new:
        print(" [i] 下一步：按 configs/caption_rules_branchA.md 逐条把 .txt 里的 TODO 换成内容描述。")
        print("     任何残留 TODO 都会被 audit_manifest.py 拦下。")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())

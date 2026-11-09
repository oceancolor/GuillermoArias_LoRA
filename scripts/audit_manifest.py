#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S0 门禁 · 分支 A 数据集总验收（主方案 §2.4）

逐项判定，返回码：
  0 = 可进入 S1（允许 WARN）
  1 = 有 FAIL，禁止开训
  2 = 台账仍是模板状态 / 无法读取

检查项：
  必填字段与取值合法性、id 唯一、入选数、溯源率、题材桶配额、
  配色支 60:40、AI 高嫌疑、授权状态、图文一一对应、caption 违规、孤儿文件。

仅依赖 Python 标准库 + Pillow（风格词检查复用 audit_dataset.py，缺失时自动降级）。

用法：
  python scripts/audit_manifest.py
  python scripts/audit_manifest.py --sources configs/sources.csv --dataset dataset/train/10_mdtr
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_TRIGGER = "mdtr_style"

REQUIRED = [
    "id", "source_url", "collected_at", "reverse_image_checked",
    "license_status", "palette_branch", "subject_bucket",
    "ai_suspicion", "decision", "local_file",
]
ENUMS = {
    "reverse_image_checked": {"yes", "no"},
    "license_status": {"unknown", "personal_research_only", "authorized"},
    "palette_branch": {"retro_muted", "candy_bright", "undecided"},
    "subject_bucket": {"person_outdoor", "person_indoor", "landscape",
                       "still_life", "ood_anchor"},
    "ai_suspicion": {"none", "low", "medium", "high"},
    "decision": {"keep", "review", "drop"},
}
TARGET_BUCKETS = {"person_outdoor": 40, "person_indoor": 20,
                  "landscape": 20, "still_life": 10, "ood_anchor": 10}
TARGET_RETRO_PCT = 60.0
TOL_BUCKET_PP = 10.0
TOL_PALETTE_PP = 15.0
MIN_KEEP, IDEAL_LOW, IDEAL_HIGH, MAX_KEEP = 60, 90, 150, 200

# 复用 audit_dataset.py 的 caption 检查（含完整风格词黑名单）；缺 Pillow 时降级
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from audit_dataset import check_caption as _check_caption  # type: ignore
    HAS_FULL_CHECK = True
except Exception:  # noqa: BLE001
    HAS_FULL_CHECK = False

    def _check_caption(txt: Path, trigger: str) -> list[str]:
        if not txt.exists():
            return ["缺 caption 文件"]
        t = txt.read_text(encoding="utf-8", errors="replace").strip()
        issues = []
        if not t:
            return ["caption 为空"]
        if not t.lower().startswith(trigger.lower()):
            issues.append(f"触发词 {trigger} 未置于首位")
        if len(t.split()) < 8:
            issues.append("caption 过短，内容描述不足")
        return issues


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k}
                for r in reader]
    return cols, rows


def main() -> int:
    ap = argparse.ArgumentParser(description="S0 门禁：分支 A 数据集总验收")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent),
                    help="项目根目录")
    ap.add_argument("--sources", default="configs/sources.csv", help="采集台账")
    ap.add_argument("--dataset", default="dataset/train/10_mdtr", help="训练图目录")
    ap.add_argument("--out", default="s0_gate_report.txt", help="门禁报告输出")
    ap.add_argument("--trigger", default=DEFAULT_TRIGGER, help="触发词")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    def rp(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else root / q

    sources, dataset, report_path = rp(args.sources), rp(args.dataset), rp(args.out)

    if not sources.is_file():
        print(f"[!] 找不到采集台账：{sources}")
        return 2
    cols, rows = read_rows(sources)
    missing_cols = [c for c in REQUIRED if c not in cols]
    if missing_cols:
        print(f"[!] 台账缺列：{missing_cols}")
        return 2

    real = [r for r in rows if not r.get("id", "").upper().startswith("EXAMPLE-")]
    if not real:
        print("=" * 66)
        print(" S0 门禁：台账仍是模板状态")
        print("=" * 66)
        print(f" {sources} 中只有 EXAMPLE- 示例行，尚无真实素材记录。")
        print(" 请按主方案 §2.1–§2.3 采集并逐条填写，再运行本脚本。")
        print("=" * 66)
        return 2

    keep = [r for r in real if r.get("decision", "").lower() == "keep"]
    checks: list[tuple[str, str, str]] = []

    def add(name: str, level: str, detail: str) -> None:
        checks.append((name, level, detail))

    # 1 必填字段
    blanks = [(r.get("id", "?"), c) for r in real for c in REQUIRED
              if not r.get(c, "").strip()]
    add("必填字段", "FAIL" if blanks else "PASS",
        "全部完整" if not blanks else
        f"{len(blanks)} 处空缺，例如 " + ", ".join(f"{i}.{c}" for i, c in blanks[:5]))

    # 2 取值合法性
    bad_enum = [(r.get("id", "?"), c, r.get(c, "")) for r in real
                for c, allowed in ENUMS.items()
                if r.get(c, "").strip() and r[c].lower() not in allowed]
    add("字段取值", "FAIL" if bad_enum else "PASS",
        "合法" if not bad_enum else
        f"{len(bad_enum)} 处非法，例如 " + ", ".join(
            f"{i}.{c}={v}" for i, c, v in bad_enum[:5]))

    # 3 id 唯一
    ids = [r.get("id", "") for r in real]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    add("id 唯一", "FAIL" if dup else "PASS",
        "无重复" if not dup else "重复 id: " + ", ".join(dup[:5]))

    # 4 入选数量
    n = len(keep)
    if n < MIN_KEEP:
        lvl, msg = "FAIL", f"{n} 张，低于下限 {MIN_KEEP}"
    elif n > MAX_KEEP:
        lvl, msg = "WARN", f"{n} 张，超过上限 {MAX_KEEP}，收益递减且易引入噪声"
    elif IDEAL_LOW <= n <= IDEAL_HIGH:
        lvl, msg = "PASS", f"{n} 张，处于推荐区间 {IDEAL_LOW}–{IDEAL_HIGH}"
    else:
        lvl, msg = "PASS", f"{n} 张，可用（推荐 {IDEAL_LOW}–{IDEAL_HIGH}）"
    add("入选规模", lvl, msg)

    # 5 溯源率
    traced = sum(1 for r in keep
                 if r.get("reverse_image_checked", "").lower() == "yes"
                 and r.get("earliest_source", "").strip() not in {"", "未找到", "none", "n/a"})
    rate = traced / max(n, 1)
    if rate >= 0.80:
        lvl, msg = "PASS", f"{rate:.0%} ({traced}/{n})"
    elif rate >= 0.60:
        lvl, msg = "WARN", f"{rate:.0%} ({traced}/{n})，模型卡须写明『作者归属未确证』"
    else:
        lvl, msg = "FAIL", f"{rate:.0%} ({traced}/{n})，低于 60%，须回到采集"
    add("可溯源率", lvl, msg)

    # 6 题材桶配额
    if n:
        devs = []
        for b, target in TARGET_BUCKETS.items():
            actual = 100.0 * sum(1 for r in keep
                                 if r.get("subject_bucket", "").lower() == b) / n
            devs.append((b, target, actual, abs(actual - target)))
        worst = [d for d in devs if d[3] > TOL_BUCKET_PP]
        detail = " / ".join(f"{b} {a:.0f}%→{t}%" for b, t, a, _ in devs)
        add("题材桶配额", "FAIL" if worst else "PASS",
            detail + ("" if not worst else "  | 偏差 >10pp: " + ", ".join(d[0] for d in worst)))
    else:
        add("题材桶配额", "FAIL", "无入选图")

    # 7 配色支
    undec = [r.get("id", "?") for r in keep
             if r.get("palette_branch", "").lower() == "undecided"]
    retro = sum(1 for r in keep if r.get("palette_branch", "").lower() == "retro_muted")
    rpct = 100.0 * retro / max(n, 1)
    if undec:
        add("配色子标签", "FAIL",
            f"{len(undec)} 张仍为 undecided（{', '.join(undec[:5])}），必须先定性")
    elif abs(rpct - TARGET_RETRO_PCT) > TOL_PALETTE_PP:
        add("配色子标签", "FAIL",
            f"retro_muted 占 {rpct:.0f}%，偏离目标 {TARGET_RETRO_PCT:.0f}% 超过 {TOL_PALETTE_PP:.0f}pp")
    else:
        add("配色子标签", "PASS", f"retro_muted / candy_bright = {rpct:.0f}% / {100-rpct:.0f}%")

    # 8 AI 高嫌疑
    ai_bad = [r.get("id", "?") for r in real
              if r.get("ai_suspicion", "").lower() == "high"
              and r.get("decision", "").lower() != "drop"]
    add("AI 高嫌疑处置", "FAIL" if ai_bad else "PASS",
        "已全部 drop" if not ai_bad else "未 drop: " + ", ".join(ai_bad[:5]))

    # 9 授权状态
    known = sum(1 for r in keep if r.get("license_status", "").lower() != "unknown")
    add("授权状态", "WARN" if known == 0 and n else "PASS",
        f"{known}/{n} 已确权；其余 unknown" +
        ("。默认仅个人研究，不发布数据集/权重、不商用（§9）" if known < n else ""))

    # 10-12 文件与 caption
    ds_exists = dataset.is_dir()
    if not ds_exists:
        add("图文一致性", "FAIL", f"目录不存在：{dataset}，请先运行 init_dataset.py")
        add("caption 合规", "FAIL", "训练目录尚未建立")
        add("孤儿文件", "WARN", "训练目录尚未建立")
    else:
        imgs = {p.stem: p for p in dataset.iterdir() if p.suffix.lower() in EXTS}
        caps = {p.stem: p for p in dataset.iterdir() if p.suffix.lower() == ".txt"}
        keep_ids = [r.get("id", "") for r in keep]

        miss_img = [i for i in keep_ids if i not in imgs]
        miss_cap = [i for i in keep_ids if i not in caps]
        add("图文一致性",
            "FAIL" if (miss_img or miss_cap) else "PASS",
            "一一对应" if not (miss_img or miss_cap) else
            f"缺图 {len(miss_img)} / 缺 caption {len(miss_cap)}，例如 " +
            ", ".join((miss_img + miss_cap)[:5]))

        cap_bad: list[str] = []
        for i in keep_ids:
            if i not in caps:
                continue
            issues = list(_check_caption(caps[i], args.trigger))
            text = caps[i].read_text(encoding="utf-8", errors="replace")
            if "TODO" in text:
                issues.append("残留 TODO（caption 未填完）")
            if issues:
                cap_bad.append(f"{i}: " + "; ".join(issues[:2]))
        add("caption 合规", "FAIL" if cap_bad else "PASS",
            "全部合规" if not cap_bad else
            f"{len(cap_bad)} 条需修正： " + " | ".join(cap_bad[:3]) +
            ("  [风格词检查已降级：缺 Pillow]" if not HAS_FULL_CHECK else ""))

        orphans = sorted(set(imgs) - set(keep_ids))
        add("孤儿文件", "WARN" if orphans else "PASS",
            "无" if not orphans else
            f"{len(orphans)} 张不在台账 keep 列表：{', '.join(orphans[:5])}")

    # 汇总
    n_fail = sum(1 for _, l, _ in checks if l == "FAIL")
    n_warn = sum(1 for _, l, _ in checks if l == "WARN")
    verdict = "禁止开训（存在 FAIL）" if n_fail else "通过，可进入 S1"

    lines = ["=" * 70, " S0 门禁报告 · 分支 A / style corpus LoRA", "=" * 70,
             f" 台账      : {sources}", f" 训练目录  : {dataset}",
             f" 记录/入选 : {len(real)} / {len(keep)}", "-" * 70]
    for name, lvl, detail in checks:
        lines.append(f" [{lvl:<4}] {name:<14} {detail}")
    lines += ["-" * 70,
              f" 结论：{verdict}   (FAIL {n_fail} / WARN {n_warn} / PASS "
              f"{len(checks)-n_fail-n_warn})", "=" * 70]

    text = "\n".join(lines)
    print(text)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    print(f" 报告已写出：{report_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())

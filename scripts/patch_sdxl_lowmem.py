#!/usr/bin/env python
"""Idempotent low-RAM patch for kohya sd-scripts SDXL loading (Windows 16 GB hosts).

Problem
-------
In ``library/sdxl_model_util.py`` the U-Net and VAE receive ``dtype`` when
loading from an SDXL checkpoint, but the two text encoders do not::

    info1 = _load_state_dict_on_device(text_model1, te1_sd, device=map_location)
    info2 = _load_state_dict_on_device(text_model2, converted_sd, device=map_location)

``set_module_tensor_to_device`` (accelerate/utils/modeling.py:314) then calls
``value.to(old_value.dtype)`` for EVERY fp16 checkpoint tensor, because
``init_empty_weights()`` built the modules in fp32. TE2 alone (~1.4B params)
materialises a second fp32 copy. On a 16 GB host this is the difference between
loading and crashing with::

    RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate 6553600 bytes

``--full_bf16`` fixes U-Net/VAE but NOT the text encoders. Then
``sdxl_train_network.py`` casts TEs back to fp32 after caching their outputs,
creating the copies a second time.

Patch
-----
1. Pass ``dtype`` to both TE load calls. ``dtype`` is ``None`` unless
   ``--full_bf16`` / ``--full_fp16`` is used, so behaviour is unchanged for
   normal training; under full_bf16 the TEs are loaded in bf16 directly.
2. After text-encoder-output caching, move TEs back to CPU keeping bf16.
   fp16 still falls back to fp32 (fp16 CLIP does not work on CPU); bf16 is fine
   as storage because the TEs are never run on CPU once outputs are cached.
3. Skip the pointless TE/U-Net/VAE GPU round-trip when every text-encoder
   output is already cached on disk. ``cache_text_encoder_outputs_if_needed``
   moves both TEs (and U-Net/VAE) to GPU and back UNCONDITIONALLY, even when
   ``new_cache_text_encoder_outputs`` would find 100% disk hits. On an 8 GB
   WDDM host that round-trip alone can OOM at ``text_encoders[1].to(cuda)``.
   The patch binds each ImageInfo's ``text_encoder_outputs_npz`` path (which
   is what the training loop needs) and returns early. If ANY cache is missing
   or invalid, the original code path runs unchanged.

Usage
-----
    python scripts/patch_sdxl_lowmem.py --check
    python scripts/patch_sdxl_lowmem.py --apply
    python scripts/patch_sdxl_lowmem.py --revert
    python scripts/patch_sdxl_lowmem.py --apply --sd-scripts /path/to/sd-scripts

Backups are written next to each file as ``<name>.orig`` and are never
overwritten. Reverting restores those backups.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_SD_SCRIPTS = Path("E:/lora_local/sd-scripts")

MARKER = "# patched by patch_sdxl_lowmem.py"

# (relative path, [(old, new), ...])
PATCHES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "library/sdxl_model_util.py",
        [
            (
                "    info1 = _load_state_dict_on_device(text_model1, te1_sd, device=map_location)  # remain fp32",
                "    info1 = _load_state_dict_on_device(text_model1, te1_sd, device=map_location, dtype=dtype)  "
                + MARKER,
            ),
            (
                "    info2 = _load_state_dict_on_device(text_model2, converted_sd, device=map_location)  # remain fp32",
                "    info2 = _load_state_dict_on_device(text_model2, converted_sd, device=map_location, dtype=dtype)  "
                + MARKER,
            ),
        ],
    ),
    (
        "sdxl_train_network.py",
        [
            (
                '            text_encoders[0].to("cpu", dtype=torch.float32)  '
                "# Text Encoder doesn't work with fp16 on CPU\n"
                '            text_encoders[1].to("cpu", dtype=torch.float32)',
                "            te_cpu_dtype = torch.float32 if weight_dtype == torch.float16 else weight_dtype  "
                + MARKER
                + "\n"
                '            text_encoders[0].to("cpu", dtype=te_cpu_dtype)\n'
                '            text_encoders[1].to("cpu", dtype=te_cpu_dtype)',
            ),
            (
                "        if args.cache_text_encoder_outputs:\n"
                "            if not args.lowram:",
                "        if args.cache_text_encoder_outputs:\n"
                "            # [lowmem patch] Bind disk-cached TE output paths and skip the\n"
                "            # whole GPU round-trip when every cache file is present+valid.\n"
                "            if args.cache_text_encoder_outputs_to_disk:\n"
                "                _te_strategy = strategy_base.TextEncoderOutputsCachingStrategy.get_strategy()\n"
                "                if (\n"
                "                    _te_strategy is not None\n"
                "                    and getattr(_te_strategy, \"cache_to_disk\", False)\n"
                "                    and not getattr(_te_strategy, \"is_weighted\", False)\n"
                "                ):\n"
                "                    _infos = list(getattr(dataset, \"image_data\", {}).values())\n"
                "                    _npz_paths = []\n"
                "                    for _info in _infos:\n"
                "                        _npz = _te_strategy.get_outputs_npz_path(_info.absolute_path)\n"
                "                        if not _te_strategy.is_disk_cached_outputs_expected(_npz):\n"
                "                            _npz_paths = None\n"
                "                            break\n"
                "                        _npz_paths.append(_npz)\n"
                "                    if _infos and _npz_paths is not None:\n"
                "                        for _info, _npz in zip(_infos, _npz_paths):\n"
                "                            _info.text_encoder_outputs_npz = _npz\n"
                "                        logger.info(\n"
                "                            f\"all {len(_infos)} text encoder outputs found on disk; \"\n"
                "                            \"skip TE/U-Net/VAE GPU round-trip\"  " + MARKER + "\n"
                "                        )\n"
                "                        return\n"
                "            if not args.lowram:",
            ),
        ],
    ),
]


def backup_path(path: Path) -> Path:
    return path.with_name(path.name + ".orig")


def ensure_backup(path: Path) -> None:
    dst = backup_path(path)
    if not dst.exists():
        shutil.copy2(path, dst)
        print(f"  backup -> {dst}")


def status(root: Path) -> bool:
    """Return True when every hunk is applied."""
    applied = True
    for rel, hunks in PATCHES:
        path = root / rel
        if not path.is_file():
            print(f"  MISSING {rel}")
            return False
        text = path.read_text(encoding="utf-8")
        for old, new in hunks:
            has_new = new in text
            has_old = old in text
            state = "applied" if has_new else ("pending" if has_old else "NOT FOUND")
            if not has_new:
                applied = False
            print(f"  {state:9s} {rel}: {old.strip().splitlines()[0][:60]}")
    return applied


def apply(root: Path) -> int:
    for rel, hunks in PATCHES:
        path = root / rel
        if not path.is_file():
            print(f"[ERROR] file not found: {path}")
            return 2
        text = path.read_text(encoding="utf-8")
        if all(new in text for _, new in hunks):
            print(f"  already patched: {rel}")
            continue
        for old, _ in hunks:
            if old not in text and MARKER not in text:
                print(f"[ERROR] expected line not found in {rel}:\n  {old}")
                print("        sd-scripts version differs; patch manually before continuing.")
                return 2
        ensure_backup(path)
        for old, new in hunks:
            if new not in text:
                text = text.replace(old, new, 1)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"  patched: {rel}")
    return 0


def revert(root: Path) -> int:
    for rel, _ in PATCHES:
        path = root / rel
        bak = backup_path(path)
        if not bak.is_file():
            print(f"  no backup for {rel}, skipping")
            continue
        shutil.copy2(bak, path)
        print(f"  reverted: {rel}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="report patch status only")
    g.add_argument("--apply", action="store_true", help="apply the patch (idempotent)")
    g.add_argument("--revert", action="store_true", help="restore .orig backups")
    ap.add_argument("--sd-scripts", type=Path, default=DEFAULT_SD_SCRIPTS)
    args = ap.parse_args()

    root = args.sd_scripts
    if not root.is_dir():
        print(f"[ERROR] sd-scripts directory not found: {root}")
        return 2

    print(f"sd-scripts: {root}")
    if args.check:
        return 0 if status(root) else 1
    if args.apply:
        rc = apply(root)
        if rc == 0:
            print("--- verification ---")
            return 0 if status(root) else 1
        return rc
    return revert(root)


if __name__ == "__main__":
    sys.exit(main())

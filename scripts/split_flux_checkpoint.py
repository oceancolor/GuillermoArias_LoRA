"""把 ComfyUI 一体化 FLUX.1-dev checkpoint 拆成 sd-scripts 需要的独立权重文件。

背景
----
`Comfy-Org/flux1-dev` 的 `flux1-dev-fp8.safetensors`（17.25 GB）把四个组件
打包在一个文件里，键前缀分别是：

    model.diffusion_model.*   780 个张量  -> DiT（纯 BFL 格式）
    text_encoders.clip_l.*    198 个张量
    text_encoders.t5xxl.*     220 个张量
    vae.*                     244 个张量  -> AE

`flux_train_network.py` 要求 4 个独立文件路径，且 `analyze_checkpoint_state()`
只看 keys[0] 判断前缀，直接喂一体化文件会解析失败。本脚本按前缀流式复制，
去掉前缀后写成独立的 safetensors 文件。

已验证（2026-11-09，本机 sd-scripts main）：
    去前缀后 DiT 780/780、AE 244/244 与 flux_models.Flux / AutoEncoder 的
    state_dict 键完全一致，无 missing、无 unexpected。

实现要点
--------
只重写 safetensors 的 header 并按字节搬运数据区，不把张量反序列化进内存。
峰值内存约 8 MB，16 GB 内存机器也能跑；不需要 torch / safetensors 依赖。

用法
----
    python scripts/split_flux_checkpoint.py \
        --src E:/lora_local/models/flux1-dev-fp8.safetensors \
        --out-dir E:/lora_local/models \
        --parts dit,ae

    python scripts/split_flux_checkpoint.py --selftest   # 用小型合成文件自检
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

CHUNK = 8 * 1024 * 1024

# 输出文件名 -> (源键前缀, 要去掉的前缀)
PARTS = {
    "dit": ("model.diffusion_model.", "model.diffusion_model."),
    "ae": ("vae.", "vae."),
    "clip_l": ("text_encoders.clip_l.", "text_encoders.clip_l."),
    "t5xxl": ("text_encoders.t5xxl.", "text_encoders.t5xxl."),
}
OUT_NAMES = {
    "dit": "flux1-dev-fp8-dit.safetensors",
    "ae": "ae.safetensors",
    "clip_l": "clip_l_from_bundle.safetensors",
    "t5xxl": "t5xxl_from_bundle.safetensors",
}

# safetensors dtype 名称 -> 每个元素字节数（用于自检时构造数据）
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def read_header(src: Path) -> tuple[dict, int]:
    """返回 (header dict, 数据区在源文件中的绝对起始偏移)。"""
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def build_output_header(src_header: dict, mapping: list[tuple[str, str]]) -> tuple[bytes, list[tuple[str, tuple[int, int]]]]:
    """按输出顺序重新计算 data_offsets，返回 (已补齐的 header 字节, 源偏移表)。"""
    out: dict = {}
    offsets: list[tuple[str, tuple[int, int]]] = []
    cursor = 0
    for new_key, src_key in mapping:
        info = src_header[src_key]
        start, end = info["data_offsets"]
        size = end - start
        if size <= 0:
            raise ValueError(f"invalid data_offsets for {src_key}: {info['data_offsets']}")
        out[new_key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        offsets.append((new_key, (start, end)))
        cursor += size

    raw = json.dumps(out, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(raw) % 8) % 8
    raw += b" " * pad  # safetensors 要求 header 按 8 字节对齐
    return raw, offsets


def split_one(src: Path, dst: Path, part: str, force: bool) -> int:
    prefix, strip = PARTS[part]
    src_header, src_data_start = read_header(src)
    mapping = [
        (k[len(strip):], k)
        for k in src_header
        if k != "__metadata__" and k.startswith(prefix)
    ]
    if not mapping:
        raise RuntimeError(f"no keys with prefix '{prefix}' in {src} — wrong checkpoint format?")

    if dst.exists() and not force:
        print(f"[skip] {dst} already exists (use --force to overwrite)")
        return 0

    header_bytes, offsets = build_output_header(src_header, mapping)
    total = sum(e - s for _, (s, e) in offsets)
    print(f"[{part}] {len(mapping)} tensors, {total / 1024**3:.2f} GB -> {dst}")

    tmp = dst.with_suffix(dst.suffix + ".partial")
    with open(src, "rb") as sf, open(tmp, "wb") as df:
        df.write(struct.pack("<Q", len(header_bytes)))
        df.write(header_bytes)
        written = 0
        for _, (s, e) in offsets:
            sf.seek(src_data_start + s)
            remaining = e - s
            while remaining > 0:
                chunk = sf.read(min(CHUNK, remaining))
                if not chunk:
                    raise RuntimeError("unexpected EOF while copying tensor data")
                df.write(chunk)
                remaining -= len(chunk)
                written += len(chunk)
        if written != total:
            raise RuntimeError(f"size mismatch: wrote {written}, expected {total}")
    tmp.replace(dst)
    print(f"[{part}] OK")
    return len(mapping)


def selftest() -> int:
    """构造一个小型一体化 safetensors，拆分后用 safetensors 库回读校验。"""
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        import torch
    except ImportError:
        print("[selftest] safetensors/torch not available - skipped")
        return 0

    td = Path(tempfile.mkdtemp(prefix="flux_split_test_"))
    try:
        bundle = td / "bundle.safetensors"
        sd = {
            "model.diffusion_model.double_blocks.0.img_attn.proj.weight": torch.arange(16, dtype=torch.float16).reshape(4, 4),
            "model.diffusion_model.double_blocks.0.img_attn.proj.bias": torch.zeros(4, dtype=torch.float16),
            "vae.decoder.conv_in.weight": torch.ones(8, dtype=torch.float32),
            "text_encoders.clip_l.logit_scale": torch.ones(2, dtype=torch.float32),
        }
        save_file(sd, str(bundle))

        n_dit = split_one(bundle, td / "dit.safetensors", "dit", True)
        n_ae = split_one(bundle, td / "ae.safetensors", "ae", True)
        assert n_dit == 2 and n_ae == 1, (n_dit, n_ae)

        with safe_open(td / "dit.safetensors", framework="pt") as f:
            keys = set(f.keys())
            assert keys == {"double_blocks.0.img_attn.proj.weight", "double_blocks.0.img_attn.proj.bias"}, keys
            w = f.get_tensor("double_blocks.0.img_attn.proj.weight")
            assert torch.equal(w, sd["model.diffusion_model.double_blocks.0.img_attn.proj.weight"])
        with safe_open(td / "ae.safetensors", framework="pt") as f:
            assert set(f.keys()) == {"decoder.conv_in.weight"}
            assert torch.equal(f.get_tensor("decoder.conv_in.weight"), sd["vae.decoder.conv_in.weight"])
        del f
        gc.collect()  # Windows: release mmap handles before deleting temp files
    finally:
        gc.collect()
        shutil.rmtree(td, ignore_errors=True)
    print("[selftest] PASS - header rewrite + byte copy verified")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, help="一体化 checkpoint 路径")
    ap.add_argument("--out-dir", type=Path, help="输出目录")
    ap.add_argument("--parts", default="dit,ae", help="要拆出的组件，逗号分隔：dit,ae,clip_l,t5xxl")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出文件")
    ap.add_argument("--selftest", action="store_true", help="用合成文件自检，不读取真实模型")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.src or not args.out_dir:
        ap.error("--src and --out-dir are required (or use --selftest)")
    if not args.src.is_file():
        print(f"[ERROR] source not found: {args.src}", file=sys.stderr)
        return 1

    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        print(f"[ERROR] unknown parts: {unknown}; choose from {list(PARTS)}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        split_one(args.src, args.out_dir / OUT_NAMES[part], part, args.force)
    print("[done] split complete")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

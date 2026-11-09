# FLUX LoRA 训练补充方案（分支 A 风格谱系）

> 版本 v1.0 · 2026-11-09 · 配套主方案 `GuillermoArias_风格LoRA训练方案.md`
> 适用对象不变：**分支 A**（地中海 / 海边阅读 / 厚涂肌理 / 复古低饱和），数据集与 caption 规则完全复用。
> 合规叠加：分支 A 归属未确证 + **FLUX.1 [dev] 非商业许可** → 默认仅个人学习与风格研究。

---

## 1. 结论速览

| 路径 | 可行性 | 前置条件 | 定位 |
| --- | --- | --- | --- |
| **A. H20 96GB 正式 FLUX.1-dev 对照** | ✅ 推荐 | 4 个模型文件 + S0 数据集 | 主方案里预留的 FLUX 对照实验，脚本已就绪 |
| **B. 本机 8GB 探测** | ⚠️ 有条件 | **先扩容虚拟内存 / 物理内存** + 下载约 22.4 GB + 拆分 | 只做可行性验证，不作为产出主力 |
| **C. 本机现有 `flux-2-klein-4b-nvfp4`** | ❌ 放弃 | — | sd-scripts 当前 main 无 FLUX.2 trainer；NVFP4 是推理量化格式 |

**一句话**：FLUX 在 H20 上今天就能排期；在本机上，卡点**不是显存而是 16 GB 系统内存**，而且这个卡点目前连 SDXL 都还没完全跨过去。

---

## 2. 本机真正的硬门：CPU 内存 / 提交上限

### 2.1 2026-11-09 实测链路

| 尝试 | 结果 |
| --- | --- |
| `--disable_mmap_load_safetensors`（bytes 模式） | U-Net、TE1 加载成功，TE2 阶段 `0xC0000005`。该开关在 sd-scripts 里走 `safetensors.torch.load(open(path,"rb").read())`，峰值同时持有约 7 GB 字节流 + 7 GB 张量，实测物理内存只剩 **1.02 GB**、页面文件可用 **5.98 GB** |
| 改用 mmap（去掉该开关） | 崩溃点前移，但换来明确错误：`RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate 6553600 bytes` |
| 隔离诊断 `diag_te2.py` | **两种模式都 `DIAG_OK`**：TE2 键名转换 517 键、missing=0 / unexpected=0、权重注入成功 → 排除 transformers 5.20 ABI、排除键名不匹配 |
| 定位根因 | `accelerate/utils/modeling.py:314`：`init_empty_weights()` 建的 U-Net 是 **fp32**，底模是 **fp16**，`dtype=None` 时逐权重 `value.to(torch.float32)`，把 2.6B 参数从 5 GB 复制成约 10 GB |
| 加 `--full_bf16` | ✅ U-Net / TE1 / TE2 / VAE **四个模型全部 `All keys matched successfully`**（第一次越过 TE2）<br>❌ 但紧接着 `vae.to(cuda)` 处 `CUDA out of memory`（申请 20 MiB，报告 6.57 GiB free），**尚未进入训练步，实测 `s/it` 仍未拿到** |

最后那个 CUDA OOM 与 CPU 内存是同一条链：WDDM 下 CUDA 分配需要系统侧可提交内存，进程常驻约 11 GB 后提交量逼近上限，驱动侧分配失败并被 PyTorch 报成 OOM。

### 2.2 内存账本

| 项 | 实测值 |
| --- | --- |
| 物理内存 | 16.22 GB（空闲约 7.3 GB） |
| 页面文件 | 原 `D:\pagefile.sys` 19.4 GB（D 盘仅剩约 11 GB）；**已改为 `E:\pagefile.sys` 32–48 GB，重启后生效** |
| 提交上限 | 34.77 GB（空载已提交 20.39 GB） |
| E 盘可用 | 约 78 GB |
| FLUX fp8 需求 | DiT 约 12 GB + T5-XXL fp8 约 5 GB + `blocks_to_swap 28` 常驻 CPU 的 block + 激活 → **提交上限建议 ≥ 48 GB** |

### 2.3 处置（按代价排序，需用户授权后执行）

> **2026-11-09 已按用户授权完成配置**：关闭「自动管理分页文件」，把页面文件设为 `E:\pagefile.sys`，初始 32 GB / 上限 48 GB；原配置（`?:\pagefile.sys` / AutoManaged=True）已备份到 `E:/lora_local/diag/pagefile_backup.json`。注册表与 WMI 均已确认写入，**但提交上限当前仍显示 34.77 GB —— 必须重启 Windows 才生效**，本次未擅自重启。
>
> 重启后预期提交上限约 64–80 GB，满足本机 SDXL 与 FLUX 探测需求。回滚方式：把 `AutomaticManagedPagefile` 设回 `True`，并将 `PagingFiles` 还原为 `?:\pagefile.sys`。

1. **把页面文件迁到 E 盘并增大到 32–48 GB**：E 盘有 78 GB，最有效；需管理员权限，可能重启。
2. **关闭占内存程序**：Steam、`WeChatAppEx`、Edge WebView、GIGABYTE Control Center 等，预计释放 1–2 GB；WorkBuddy 自身无法关闭。
3. **加装物理内存到 32 GB**：一劳永逸。
4. 上述任一完成后，先重跑 `FULL_BF16=1 BENCH=1 ...` 让 SDXL 拿到实测 `s/it`，再考虑 FLUX。

> `scripts/train_local_flux.sh` 已内建提交上限预检（< 48 GB 直接阻断），不会让你白跑几小时。

---

## 3. 模型获取：一条不需要 HF 授权的路

### 3.1 官方来源被 gated

sd-scripts `docs/flux_train_network.md` §3 指定：DiT 与 AE 取 `black-forest-labs/FLUX.1-dev` 的 `flux1-dev.safetensors` + `ae.safetensors`（**子文件夹里的 Diffusers 格式不能用**），T5-XXL 用 `comfyanonymous/flux_text_encoders/t5xxl_fp16.safetensors`。本机经 hf-mirror 实测 **BFL 两个文件均返回 403**。

### 3.2 非 gated 替代方案（已验证）

| 文件 | 源 | 大小 | 状态 |
| --- | --- | --- | --- |
| `flux1-dev-fp8.safetensors` | `Comfy-Org/flux1-dev` | 17.25 GB | 200 OK |
| `clip_l.safetensors` | `comfyanonymous/flux_text_encoders` | 246 MB | 200 OK |
| `t5xxl_fp8_e4m3fn.safetensors` | `comfyanonymous/flux_text_encoders` | 4.89 GB | 200 OK |

一体化文件内部是 1442 个张量：`model.diffusion_model.*`(780) + `text_encoders.clip_l.*`(198) + `text_encoders.t5xxl.*`(220) + `vae.*`(244)。`flux_train_network.py` 要求 4 个独立文件，且 `analyze_checkpoint_state()` 只看 `keys[0]` 判断前缀，**直接喂会解析失败**。

`scripts/split_flux_checkpoint.py` 按前缀流式重写 safetensors header、按字节搬运数据区（峰值内存 8 MB），产出：

- `flux1-dev-fp8-dit.safetensors`（纯 BFL 格式 DiT，约 12 GB）
- `ae.safetensors`（约 335 MB）

**键名严格校验结果**：去前缀后 DiT **780/780**、AE **244/244** 与 `flux_models.Flux` / `AutoEncoder` 的 `state_dict` 完全一致，无 missing、无 unexpected。脚本 `--selftest` 已通过（含 Windows mmap 句柄清理修复）。

> 合规提醒：绕开 gated 只改变获取方式，**不改变 FLUX.1 [dev] 非商业许可**。

### 3.3 一键命令

```bash
bash scripts/fetch_flux_models.sh          # 下载 + 拆分（支持断点续传）
# 若已手动放好文件：
SKIP_DOWNLOAD=1 bash scripts/fetch_flux_models.sh
```

---

## 4. 路径 A：H20 正式 FLUX 对照（已选为主线）

### 4.1 在 H20 上一次性准备

`scripts/setup_flux_h20.sh` 负责环境体检、拉取 sd-scripts、准备 4 个模型文件。**官方权重是 gated**，需先到 `black-forest-labs/FLUX.1-dev` 页面接受非商业许可并生成 HF token：

```bash
# 官方 fp16（推荐，质量最好；约 34 GB）
HF_TOKEN=hf_xxx bash scripts/setup_flux_h20.sh

# 暂时拿不到 token 时：非 gated fp8 降级（约 22.4 GB，需另加 --fp8_base）
MODE=fp8 bash scripts/setup_flux_h20.sh

# 模型已手动放好，只装 sd-scripts
SKIP_MODELS=1 bash scripts/setup_flux_h20.sh
```

脚本会把模型放到 `models/`（对应 `train_flux.sh` 的默认 `MODEL_DIR`），并打印后续三条命令。H20 上 `SD_SCRIPTS` / `MODEL_DIR` 均可用环境变量覆盖。

### 4.2 训练

```bash
SD_SCRIPTS=/path/sd-scripts MODEL_DIR=/path/models DRY_RUN=1 bash scripts/train_flux.sh   # 检查参数
SD_SCRIPTS=/path/sd-scripts MODEL_DIR=/path/models BENCH=1  bash scripts/train_flux.sh   # 200 步测 s/it
SD_SCRIPTS=/path/sd-scripts MODEL_DIR=/path/models bash scripts/train_flux.sh            # 正式 2000 步
```

`scripts/train_flux.sh` 本轮修正：

- **补上缺失的 `--cache_latents`**（sd-scripts 不会因 `--cache_latents_to_disk` 自动开启；缺它时 H20 每步都要重跑 AE，白烧机时）；
- 新增 4 个模型文件 + `flux_train_network.py` 存在性预检，避免启动几十秒后才报错；
- 新增 `DRY_RUN`、`BENCH` 独立输出目录、`SD_SCRIPTS` / `MODEL_DIR` 覆盖。

关键参数（96 GB 显存富余，不做任何量化 / offload）：`batch 2` · `dim 16` · `AdamW8bit` · `bf16` · `SDPA` · `--highvram` · 全量 latent / TE 缓存 · flow-matching 三件套 `flux_shift / raw / guidance_scale=1.0`。

H20 上建议用**官方 fp16 权重**（`flux1-dev.safetensors` + `ae.safetensors` + `t5xxl_fp16.safetensors`），不加 `--fp8_base`，质量优于 fp8。

---

## 5. 路径 B：本机 8GB 探测配方

官方 `docs/flux_train_network.md` §6.1 给出的 8 GB 档：`--fp8_base` + `--blocks_to_swap 28` + fp8 T5-XXL + 缓存 + 优化器裁剪。`scripts/train_local_flux.sh` 已落地：

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `--fp8_base` | 开 | fp8 底座，显存与内存都省 |
| `--blocks_to_swap` | 28 | 官方 8 GB 建议值（最大 35），越大越省显存但越慢 |
| `--gradient_checkpointing` | 开 | 必开；**不能**与 `--cpu_offload_checkpointing` 同用（源码有断言） |
| `--t5xxl_max_token_length` | 256 | 默认 512，减半省内存 |
| `resolution` | 768 | 1024 会让激活显存接近翻倍 |
| `network_dim / batch` | 8 / 1 | 8 GB 下限 |
| 优化器 | Adafactor | 省状态显存 |
| 缓存 | latent + TE 全部落盘 | 复用 `E:/lora_local/cache` |
| 禁止 | `--highvram` / `--lowram` / `--cpu_offload_checkpointing` | 前两者分别属于 H20 与显存富余机器；后者与 block swap 互斥 |

```bash
DRY_RUN=1 bash scripts/train_local_flux.sh    # 只打印命令
BENCH=1  bash scripts/train_local_flux.sh    # 200 步 s/it 探测（合成图集）
```

> **现实预期**：`blocks_to_swap 28` 意味着每个 step 都有大量 CPU↔GPU 权重交换，本机 `s/it` 会显著高于 SDXL；叠加 16 GB 内存约束，10 h 窗口内很可能只能完成几百步。它适合回答「能不能跑」，不适合产出正式权重。**正式 FLUX 权重仍应走 H20。**

---

## 6. 与分支 A 主线的关系

| 项 | 是否复用 |
| --- | --- |
| 数据集 / S0 门禁 | ✅ 完全复用 `configs/sources.csv`、`init_dataset.py`、`audit_manifest.py` |
| caption 规则 | ✅ `mdtr_style` 首位、只写客观内容、`retro_muted` / `candy_bright` 子标签、风格词黑名单 |
| 触发词与验证 prompt 网格 | ✅ `configs/val_prompts.txt` |
| 评测 | ✅ `eval_checkpoints.py` + 固定 prompt 盲评；新增 FLUX vs SDXL 同 prompt 横向对比 |
| 新增配置 | `configs/dataset_flux.toml`（H20 1024px / batch2）、`configs/dataset_local_flux.toml`（本机 768px / batch1）、`configs/dataset_local_flux_bench.toml`（合成探测） |

**S0 仍是唯一硬门**：`sources.csv` 目前仍为 8 行 `EXAMPLE-`、0 条真实记录，`dataset/` 未建立。所有 FLUX 命令在真实数据到位前都只能跑合成 benchmark。

---

## 7. 合规边界（叠加后更严）

1. 分支 A 作者归属未确证 → 按「风格谱系」处理，不声称复刻某位在世艺术家。
2. FLUX.1 [dev] 权重为**非商业许可**；FLUX.1-schnell / FLUX.2 klein 才是可商用变体。
3. 默认边界：仅个人学习与风格研究；不发布数据集、不发布权重、不商用、不代客出图、不冒名。
4. 若未来有商用打算，需改用可商用底座，并重新核实训练图集的版权归属。

---

## 8. 交付物索引

| 文件 | 作用 |
| --- | --- |
| `FLUX_LoRA训练方案.md` | 本文档 |
| `scripts/train_flux.sh` | H20 FLUX.1-dev 正式训练（已修 `--cache_latents`、加预检 / DRY_RUN / BENCH） |
| `scripts/train_local_flux.sh` | 本机 8 GB 探测（内建提交上限 / 模型 / 显存预检） |
| `scripts/fetch_flux_models.sh` | 非 gated 三件套下载 + 拆分，断点续传（本机探测用） |
| **`scripts/setup_flux_h20.sh`** | **H20 一键准备**：环境体检、sd-scripts、官方 gated 权重下载 / fp8 降级方案 |
| `scripts/split_flux_checkpoint.py` | 一体化 checkpoint → 纯 DiT + AE，低内存流式，含 `--selftest` |
| `configs/dataset_flux.toml` | H20 1024px / batch 2 |
| `configs/dataset_local_flux.toml` | 本机 768px / batch 1 |
| `configs/dataset_local_flux_bench.toml` | 本机合成图集 200 步探测 |
| `E:/lora_local/diag/diag_te2.py` | TE2 隔离诊断脚本（非项目交付物，排错用） |

---

## 9. 决策状态与待办

**2026-11-09 用户已决策**：FLUX 主线走 **路径 A（H20 正式对照）**；**授权修改页面文件**。

| # | 事项 | 状态 |
| --- | --- | --- |
| 1 | FLUX 主线路径 | ✅ 已定：H20 正式对照；`scripts/setup_flux_h20.sh` 已产出 |
| 2 | 页面文件迁移 | ✅ 已配置为 `E:\pagefile.sys` 32–48 GB，备份已留；**待用户重启后生效** |
| 3 | 关闭占内存程序 | ⬜ 未执行（用户本次未授权，也未擅自结束任何进程） |
| 4 | 本机下载 22.4 GB 模型 | ⬜ 未执行（主线改走 H20 后不再必要） |
| 5 | S0 数据采集 | ⬜ **仍是正式训练的唯一硬门** |
| 6 | 确认 H20 机时性质（内部配额 / 按小时付费）与用途边界 | ⬜ 待用户确认 |
| 7 | 提供 H20 可用的 `HF_TOKEN`（BFL gated） | ⬜ 待用户准备；或改用 `MODE=fp8` 降级 |
| 8 | 重启 Windows 使页面文件生效，然后重跑 `FULL_BF16=1 BENCH=1 ...` | ⬜ 待用户执行（本次未擅自重启） |

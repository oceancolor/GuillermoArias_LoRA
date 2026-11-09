# FLUX LoRA 训练补充方案（分支 A 风格谱系）

> 版本 v1.1 · 2026-11-10 · 配套主方案 `GuillermoArias_风格LoRA训练方案.md`
> 适用对象不变：**分支 A**（地中海 / 海边阅读 / 厚涂肌理 / 复古低饱和），数据集与 caption 规则完全复用。
> 合规叠加：分支 A 归属未确证 + FLUX.1 [dev] 非商业许可 → 默认仅个人学习与风格研究。
> **v1.1 更正**：新增路径 D。上一版「本机 8GB 不能训 FLUX」只适用于 FLUX.1-dev 12B + sd-scripts 组合，不是对 FLUX 全族的判决，见 §1.1。

---

## 1. 结论速览

| 路径 | 可行性 | 前置条件 | 定位 |
| --- | --- | --- | --- |
| **A. H20 96GB 正式 FLUX.1-dev 对照** | ✅ 推荐 | 4 个模型文件 + S0 数据集 | 主方案里预留的 FLUX 对照实验，脚本已就绪 |
| **B. 本机 8GB + sd-scripts + FLUX.1-dev 12B** | ⚠️ 仅限探测 | **先扩容虚拟内存 / 物理内存** + 下载约 22.4 GB + 拆分 | 内存墙未拆；`blocks_to_swap 28` 每步换权重，只回答「能不能跑」 |
| **C. 本机现有 `flux-2-klein-4b-nvfp4`（distilled）** | ❌ 放弃 | — | sd-scripts 当前 main 无 FLUX.2 trainer；NVFP4 是推理量化格式，且蒸馏版不是训练目标 |
| **D. 本机 8GB + ai-toolkit + FLUX.2 klein 4B Base（v1.1 新增）** | ✅ **可行，未实测** | 下载约 16 GB + 人工装依赖 + S0 数据集 | 4B 底座 fp8 后约 4 GB，Windows 8GB 有公开成功案例；约 5 s/it、2000 步约 3 小时 |

**一句话（v1.1）**：被判死的是**「12B FLUX.1-dev 在 16GB 内存机器上用 sd-scripts 硬扛」**，不是「本机训 FLUX」这件事。换到 4B base + ai-toolkit，显存和内存都不再是硬门；H20 仍是正式权重与对照实验的主线，路径 D 是本机可用的第二产出通道。

---

## 1.1 结论更正：本机 FLUX 并未判死（路径 D）

### 1.1.1 上一版错在哪

v1.0 把 **FLUX.1-dev 12B + sd-scripts** 的失败（CPU 内存 / 提交上限耗尽、`blocks_to_swap 28` 的换入换出）外推成了「本机 8GB 训 FLUX 不可行」。这个推理漏了一个变量：sd-scripts 的 8GB 配方是为 **12B** 模型设计的，而 FLUX 家族里还有**官方定位就是「受限硬件微调」的 4B base**。

2026-11-10 复查到的公开实测：Windows + RTX 4070 Laptop **8GB** + ai-toolkit + `FLUX.2-klein-base-4B`，3000 步约 4 小时完成，**5.1–5.7 s/it、专用显存约 6 GB、共享显存近零**。同文还记录了三个不报错的坑，恰好全部与我们本机踩过的 WDDM / 内存现象同源。

### 1.1.2 分层判定

| 判定对象 | 结论 | 依据 |
| --- | --- | --- |
| 本机训 **FLUX.1-dev 12B**（sd-scripts） | ❌ 工程上无意义 | fp8 DiT 约 12 GB + T5-XXL 约 5 GB 常驻 CPU，16 GB 内存 + `blocks_to_swap 28` 会在每一步走 PCIe；即使重启拆掉内存墙，10 h 窗口也只能跑几百步 |
| 本机训 **现有 NVFP4 distilled Klein** | ❌ 不可行 | 蒸馏 + 推理量化格式，无对应 trainer，且蒸馏版不是训练目标 |
| 本机训 **FLUX.2 klein 4B Base**（ai-toolkit） | ✅ 可行 | 4B 底座 fp8 约 4 GB 常驻 GPU，文本编码器缓存后卸载；有 Windows 8GB 成功案例 |
| 本机产出**正式 FLUX 权重** | ✅ 从「不可能」改为「有条件」 | 路径 D + S0 数据集；质量上限仍低于 H20 fp16 对照 |

### 1.1.3 三个反直觉但决定成败的设置

| 设置 | 必须值 | 原因 |
| --- | --- | --- |
| `model.low_vram` | **`false`** | 名字骗人。`true` 会把 transformer 停在系统内存、每步经 PCIe 流式搬入，实测 **266 s/it（3000 步约 9 天）**；`false` 让 fp8 后约 4 GB 的 transformer 常驻显卡，降到约 5 s/it |
| `train.cache_text_embeddings` | **`true`** | 8GB 的 make-or-break。启动时把 caption 编码一次后**卸载整个 Qwen3-4B TE**；否则采样阶段 transformer + TE 合计约 8 GB，Windows WDDM 不会报 OOM，而是静默 spill 到共享显存、慢 10–50 倍 |
| `sample.prompts` | **字面写 `mdtr_style`** | 开启文本嵌入缓存后，sample prompt 走另一条预编码路径，`[trigger]` 占位符不会被替换，样本会画出一个「没有主体」的通用人像 |

> Windows 特有监控点：训练中要看**任务管理器 GPU 页的「专用 / 共享 GPU 内存」**，不是控制台。共享内存持续上涨即溢出，日志不会报错。这与 §2.1 里 `vae.to(cuda)` 报成 CUDA OOM 是同一类 WDDM 行为。

### 1.1.4 本机特有的两个环境坑（不同于公开的 4070 案例）

1. **sm_120 / Blackwell 不能用教程里的 torch**：公开案例的 `pip install torch==2.6.0+cu124` 面向 Ada（sm_89），不支持本机 RTX 5060 Ti。必须保留本机 ComfyUI 已有的 **torch 2.12.1+cu130**（或任何 cu128+ 版本），安装 ai-toolkit 依赖时严禁被 requirements 覆盖。
2. **Python 3.13 会卡 scipy**：本机 venv 实测 `3.13.12`，ai-toolkit 锁的 `scipy==1.12.0` 无 py3.13 wheel，会去找 Fortran 编译器。处置：把 pin 放宽为 `scipy>=1.14`，或另建 Python 3.12 venv；并**显式安装 `torchaudio`**（toolkit 无条件 import，requirements 却假定它已随 torch 装好）。Windows 无 Triton 的警告可忽略。

### 1.1.5 与内存墙的关系

路径 D **大幅绕开** §2 的提交上限问题：`low_vram: false` 让底座常驻显存而非 CPU 内存，`cache_text_embeddings: true` 让 TE 在启动后卸载，训练稳态只剩约 4 GB 显存里的 transformer + LoRA 梯度。启动瞬间仍需加载 Qwen3-4B（约 8 GB bf16 / 4 GB fp8），因此**重启让页面文件生效依然是建议第一步**，但不再是「提交上限必须 ≥ 48 GB」的硬门。

### 1.1.6 许可变化（不等于数据合规放宽）

`FLUX.2-klein-base-4B` 是 **Apache 2.0**（可商用），FLUX.1 [dev] / FLUX.2 [dev] 均非商业许可。但**分支 A 作者归属未确证的约束不变**：默认仍仅个人研究、不发布数据集、不发布权重、不商用。底座许可变宽松只移除了「底座」这一层限制。

### 1.1.7 可执行产物与状态

- `configs/ai_toolkit_klein_4b_local.yaml` —— 8GB 专用完整配置，已写入上述三个修正；`low_vram:false`、`cache_text_embeddings:true`、不设 `trigger_word`（caption 里已有 `mdtr_style`，避免二次注入）、Klein 专属 `timestep_type: weighted` / `sample_steps: 28` / `guidance_scale: 4`。
- `scripts/setup_flux_klein_local.sh` —— 预检 + 断点续传下载（transformer 7.75 GB、AE 336 MB、ai-toolkit 源码 36 MB）+ 复用本机已有 `flux2-vae.safetensors`；**刻意不自动装依赖**（见 §1.1.4）。
- 权重可达性已实测：`flux-2-klein-base-4b.safetensors` 7.75 GB / `ai-toolkit/flux2_vae/ae.safetensors` 336 MB 经 hf-mirror 均 200 OK；`Qwen/Qwen3-4B` 由 toolkit 自动分片下载。
- **状态：配置与下载链已核对，尚未在本机实跑**。首次运行前必须先人工确认 torch 版本与 Python 依赖；正式训练仍需 S0 放行。

```bash
DRY_RUN=1 bash scripts/setup_flux_klein_local.sh       # 预检 + 打印下载清单
bash scripts/setup_flux_klein_local.sh                  # 下载（用户确认后执行）
# 装完依赖、S0 放行后：
cd E:/lora_local/ai-toolkit && HF_ENDPOINT=https://hf-mirror.com \
  venv/Scripts/python.exe run.py "C:/Users/Administrator/WorkBuddy/LoRA训练/configs/ai_toolkit_klein_4b_local.yaml"
```

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

> **现实预期**：`blocks_to_swap 28` 意味着每个 step 都有大量 CPU↔GPU 权重交换，本机 `s/it` 会显著高于 SDXL；叠加 16 GB 内存约束，10 h 窗口内很可能只能完成几百步。它适合回答「能不能跑」，不适合产出正式权重。
>
> ⚠️ **本节的「不可行」仅针对 FLUX.1-dev 12B。** 换用 4B base + ai-toolkit 的路径 D（§1.1）不受此结论约束，且已有 Windows 8GB 成功实测。若目标是「在本机真正训出一个 FLUX LoRA」，应优先走路径 D 而非本节。

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
2. FLUX.1 [dev] 权重为**非商业许可**；FLUX.1-schnell 与 **FLUX.2 [klein] 4B / 4B Base 为 Apache 2.0**（路径 D 采用后者）。底座许可变宽松**不改变**第 1、3 条约束。
3. 默认边界：仅个人学习与风格研究；不发布数据集、不发布权重、不商用、不代客出图、不冒名。
4. 若未来有商用打算，需改用可商用底座，并重新核实训练图集的版权归属。

---

## 8. 交付物索引

| 文件 | 作用 |
| --- | --- |
| `FLUX_LoRA训练方案.md` | 本文档 |
| `scripts/train_flux.sh` | H20 FLUX.1-dev 正式训练（已修 `--cache_latents`、加预检 / DRY_RUN / BENCH） |
| `scripts/train_local_flux.sh` | 本机 8 GB FLUX.1-dev 探测（内建提交上限 / 模型 / 显存预检） |
| **`configs/ai_toolkit_klein_4b_local.yaml`** | **路径 D**：本机 8GB + FLUX.2 klein 4B Base 完整配置（含三个反直觉修正） |
| **`scripts/setup_flux_klein_local.sh`** | **路径 D**：预检 + 权重断点续传 + 复用本机 FLUX.2 VAE；不自动装依赖 |
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
**2026-11-10 结论更正**：新增**路径 D**（本机 FLUX.2 klein 4B Base + ai-toolkit），本机 FLUX 不判死；配置与下载链已产出，尚未实跑，详见 §1.1。路径 A 仍是正式权重与对照实验的主线，路径 D 为可选的第二通道，需用户确认后启用。

| # | 事项 | 状态 |
| --- | --- | --- |
| 1 | FLUX 主线路径 | ✅ 已定：H20 正式对照；`scripts/setup_flux_h20.sh` 已产出 |
| 2 | 页面文件迁移 | ✅ 已配置为 `E:\pagefile.sys` 32–48 GB，备份已留；**2026-11-10 01:04 实测仍未重启生效**（`LastBoot=10/15`、`Pagefile=D:\pagefile.sys`、`CommitLimit=34.77 GB`） |
| 3 | 关闭占内存程序 | ⬜ 未执行（用户本次未授权，也未擅自结束任何进程） |
| 4 | 本机下载 22.4 GB 模型 | ⬜ 未执行（主线改走 H20 后不再必要） |
| 5 | S0 数据采集 | ⬜ **仍是正式训练的唯一硬门** |
| 6 | 确认 H20 机时性质（内部配额 / 按小时付费）与用途边界 | ⬜ 待用户确认 |
| 7 | 提供 H20 可用的 `HF_TOKEN`（BFL gated） | ⬜ 待用户准备；或改用 `MODE=fp8` 降级 |
| 8 | 重启 Windows 使页面文件生效，然后重跑 `FULL_BF16=1 BENCH=1 ...` | ⬜ 待用户执行（本次未擅自重启） |
| 9 | **路径 D**：确认是否启用（约 16 GB 下载 + 人工装依赖） | ⬜ 待用户决策；脚本与 YAML 已就绪 |
| 10 | **路径 D**：人工确认 torch cu128+ / Python 3.12 或放宽 scipy pin | ⬜ 未执行；本机 sm_120 + py3.13 是两个已知坑（§1.1.4） |
| 11 | **路径 D**：首次实跑 200 步冒烟，看专用 / 共享 GPU 内存 | ⬜ 未执行 |

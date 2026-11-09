# Guillermo Arias 风格 LoRA 训练方案（NVIDIA H20 单卡）

| 项目 | 内容 |
| --- | --- |
| 版本 | **v1.1（分支 A 已锁定）** |
| 日期 | 2026-11-06 |
| 硬件 | NVIDIA H20（96 GB HBM3，单卡） |
| 训练框架 | kohya_ss / sd-scripts（**`sdxl_train_network.py`** + `flux_train_network.py`） |
| 目标 | **分支 A**：中文社媒流传的"地中海 / 海边阅读 / 厚涂肌理"图集 |
| 状态 | 分支已确认；**Phase 0 溯源 + S0 门禁通过前不开训** |

> **v1.1 变更**：用户确认采用分支 A。分支 B / C / D 全部排除；§2 从"决策门"改为分支 A 的可执行采集与准入规范；新增 `configs/sources.csv`、`configs/caption_rules_branchA.md`、`scripts/init_dataset.py`、`scripts/audit_manifest.py` 四件 S0 交付物。触发词 `mdtr_style`、训练框架、H20 配置与评估方案均不变。
>
> **补充（2026-11-08，v1.1 修订）**：新增《`本机训练评估与成本.md`》——实测本机为 RTX 5060 Ti 8GB / 16GB 内存。用户确认可提供 **>10 小时独占窗口**后机会成本归零，结论升级为：**本机承担全部 SDXL 训练与消融，H20 只保留 dim64 高容量正式版与 FLUX.1-dev 对照**（H20 配额从 30–40 降至约 4–6 GPU·h）。8GB 的 6 项显存妥协、FLUX 本机不可行、无人值守长跑风险三项**不随时间改变**。新增 `scripts/run_local_ablations.sh`（10h 排期 / 按实测 s/it 自动裁剪 / 组级断点续跑 / GPU 监控）与 `configs/dataset_local_single_palette.toml`（G4 可选消融）。本文 §5–§6 的 H20 参数**不适用于本机**，切勿照抄。

---

## 0. 执行摘要

三句话结论：

1. **先解决"训练谁"**：`Guillermo Arias` 至少指向四位可查的同名创作者，而中文社媒广为流传的那批"地中海治愈系插画"存在**籍贯、配色、笔触、身份四条互相矛盾的描述，且找不到可验证的一手作品集**。这直接决定数据集怎么建，是本项目的第一号风险。
2. **风格本身是可以被建模的**：无论作者归属如何，这批图像有清晰、可拆解、可量化的视觉语言（色块厚涂 + 强明暗 + 简化形体 + 留白构图 + 松弛日常题材）。方案把它当作**风格谱系（style corpus）LoRA** 来设计，而非"某位艺术家复刻"，规避归属风险。
3. **H20 的正确用法是"用显存换算力"**：H20 有 96 GB / 4.0 TB/s 的旗舰级显存系统，但 BF16 稠密算力仅约 148 TFLOPS（约为 H100 SXM 的 15%）。因此**关掉一切"用算力换显存"的优化**（梯度检查点、CPU offload、块交换、量化底座），把 latent 与文本编码全部预缓存，开大 batch，用 bf16 + SDPA。LoRA 训练在 H20 上是 **compute-bound**，不是显存瓶颈。

**推荐主路径（v1.1 修订）**：**本机长跑为主 + H20 只做补完**。本机按 views/image 重排步数，跑完 benchmark、Pilot、80 views 主跑与最多 3 组消融（约 8.6–11.3 h）；H20 保留 dim64 高容量正式版、FLUX.1-dev 对照及被裁剪的消融，机时预留从 **30–40 GPU·小时** 下调为 **4–6 GPU·小时**。固定 prompt 网格选 checkpoint 的评估流程不变。

**本机路径**：RTX 5060 Ti 8GB 需 6 项显存妥协，且**不能照抄 H20 的 3000 步**（batch 从 6 降到 1 会欠训 6 倍）。时间充裕时应把机时花在多组消融而非拉长单轮（views/图 上限约 120，超过即过拟合）。详见《`本机训练评估与成本.md`》v1.1 §3–§6。

---

## 1. 艺术家研究

### 1.1 身份辨析：四位同名的 Guillermo Arias

| # | 可验证身份 | 媒介与题材 | 一手/准一手线索 | 与"地中海插画"是否吻合 |
| --- | --- | --- | --- | --- |
| A | **中文社媒流传的"西班牙插画师"**（加利西亚 / 阿利坎特说法并存，称任职 Inditex 印花部门） | 地中海日常、海边阅读、女性与花卉、复古色、数字模拟油画肌理 | 仅见公众号、微博、头条、百家号等二手转述；**未检索到个人网站、作品集或可确认的官方社媒账号** | 就是目标图集本身，但归属存疑 |
| B | **墨西哥插画师 @hangtheguille**（品牌 Besos Robados） | 手绘电影海报、独立乐队视觉、时尚图形、70 年代恐怖片单页风；手机 + 触控笔创作 | Cool Stories About Art, 2026-05-03 专题；Redbubble 店铺；Instagram 约 2,500 帖 | ❌ 视觉语言完全不同 |
| C | **墨西哥蒂华纳摄影记者 @guillermoarias**（AFP 供稿） | 美墨边境纪实摄影；另有被中文自媒体称作"Procreate 油画作品"的图 | 圣地亚哥国际机场艺术项目页 san.org；MutualArt 艺术家页；Istanbul Photo Awards 2019 | ❌ 主体是新闻摄影 |
| D | **Guillermo Arias-Camisón Cano**（西班牙概念艺术家，Codigames） | 游戏 2D 概念设计、角色设计、插画教学 | RocketReach 职业档案（ESDIP 等） | ❌ 游戏概念设计 |

### 1.2 目标图集的内部矛盾（关键发现）

把中文来源并排看，描述并不自洽：

| 维度 | 说法一 | 说法二 | 影响 |
| --- | --- | --- | --- |
| 籍贯 | 西班牙**加利西亚**（西北海岸，阴雨多风） | 西班牙**阿利坎特**（东南地中海，强光白墙） | 光影设定不同 |
| 色彩 | 低饱和莫兰迪：米白、灰粉、橄榄绿、淡蓝 | 明亮糖果色：橙、蓝、绿、粉高饱和并置 | 训练目标色分布直接冲突 |
| 笔触 | 线条流畅柔和、光影过渡细腻 | 粗粝厚涂、方向性明显、色块堆叠、不做渐变 | 纹理学习目标冲突 |
| 身份 | 自学成才的西班牙平面设计师 / Inditex | "墨西哥摄影师 Guillermo Arias 的 Procreate 油画作品" | 疑为把 C 的署名套到 A 的图上 |
| 一手来源 | 无 | 无 | 图集可能是多作者拼合，或含 AI 生成图 |

**判断**：这批图更接近一个**被社交媒体聚合、贴了同一名字的视觉图集**，而不是可确证的单一作者作品集。这不是坏消息——对 LoRA 而言，"一致的视觉语言"才是可学习的对象——但必须**在方案里去掉"复刻某位在世艺术家"的表述**，并在数据采集阶段做溯源审计，否则会把不同作者甚至 AI 图的审美混进同一个 LoRA，学出一锅粥。

### 1.3 风格 DNA 拆解（可直接映射到训练目标）

| 维度 | 特征 | 可学习的信号 | 训练侧抓手 |
| --- | --- | --- | --- |
| 色彩 | 大面积低饱和底色 + 小面积高亮点缀（朱红唇、明黄花、钴蓝布）；另一支为明亮橙/蓝/绿/粉并置但不刺眼 | 全局 HSV 分布、饱和度双峰、色块面积比 | 色彩直方图作为评估指标；**两支配色做成可控子标签** |
| 光影 | 地中海硬光、强明暗对比、树影投射；不靠平滑渐变，靠色块堆叠形成明暗 | 亮度直方图双峰、边缘处的高对比 | 保留原图明暗，勿做自动对比度增强 |
| 笔触/肌理 | 数字模拟厚涂油画 / 水粉 / 水彩；可见涂抹痕迹、方向性笔触、颗粒感；**拒绝平涂** | 高频能量、局部方向性纹理 | 1024+ 分辨率训练；Laplacian 方差作为评估指标 |
| 造型 | 简化形体、弱化五官、无硬轮廓线、以色块拼接塑形；重姿态不重面孔 | 中低频形状统计 | 风格 LoRA 天然擅长；但需防止把所有人都画成同一张脸 |
| 构图 | 简洁背景、大面积留白、扁平化空间、几何元素穿插、"呼吸感" | 空间布局、主体占比 | **必须保持原图长宽比 + 多分辨率分桶，禁止强行居中裁方** |
| 题材 | 独处女性、海边阅读、露台发呆、庭院浇花、晒太阳；音乐（吉他/唱片）、花卉、几何图形 | 语义内容 | **这是内容，不是风格** → 用 caption 显式剥离，防内容泄漏 |
| 情绪 | 松弛、慵懒、静谧、度假感、"浪费时间的浪漫" | 难以直接监督 | 靠数据集整体一致性 + 人工盲评 |
| 反锚点 | 不是照片写实、不是赛璐璐动漫、不是高细节概念设计、不是光滑矢量插画 | — | 正则化图像 / 域外验证 prompt 检验 |

### 1.4 难度分级与风险

- **容易学到**：色板、饱和度、厚涂颗粒、简化五官、强明暗。这类"全局统计量"LoRA 几百步就能抓到。
- **难学到**：笔触的方向性与堆叠逻辑、留白构图、松弛情绪。需要高分辨率、足够样本、保持长宽比分桶，且依赖 checkpoint 早停。
- **最大风险：内容泄漏（content leakage）**。数据集里"海边 + 女性 + 阅读"高度重复，模型很容易把**题材**当成**风格**学进去，导致换个 prompt 也只画海边女人。对策是 §3.5 的标注策略（caption 只写内容、触发词承载风格）与 §7.3 的域外验证集。

---

## 2. Phase 0：分支 A 锁定、采集与溯源门禁

**用户已确认：目标 = 分支 A。** 分支 B / C / D 全部排除，不进入数据集。本项目正式定义为：

> **网络图集风格谱系 LoRA（style corpus study）**：训练目标是一组可验证的视觉语言，
> 不声称作者归属、不冒名、不使用艺术家姓名作为触发词。

| 已排除分支 | 排除理由 |
| --- | --- |
| B · @hangtheguille（Besos Robados） | 70 年代海报 / 乐队视觉，含大量文字，视觉语言完全不同 |
| C · @guillermoarias（AFP 摄影记者） | 主体是新闻摄影；中文自媒体把其署名套到 A 的图上，正是混淆源头 |
| D · Arias-Camisón Cano（Codigames） | 游戏 2D 概念设计，非目标图集 |

### 2.1 采集渠道（按可信度排序）

| 优先级 | 渠道 | 怎么用 | 注意 |
| --- | --- | --- | --- |
| **P0** | 你手上已保存的原图 / 收藏夹 | 直接作为种子图，用于反搜定位原始出处 | 可信度最高，先从这里建种子集 |
| **P1** | 中文公众号 / 微博 / 小红书 / 头条的**配图原文** | 记录文章 URL 与页面声称的署名（**原样抄录，不要自行"纠正"**） | 多为转载，必须反搜 |
| **P2** | Pinterest / Tumblr / 图集站的同主题 board | 用 P0 种子图做以图搜图扩样 | 极易混入其他作者与 AI 生成图，逐张过 §2.2 |
| **P3** | Behance / Instagram / ArtStation 个人主页 | **仅当反搜命中可确认的原创主页**才采 | 分支 A 至今未检索到可确证的一手主页 |

**禁止**：把搜索引擎"相关图片"整批打包下载；为凑数量把 AI 生成图当风格样本；用低分图上采样补齐分辨率。

### 2.2 视觉准入判据（每张逐条过）

一张图必须同时满足 A 组全部、且 B 组命中 ≥ 4 项，才能 `decision=keep`。

**A 组 · 硬门槛（任一不满足直接 drop）**

1. 长边 ≥ 1024 px，且为完整单幅作品（非九宫格 / 营销长图 / 带 UI 的转载页截图）；
2. 无水印、账号名、二维码、logo、可见签名、大段文字；
3. 未被拉伸变形，保留原图长宽比与留白；
4. 明显不是照片、3D 渲染、赛璐璐动漫、光滑矢量插画。

**B 组 · 风格一致性（≥ 4 / 6）**

1. 色块塑形、先块后面，五官被弱化，无硬轮廓线；
2. 可见方向性笔触 / 厚涂堆叠 / 颗粒感，不是平涂渐变；
3. 硬光强明暗，明暗靠色块交界形成而非平滑过渡；
4. 大面积简洁背景 / 留白，空间扁平、有呼吸感；
5. 松弛日常题材：独处、阅读、露台、庭院、花卉、器物；
6. 色板符合两支之一：低饱和米白灰粉橄榄 + 小面积高亮点色，或明亮橙蓝绿粉并置。

**量化辅助**（`scripts/audit_dataset.py` 输出，用于快速筛选，**不替代人眼**）：

| 指标 | 判据 | 动作 |
| --- | --- | --- |
| `lap_var` | < 全集中位数的 1/3 | 疑似平涂，人工复核 |
| `mean_s` | < 0.35 → `retro_muted`；≥ 0.35 → `candy_bright` | 写入台账 `palette_branch` |
| `border_std` | < 3.0 | 疑似截图 / 边框，人工复核 |
| `near_duplicate_of` | 非空 | pHash ≤ 6，只留分辨率最高的一张 |

> 低分图**禁止上采样**：放大算法会把粗粝笔触边缘"合理化"成平滑过渡，恰好抹掉要学的那一层。短边不足 1024 px 的图宁可不要。

### 2.3 溯源审计（逐张打勾，写入 `configs/sources.csv`）

1. 反向图片检索（Google Lens / Bing Visual Search / Yandex），记录**最早出处**与页面署名；
2. 区分"原创发布"与"二次转载"：转载页署名一律记为 `claimed_attribution`，**不算确证**；
3. 检查 EXIF / 文件哈希，识别同一图的多次压缩版本（pHash ≤ 6 交给 `audit_dataset.py`）；
4. 人工复核疑似 AI 生成痕迹：手指、织物纹理、文字、笔触逻辑不连贯、局部过度精细；
   **不要依赖所谓 AI 检测器**，其结论不可靠；
5. 逐条记录 `来源 URL / 平台 / 页面声称署名 / 采集日期 / 最早出处 / 授权状态 / 配色支 / 题材桶 / AI 嫌疑 / 去留`；
6. `ai_suspicion=high` 必须 `drop`；`license_status` 一律先填 `unknown`，**不得臆断授权**。

**判定阈值**

| 可溯源率 | 结论 |
| --- | --- |
| ≥ 80% | 正常推进 |
| 60%–80% | 继续，但模型卡必须写明"网络图集风格研究，作者归属未确证" |
| < 60% | **停止开训**，回到采集 |
| 溯源后 AI 图占比 > 20% | 训练目标如实改称"网络图集视觉语言"，并重新评估是否值得继续 |

### 2.4 S0 门禁（自动判定，返回码 0 才准进 S1）

```bash
# 1. 填好 configs/sources.csv 后，建训练目录、caption 骨架与素材台账
python scripts/init_dataset.py --sources configs/sources.csv

# 2. 人工逐条把 caption 里的 TODO 换成内容描述（见 configs/caption_rules_branchA.md）

# 3. 图像体检（pHash / 色板 / 纹理 / caption 违规）
python scripts/audit_dataset.py --dir dataset/train/10_mdtr --out audit_report.csv

# 4. S0 总门禁
python scripts/audit_manifest.py --sources configs/sources.csv --dataset dataset/train/10_mdtr
```

`audit_manifest.py` 检查：必填字段、id 唯一、入选数 60–200、题材桶配额偏差 ≤ 10pp、
配色支 60:40（容差 15pp）、`undecided` 必须为 0、AI 高嫌疑必须 drop、
caption 无 TODO / 无风格词 / 触发词首位、图片与 caption 一一对应、无孤儿文件。

**红线**：未完成溯源、未清干净重复、caption 仍残留 TODO 之前，不要烧 H20 机时。H20 算力弱，跑错方向的代价比高算力卡更贵（以时间计）。

---

## 3. 数据集工程

### 3.1 规模与配额

目标 **90–150 张** 高质量图（下限 60 张可跑通，上限 200 张以上收益递减且易引入噪声）。风格 LoRA 的"质量 > 数量"，且**一致性 > 多样性**：宁可 80 张同一视觉语言，不要 200 张混三个作者。

建议配额（防止题材过拟合）：

| 题材桶 | 占比 | 说明 |
| --- | --- | --- |
| 人物 + 户外（海边/庭院/露台） | 40% | 主分布 |
| 人物 + 室内 / 静物 | 20% | 打散"海边"绑定 |
| 纯风景 / 建筑 / 空镜 | 20% | 教构图与光影，不依赖人物 |
| 花卉 / 器物 / 局部特写 | 10% | 教笔触与色块 |
| 域外锚点（非典型场景） | 10% | 城市、夜景、动物等，逼模型学风格而非题材 |

配色两支按约 60 : 40 分布，并分别打子标签（见 §3.5）。

### 3.2 清洗规则

| 保留 | 剔除 |
| --- | --- |
| 长边 ≥ 1024 px、细节清晰 | 缩略图、强 JPEG 压缩、模糊、放大插值图 |
| 完整单幅作品 | 九宫格、拼图、长图截图、含 UI 的转载页 |
| 无遮挡画面 | 水印、账号名、二维码、logo、边框 |
| 原图长宽比 | 被裁掉留白/被拉伸变形的版本 |
| 风格一致 | 明显属于其他作者/其他画风/照片/3D 渲染 |
| 单幅单主题 | 含大段文字的海报 / 排版物料 |
| 符合 §2.2 视觉准入判据 | 疑似 AI 生成图（`ai_suspicion=medium/high`） |

> 分支 A 的完整准入判据、量化阈值与溯源流程见 **§2.2 / §2.3**，台账字段见 `configs/sources.csv`。

### 3.3 去重与多样性

- **近重复检测**：pHash 汉明距离 ≤ 6 视为重复，只留分辨率最高的一张（`scripts/audit_dataset.py` 已实现，无需外部依赖）。
- **语义近重复**：可选，用 CLIP image embedding 余弦相似度 > 0.95 再筛一轮，防止构图雷同。
- **色彩统计体检**：`audit_dataset.py` 输出平均饱和度、亮度、色相直方图；若出现两个明显分离的簇，正是 §3.5 子标签的依据。

### 3.4 分辨率与分桶

SDXL 训练分辨率 1024，启用 Aspect Ratio Bucketing，**保持原始长宽比**：

| bucket | 用途 |
| --- | --- |
| 768×1344 / 832×1216 | 竖构图（人物、门廊） |
| 1024×1024 | 方形 |
| 1216×832 / 1344×768 | 横构图（海景、留白） |

`--enable_bucket --min_bucket_reso=768 --max_bucket_reso=1536 --bucket_reso_steps=64`。**不要把所有图居中裁成正方形**——留白与横向呼吸感本身就是风格特征。

### 3.5 标注策略（本方案最关键的设计）

#### 原则一：触发词用自造 token

选一个**不是真实单词、不会和模型词表冲突**的 token，例如 `mdtr_style`。不要用 `Guillermo Arias`、`gArias` 之类：既避免暗示官方/署名，也避免人名被 tokenizer 拆成已有语义片段。

#### 原则二：caption 只写"画面里有什么"，绝不写"这是什么风格"

这是 Black Forest Labs 官方 FLUX.2 klein LoRA 指南明确给出的做法：**若把 `oil painting`、`muted palette` 写进 caption，模型会把风格挂到这些词上，而不是烘进 LoRA 权重**，结果是触发词失效、风格泄漏到所有生成里。风格必须由模型从像素中自行推断。

| | 示例 |
| --- | --- |
| ✅ 正例 | `mdtr_style, a woman reading a book on a seaside terrace, striped chair, potted plant, white wall, strong sunlight` |
| ✅ 正例 | `mdtr_style, a vase of flowers on a table, scattered fruit, window light, simple background` |
| ❌ 反例 | `a beautiful oil painting in muted retro colors with thick impasto brushstrokes, Mediterranean light` |
| ❌ 反例 | `mdtr_style, impressionist painterly illustration, low saturation palette, healing summer mood` |

流程：用 BLIP / WD-Tagger / 视觉大模型自动打内容描述 → 人工扫一遍，删掉所有风格形容词 → 在**每条 caption 固定首位**插入触发词。

> ⚠️ 若启用 `--shuffle_caption`，必须同时设 `--keep_tokens=1` 锁住触发词位置；本方案默认**不启用** shuffle。

#### 原则三：把"资料矛盾"变成"推理旋钮"

中文资料对配色的两派描述（低饱和复古 vs 明亮糖果色）不必二选一。给对应图片各加一个子标签，这些词就会变成推理时可调节的旋钮：

```
mdtr_style retro_muted, a woman ...     # 低饱和莫兰迪支
mdtr_style candy_bright, a woman ...    # 明亮橙蓝绿粉支
```

推理时 `mdtr_style retro_muted, a night market in Tokyo` 即可把风格迁移到全新题材，同时指定配色支。**只给"你希望在生成时控制"的维度打标签**，其余一律不写。

#### 原则四：正则化图像（可选，列为消融项）

用底模按训练 caption（**去掉触发词**）生成 60–100 张其他风格的图，放在 `reg/` 目录、repeat = 1，用于保住底模的人体/构图常识，防止 LoRA 把"画人"的能力一起改掉。风格 LoRA 有时不需要正则图，因此作为消融实验 A/B 决定是否启用。

### 3.6 目录结构

```
configs/
├── sources.csv             # 【S0 主台账】来源 / 署名 / 溯源 / 授权 / 配色支 / 题材桶 / 去留
├── caption_rules_branchA.md# 分支 A 标注手册（结构 / 词表 / 禁用词 / 范例）
└── dataset_sdxl.toml       # kohya 数据集配置
dataset/
├── raw/                    # 采集原图，按 id 命名（A001.jpg ...）
├── train/
│   ├── 10_mdtr/            # 主训练集，repeat 10（init_dataset.py 生成）
│   │   ├── A001.png
│   │   └── A001.txt        # 同名 caption，人工填满、无 TODO
│   └── 1_reg/              # 可选正则图，repeat 1
├── dataset_manifest.csv    # init_dataset.py 生成的素材台账（含 sha256）
├── captions/               # caption 备份与审核记录
└── audit_report.csv        # audit_dataset.py 输出
s0_gate_report.txt          # audit_manifest.py 门禁报告
```

---

## 4. 底座模型选择

| 底座 | 优势 | 劣势 | 许可 | 本方案角色 |
| --- | --- | --- | --- | --- |
| **SDXL 1.0** | 生态最成熟、LoRA 工具链最全、迭代快、风格吸收好、ControlNet 完备 | 笔触细节与 prompt 遵循弱于 FLUX | CreativeML Open RAIL++-M，**允许商用** | ✅ **主实验** |
| **FLUX.1-dev** | 构图/解剖/纹理保真更强，厚涂笔触更可能还原 | 12B 更吃算力；LoRA 生态较小 | **非商业许可**，商用需 BFL 单独授权 | ✅ 对照实验 |
| FLUX.1-schnell | Apache 2.0，商用无忧；4 步出图 | 蒸馏后质量较低，笔触细节弱 | Apache 2.0 | 仅商用备选 |
| SD 3.5 Large | MMDiT 架构 | 生态小，LoRA 支持弱 | Stability Community License | 不选 |
| FLUX.2 klein 4B | 轻、Apache 2.0、官方有 LoRA 指南 | 生态新，风格容量待验证 | Apache 2.0 | 后续可试 |

**决策**：主交付用 SDXL（可商用、迭代快、适合 H20 的弱算力）；FLUX.1-dev 仅作"笔触保真上限"的对照，若最终要商用则不纳入交付。两者 LoRA **互不兼容**，需分别训练。

---

## 5. H20 硬件特性与配置策略

### 5.1 规格（以 `nvidia-smi` / 服务器实际拓扑为准）

| 参数 | H20 (SXM / HGX) |
| --- | --- |
| 架构 | Hopper（GH100，约 78 SM） |
| 显存 | 96 GB HBM3 |
| 显存带宽 | 4.0 TB/s（**高于 H100 SXM 的 3.35 TB/s**） |
| BF16 / FP16 Tensor Core（dense） | ~148 TFLOPS（约 H100 SXM 的 15%） |
| FP8 / INT8 Tensor Core（dense） | ~296 TFLOPS / TOPS |
| TF32 / FP32 | 74 / 44 TFLOPS |
| NVLink | 900 GB/s（8 路 HGX） |
| TDP | ~400 W（部分资料标 500 W） |

> 社区文章对 H20 参数存在大量互相矛盾的说法（有称 400 TFLOPS FP16、2 TB/s 带宽、不支持 FP8、无 NVLink 等）。**以上表为准，并在开训当天用 `nvidia-smi -q` 与 200 步 benchmark 实测校准**，不要照搬博客数字。

### 5.2 瓶颈判断

LoRA 只训练低秩适配器，优化器状态与激活都很小。在 96 GB 显存上，SDXL / FLUX 全精度权重 + latent + 激活都能**常驻显存**。因此 H20 上：

- **不是显存瓶颈**，是**算力瓶颈**：前/反向的卷积与注意力 GEMM 受 148 TFLOPS 限制；
- 任何"用额外计算换显存"的优化，在 H20 上都是纯亏损。

### 5.3 H20 专属配置原则

| 操作 | 建议 | 理由 |
| --- | --- | --- |
| `--gradient_checkpointing` | **关闭**（SDXL）；FLUX 仅在 OOM 时才加 | 会重算前向，多耗约 25–30% 算力，而显存完全够 |
| `--blocks_to_swap` / `--cpu_offload_checkpointing` | **不用** | 同上，PCIe 往返只会拖慢 |
| `--fp8_base` / 量化底座 | **不用** | 风格 LoRA 对色彩/笔触精度敏感；显存够就保持 bf16 |
| `--cache_latents` + `--cache_latents_to_disk` | **开启** | 每 epoch 省掉 VAE 编码，直接省算力 |
| `--cache_text_encoder_outputs`（FLUX） | **开启** | 省掉 CLIP-L / T5-XXL 前向 |
| `--mixed_precision=bf16` / `--save_precision=bf16` | **开启** | Hopper 原生 BF16，比 fp16 稳定且无需 loss scale |
| `--sdpa` | **开启** | PyTorch SDPA / FlashAttention 路径，Hopper 上优于 xformers |
| `train_batch_size` | **4–8**（SDXL）/ **2–4**（FLUX） | 提高 SM 利用率，充分利用大显存 |
| `--gradient_accumulation_steps` | **1** | 不需要用累积换显存 |
| `--highvram`（FLUX） | **开启** | 让权重常驻显存，禁止低显存模式 |
| `--persistent_data_loader_workers` | **开启** | 减少每 epoch 的 dataloader 启停抖动 |
| 多分辨率分桶 | **开启** | 保持长宽比，见 §3.4 |
| 同卡并行 | 训练 **串行**；评估生成可搭车跑 | 算力已饱和，并行训练不会提速；评估生成占用很小 |
| 多卡 | **不建议**（单卡 LoRA） | 梯度同步收益低于通信开销，且拓扑需现场确认 |

### 5.4 先测后算（强制）

开训第一步：用同一配置跑 **200 步 benchmark**，记录 `s/it`。所有机时估算用下面这个公式外推，**不要相信纸面 TFLOPS**：

```
训练时长(h) = 总步数 × 实测 s/it / 3600
总机时     = 训练 + 周期样本生成 + 检查点评估 + 消融实验 + 预留 20% 重跑
```

粗略参考区间（**仅为按算力比率外推的量级估计，务必实测校准**）：SDXL 1024 / bs4 约 3–8 s/it；FLUX.1-dev 1024 / bs2 约 8–20 s/it。若实测显著偏离，以实测为准重排预算。

---

## 6. 训练超参数与执行计划

### 6.1 SDXL 主实验

| 参数 | 值 | 备注 |
| --- | --- | --- |
| `network_module` | `networks.lora` | |
| `network_dim / alpha` | `64 / 32` | 风格类；消融比 32/16、128/64 |
| `learning_rate`（UNet） | `1e-4` | 消融比 `5e-5` |
| `text_encoder_lr` | `5e-6` | 风格 LoRA 可设 0（只训 U-Net）作为消融 |
| `lr_scheduler` | `cosine_with_restarts` | 对照 `constant_with_warmup` |
| `lr_warmup_steps` | 总步数 5% | |
| `optimizer_type` | `AdamW8bit` | 显存富余可换全精度 `AdamW`；或 `Prodigy` 免调 LR |
| `mixed_precision / save_precision` | `bf16 / bf16` | |
| `resolution` | `1024` + buckets | |
| `train_batch_size` | `6` | OOM 时降到 4 |
| `max_train_steps` | `3000` | 用 checkpoint 早停，不追求跑满 |
| `save_every_n_steps` | `250` | 得到 12 个检查点用于选优 |
| `sample_every_n_steps` | `250` | 固定 prompt 网格，见 §7.3 |
| `min_snr_gamma` | `5`（可选） | 加速收敛；列为消融 |
| `noise_offset` | `0`（默认不加） | 风格 LoRA 容易污染色板，仅作消融 |
| `seed` | `42` | 固定，保证消融可比 |
| `--shuffle_caption` | **不启用** | 触发词固定首位 |

可直接运行：`scripts/train_sdxl.sh`（配套 `configs/dataset_sdxl.toml`）。

### 6.2 FLUX.1-dev 对照实验

| 参数 | 值 |
| --- | --- |
| `network_module` | `networks.lora_flux` |
| `network_dim / alpha` | `16 / 16`（消融 32） |
| `learning_rate` | `1e-4` |
| `optimizer` | `adamw8bit` |
| `mixed_precision` | `bf16` |
| `train_batch_size` | `2` |
| `max_train_steps` | `2000` |
| `--timestep_sampling` | `flux_shift`（旧版回退 `shift` + `--discrete_flow_shift 3.1582`） |
| `--model_prediction_type` | `raw` |
| `--guidance_scale` | `1.0` |
| 缓存 | `--cache_latents --cache_latents_to_disk --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk` |
| 其他 | `--sdpa --highvram --save_model_as safetensors --persistent_data_loader_workers` |

> ⚠️ `--cache_latents` 必须显式给出：sd-scripts 不会因 `*_to_disk` 自动开启，缺它时每个 step 都会重跑 AE。2026-11-09 已修入 `scripts/train_flux.sh`。

> ⚠️ FLUX 的 flow-matching 三件套（`flux_shift` / `raw` / `guidance_scale=1.0`）是硬要求。把 `guidance_scale` 留在推理默认的 3.5 是社区已知的低质量输出成因。kohya 参数随版本变动，开训前请核对仓库当前的 `flux_train_network.md`。

可直接运行：`scripts/train_flux.sh`（H20）。**模型获取、本机 8GB 探测、非 gated 拆分方案与合规叠加见 `FLUX_LoRA训练方案.md`**；本机 FLUX 的硬门是 16 GB 系统内存而非显存。

### 6.3 分阶段执行

| Stage | 目标 | 产出 | 退出条件 |
| --- | --- | --- | --- |
| **S0** | 目标锁定 + 溯源 + 数据清洗 + 标注 | `dataset/`、`dataset_manifest.csv`、`audit_report.csv` | 分支确认；去重完成；caption 抽检 100% 无风格词 |
| **S1** | 环境 + 200 步 benchmark | 实测 `s/it`、显存峰值、机时重排 | loss 正常下降、无 OOM |
| **S2** | Pilot：60 张 / 1200 步 / SDXL | 4 个 checkpoint + 网格图 | 触发词有效、色板方向正确 |
| **S3** | 正式训练：全量 / 3000 步 / SDXL | 12 个 checkpoint | 选出最佳 checkpoint |
| **S4** | FLUX 对照 2000 步 | FLUX LoRA + 网格图 | 与 SDXL 版本 A/B |
| **S5** | 消融 + 评估 + 模型卡 | 评估报告、最终 `.safetensors` | 指标达标、盲评通过 |

**S2 是第二个决策门**：Pilot 若出现"触发词无效 / 风格泄漏 / 色板跑偏"，先回头查 caption 和数据，不要靠加步数硬推。

### 6.4 机时预算（按 §5.4 公式重算）

| 档位 | 内容 | 步数 | 估计 GPU·h |
| --- | --- | --- | --- |
| 最小可验证 | S1 benchmark + S2 Pilot（SDXL） | ~1400 | 3–6 |
| **推荐** | S1–S3 全量 SDXL + 网格评估 | ~3200 | 8–16 |
| 充分 | 上述 + S4 FLUX 对照 + S5 消融 4 组 | ~3200 + 2000 + 4×1200 | **25–40** |

建议一次性申请 **40 GPU·小时**，其中 20% 留给重跑。H20 算力弱，**省机时的关键不是减少步数，而是 S2 早停 + 固定网格评估 + 少做无效消融**。

---

## 7. 评估与消融

### 7.1 铁律：不看 loss，看固定网格

风格 LoRA 的 loss 会在图像已经开始过拟合之后继续下降。以视觉峰值选 checkpoint（经验上常在 750–2000 步，而非最后一步）。每个 checkpoint 用**同一 seed、同一组 prompt、同一采样器/CFG** 出网格图。

### 7.2 定量指标（`scripts/eval_checkpoints.py`）

| 指标 | 计算方式 | 期望 |
| --- | --- | --- |
| 色彩贴合度 | 生成图 vs 真实作品集的 HSV 直方图距离 / 平均饱和度差 | 越小越好；重点看"低饱和主峰 + 高亮小峰"是否复现 |
| 纹理粗糙度 | 灰度 Laplacian 方差 / 高频能量 | 与真实作品集同量级；过低=学成平涂塑料感 |
| 风格相似度 | CLIP image embedding 与作品集质心的余弦相似度（需 torch + CLIP，可选） | 域内高、域外仍保持 |
| 指令遵循 | prompt-image CLIP 相似度 | 不能为了风格牺牲内容 |
| **风格泄漏率** | **不带触发词**时仍呈现该风格的比例 | 越低越好；高 → 过拟合 / caption 写了风格词 |
| 配色支可控性 | 加 `retro_muted` vs `candy_bright` 的饱和度差 | 应有显著且稳定的差异 |

### 7.3 验证 prompt 集（固定，覆盖域内 / 域外）

域内：`mdtr_style, a woman reading by the sea, striped chair, strong sunlight`
域内：`mdtr_style, a vase of flowers on a table, window light`
域外（**最关键，测内容泄漏与泛化**）：
- `mdtr_style, a night market in Tokyo, lanterns, crowd`
- `mdtr_style, a snowy mountain cabin at dawn`
- `mdtr_style, a red bicycle against a white wall`
- `mdtr_style, a cat sleeping on a windowsill`
- `a night market in Tokyo, lanterns, crowd`（**无触发词**，测泄漏）

配色支：`mdtr_style retro_muted, ...` / `mdtr_style candy_bright, ...`

### 7.4 人工盲评

5–7 名评审，把**真实作品、底模直出、候选 checkpoint 出图**打乱混排，三问打分（1–5）：①像不像目标风格 ②是否仍听 prompt ③有没有"复制粘贴感/塑料感"。取平均分最高的 checkpoint，而不是最后一步。

### 7.5 消融矩阵（机时充足时建议跑）

| # | 变量 | 对照 | 想回答的问题 |
| --- | --- | --- | --- |
| 1 | `network_dim` | 32 / 64 / 128 | 笔触细节需要多大容量 |
| 2 | caption 策略 | 纯触发词 / 触发词+内容 / 含风格词 | 验证 §3.5 的关键假设 |
| 3 | 正则化图像 | 有 / 无 | 是否伤到底模构图能力 |
| 4 | 底座 | SDXL / FLUX.1-dev | 厚涂笔触的还原上限 |
| 5 | `text_encoder_lr` | 0 / 5e-6 | 是否需要动文本编码器 |
| 6 | 学习率 | 5e-5 / 1e-4 | 收敛速度与色板稳定性 |

### 7.6 失败模式速查

| 症状 | 可能原因 | 处方 |
| --- | --- | --- |
| 触发词加了和没加一样 | caption 里写了风格词；步数不足；dim 太小 | 删掉所有风格形容词；加步数；dim 提到 64 |
| 不带触发词也全是这个风格 | 过拟合 / caption 泄漏 | 降步数、早停、加正则图、LoRA 权重降到 0.6–0.75 |
| 换个 prompt 还是海边女人 | **内容泄漏**，题材被当风格 | 增加域外锚点图；caption 把场景词写全；提高数据题材多样性 |
| 整体色偏（全变米黄/灰） | caption 里反复出现 `warm/soft/cozy`；数据色板单一 | 删柔和类形容词；补冷色调样本；固定白平衡 |
| 笔触变光滑塑料 | 分辨率不足 / 图被压缩 / 步数过多 | 保证长边 ≥1024；关掉自动锐化；早停 |
| 每张图背景一模一样 | repeat 过高、近重复未清干净 | 降 repeat、重跑去重、加中性背景样本 |
| 生成图出现伪签名/文字 | 训练图含签名或水印 | 清洗阶段剔除，或推理时加负向词 |
| FLUX 输出糊/色彩怪 | `guidance_scale` 未设 1.0 / `model_prediction_type` 错 | 按 §6.2 三件套核对 |

---

## 8. 交付物

| 文件 | 用途 | 阶段 |
| --- | --- | --- |
| `GuillermoArias_风格LoRA训练方案.md` | 本文档（v1.1，分支 A 已锁定） | 全程 |
| **`configs/sources.csv`** | **S0 主台账**：采集渠道、溯源、授权、配色支、题材桶、去留 | S0 |
| **`configs/caption_rules_branchA.md`** | **分支 A 标注手册**：caption 结构、子标签判定、题材桶词表、禁用词、范例 | S0 |
| **`scripts/init_dataset.py`** | 从台账建训练目录、生成 caption 骨架与 `dataset_manifest.csv`（不覆盖人工标注） | S0 |
| **`scripts/audit_manifest.py`** | **S0 门禁**：配额 / 溯源率 / 配色支 / AI 嫌疑 / caption / 文件一致性 | S0 |
| `scripts/audit_dataset.py` | 图像体检：pHash 去重、尺寸/长宽比、色彩统计、纹理能量、caption 违规 | S0 |
| `configs/dataset_sdxl.toml` | kohya 数据集配置（SDXL：分桶 / repeat / 正则图） | S1+ |
| `configs/dataset_flux.toml` | kohya 数据集配置（FLUX：共用图文，batch 与 repeat 不同） | S4 |
| `scripts/train_sdxl.sh` | SDXL 训练命令（H20 优化版） | S1–S3 |
| `scripts/train_flux.sh` | FLUX.1-dev 训练命令 | S4 |
| `scripts/eval_checkpoints.py` | checkpoint 定量评估：色彩贴合、纹理、泄漏率 | S5 |
| `configs/val_prompts.txt` | 固定验证 prompt 集 | S2–S5 |
| **`本机训练评估与成本.md`**（v1.1） | 本机实测环境、10h 窗口排期、时间/电费/磁盘/环境成本、决策矩阵与长跑风险 | 全程 |
| **`configs/dataset_local_sdxl.toml`** | 本机 8GB 数据集配置（batch1 / repeat4 / bucket32 / max1280） | 本机 S1–S5 |
| **`configs/dataset_local_single_palette.toml`** | G4 单配色支消融配置（可选，需自备 `10_mdtr_single` 子集） | 本机 S5 |
| **`scripts/train_local_sdxl.sh`** | 本机单组启动脚本（显存预检、缓存重定向 E 盘、Adafactor、断点续跑、dry-run） | 本机 S1–S5 |
| **`scripts/run_local_ablations.sh`** | 10h+ 无人值守排期：按实测 s/it 自动选档、组级断点续跑、GPU 温度/显存监控 | 本机 S1–S5 |
| **`scripts/setup_local_env.sh`** | 本机环境一键搭建（独立 venv + `.pth` 复用 torch + sd-scripts + 依赖 + 两个 CLIP tokenizer 预缓存 + 自检） | 本机 S1 |
| **`scripts/make_bench_dataset.py`** | 合成 1024² 基准图集生成器，让 S0 之前就能实测 `s/it` | 本机 S1 |
| **`configs/dataset_local_bench.toml`** | benchmark 专用数据集配置（指向合成图，**禁止用于正式训练**） | 本机 S1 |
| **`FLUX_LoRA训练方案.md`** | FLUX 补充方案：三条路径评估、非 gated 模型获取、8GB 探测配方、合规叠加 | S4 / 全程 |
| **`scripts/split_flux_checkpoint.py`** | 一体化 FLUX checkpoint → 纯 DiT + AE（流式、峰值 8MB、含自检；键名 780/780 与 244/244 已校验） | S4 |
| **`scripts/fetch_flux_models.sh`** | 非 gated 三件套下载（约 22.4 GB、断点续传）+ 自动拆分 | S4 |
| **`scripts/train_local_flux.sh`** | 本机 8GB FLUX 探测（提交上限 / 模型 / 显存三重预检、fp8 + block swap 28） | S4 |
| **`scripts/setup_flux_h20.sh`** | H20 FLUX 一键准备：环境体检、sd-scripts、官方 gated 权重 / fp8 降级 | S4 |
| `configs/dataset_local_flux.toml` | 本机 FLUX 数据集配置（768px / batch1 / bucket64） | S4 |
| `configs/dataset_local_flux_bench.toml` | 本机 FLUX 合成图集 200 步探测配置 | S4 |

最终模型卡应包含：底座与版本、触发词、推荐权重（0.6–0.8）、推荐采样器/CFG/分辨率、数据集规模与**来源/授权声明**、已知局限（域外题材、文字、手部）、训练步数与最佳 checkpoint。

---

## 9. 合规与伦理

1. **授权（分支 A 专属）**：归属未确证 ≠ 无版权。每一张下载图仍是受保护作品，台账 `license_status` 一律先记 `unknown`，**不得臆断**。本方案默认用途限定为**个人学习与风格研究**：不发布数据集、不发布 LoRA 权重、不商用、不代客出图、不生成伪造签名或声称原作者作品。若后续要转为可分发 / 商用产品，必须先回到 §2.3 逐张确权并取得书面授权，或改用自有无争议素材重训。
2. **不冒名**：触发词与模型名不使用艺术家姓名；发布时明确标注 `style study / inspired by, non-official, not affiliated`；不得生成伪造签名或声称是原作者作品。
3. **底座许可**：SDXL（Open RAIL++-M）可商用，使用限制会向下游传播；FLUX.1-dev 为非商业许可，商用需 BFL 授权，或改用 FLUX.1-schnell / FLUX.2 klein（Apache 2.0）。LoRA 属衍生作品，底座条款会跟随。
4. **数据集台账**：保留 `dataset_manifest.csv`，记录来源、署名、采集日期、授权状态，便于后续下架请求与追溯。
5. **AI 生成图混入**：若分支 A 溯源后发现大量 AI 生成图，应在模型卡中如实说明训练目标为"网络图集视觉语言"，而非"某艺术家风格"。

---

## 10. 参考来源

**艺术家身份与风格**
- Cool Stories About Art, *10 Artists you need to Know #10*（2026-05-03）：墨西哥插画师 @hangtheguille / Besos Robados — https://www.coolstoriesabout.art/p/10-artists-you-need-to-know-10
- San Diego International Airport, *Guillermo Arias*（Terminal 2 艺术项目）：AFP 摄影记者，@guillermoarias — https://www.san.org/artists/guillermo-arias/
- MutualArt, *Guillermo Arias*：Istanbul Photo Awards 2019 相关 — https://www.mutualart.com/Artist/Guillermo-Arias/748204971562EFF6
- RocketReach, *Guillermo Arias-Camisón Cano*：Codigames 概念艺术家档案 — https://rocketreach.co/guillermo-arias-camison-cano-email_595089472
- 中文二手描述（互见矛盾，仅作为"图集话语"素材）：百度百家号《他的画让人感受到地中海的风》、微信公众号《没上过一天美院，却靠"复古摆烂美学"征服 ZARA》、新浪《Guillermo Arias作品风格解析》《Guillermo Arias油画作品》、微博 @桃乐丝白兰地、腾讯 IMA 收录《艺术家：吉列尔莫·阿里亚斯》

**H20 规格**
- 百度百科 / 互动百科 H20 参数表、AI Wiki *NVIDIA H20*、InferenceBench H20 — https://aiwiki.ai/wiki/nvidia_h20 、 https://inferencebench.io/gpus/nvidia-h20/

**训练方法**
- kohya-ss/sd-scripts README（FLUX.1 LoRA 训练与显存选项、flow matching 参数） — https://github.com/kohya-ss/sd-scripts
- Black Forest Labs / FLUX.2 klein LoRA 官方指南（**caption 只写内容、不写风格**；早停看样本不看 loss） — https://cdn-avatars.qwak.ai/blog/black-forest-labs/flux-2-klein-lora
- Sozee, *How To Train FLUX.1-dev LoRA Models*（dim/lr/steps 与三件套） — https://sozee.ai/resources/best-flux-dev-lora-trainer
- sanj.dev, *LoRA Training 2026: kohya_ss, FLUX & VRAM Optimization Guide*（LoRA+、bf16、gradient checkpointing 代价）
- Apatero, *How to Train a LoRA Locally for SDXL*（正则化图像策略、过拟合信号） — http://apatero.com/blog/train-lora-sdxl-amd-gpu-guide-2025
- LocalAIMaster / PromptZone / UncensoredHub：SDXL Open RAIL++-M vs FLUX.1-dev 非商业 vs schnell Apache 2.0 许可对比

---

## 附录 A：分支 A Caption 速查（完整手册见 `configs/caption_rules_branchA.md`）

覆盖五个题材桶、两支配色支与域外锚点。**只写内容，风格全部留给触发词承载**：

```
# person_outdoor（40%）
mdtr_style retro_muted, a woman reading a book on a seaside terrace, striped chair, potted plant, white wall, strong sunlight from the left
mdtr_style candy_bright, a woman watering plants in a courtyard, green watering can, tiled floor, dappled tree shadows

# person_indoor（20%）
mdtr_style retro_muted, a woman sitting at a table by an open window, a cup and an open book, half-open shutters, light falling on the wall

# landscape（20%）
mdtr_style, a white house with a blue door against a pale sky, dry grass, a single tree, hard midday shadows

# still_life（10%）
mdtr_style retro_muted, a vase of flowers on a wooden table, scattered fruit, a folded cloth, window light
mdtr_style candy_bright, a ceramic bowl with oranges, a striped napkin, a glass of water, plain background

# ood_anchor（10%，防内容泄漏）
mdtr_style, a red bicycle leaning against a white wall, midday sun
mdtr_style, a night market in Tokyo, lanterns, crowd, wet pavement
```

三条铁律：`mdtr_style` 固定首位；子标签必须整体写 `retro_muted` / `candy_bright`（不可拆成 `retro`）；不写媒介、技法、色彩、情绪、评价词。

> 分支 B / C 的模板在 v1.0 中提供，因用户已锁定分支 A，**v1.1 起不再维护**；如需切换分支，按 §3.5 的原则重建词表即可，训练框架与 H20 配置无需改动。

## 附录 B：环境依赖

```
Python 3.10+ / CUDA 12.x / PyTorch 2.x（cu121+）
kohya-ss/sd-scripts（固定 commit，记录到模型卡）
accelerate, transformers, diffusers, safetensors, bitsandbytes(AdamW8bit), xformers(可选), tensorboard
数据脚本：Pillow（仅此一项，纯标准库实现，无需 numpy / cv2 / imagehash）
可选指标：torch + open_clip（仅 `--clip` 时需要；缺失会自动跳过，不影响其余指标）
推理/评估：ComfyUI（SDXL + FLUX 工作流）或 diffusers
```

开训当天先执行：`nvidia-smi -q | head -60` 记录实际显存/驱动/CUDA 版本，再跑 §5.4 的 200 步 benchmark。

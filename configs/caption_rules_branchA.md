# 分支 A · Caption 标注手册

> 配套主方案 §3.5 / §2.2 / §2.4｜触发词 `mdtr_style`｜适用：中文社媒"地中海 / 海边阅读 / 厚涂肌理"图集
> 工具链：`scripts/init_dataset.py` 生成骨架 → 人工按本手册填满 → `scripts/audit_dataset.py` + `scripts/audit_manifest.py` 双重校验

---

## 0. 三条铁律

1. **`mdtr_style` 固定在首位**。不启用 `--shuffle_caption`；若日后开启，必须 `--keep_tokens=1`。
2. **只写"画面里有什么"**：主体 + 姿态/动作 + 物件 + 场景 + 光线方位 + 构图。绝不写"这是什么风格"。
   LoRA 学到的是**提示词没有解释的那部分**——你在 caption 里写了 "oil painting"，模型就认为风格由文字控制，触发词立刻失效。
3. **只给"希望在推理时控制"的维度打标签**。分支 A 只开放一个维度：`retro_muted` / `candy_bright`。其余一律不写。

---

## 1. Caption 结构

```
mdtr_style [retro_muted|candy_bright], <主体>, <姿态/动作>, <物件>, <场景>, <光线方位>, <构图补充>
```

| 约束 | 规则 |
| --- | --- |
| 语言 | 英文，全小写，逗号分隔的名词短语片段；不要写完整句子 |
| 长度 | **8–25 词**（少于 8 词会被 `audit_dataset.py` 判"内容描述不足"） |
| 顺序 | 触发词 → 子标签 → 主体 → 物件 → 场景 → 光线；同一批图保持语序一致 |
| 人称 | 用 `a woman` / `a figure`，不要用名字、不要描述长相细节（分支 A 弱化五官） |
| 光线 | 写**方位与性质**：`strong sunlight from the left`、`dappled tree shadows`、`window light`；不写 `warm glow`、`soft light` 之类氛围词 |
| 多个主体 | 用 `two figures` 并说明相对位置，不要分别命名 |

**必须写全场景词**。分支 A 的"海边 + 女性 + 阅读"高度重复，若 caption 偷懒只写 `a woman`，模型会把题材当风格学进去，换 prompt 也只画海边女人（内容泄漏）。宁可啰嗦，不可省略。

---

## 2. 配色子标签：`retro_muted` / `candy_bright`

资料里"低饱和莫兰迪"与"明亮糖果色"两派描述不必二选一，把它们变成推理旋钮。

| 条件 | 标签 | 视觉描述 |
| --- | --- | --- |
| `audit_report.csv` 中 `mean_s < 0.35` | `retro_muted` | 米白、灰粉、橄榄绿、淡蓝为主，小面积高亮点色 |
| `mean_s ≥ 0.35` | `candy_bright` | 橙、蓝、绿、粉并置，饱和度高但不刺眼 |
| 人眼与数值冲突 | **以人眼为准** | 把 `mean_s` 与该判断记进 `sources.csv` 的 `notes` |
| 暂时判不了 | 台账填 `undecided` | S0 门禁会拦下，**必须开训前解决** |

**配额目标 60 : 40，容差 ±15pp。** 推理时 `mdtr_style retro_muted, a night market in Tokyo` 即可把风格迁移到全新题材并指定配色支。

> ⚠️ 子标签必须**整体写** `retro_muted` / `candy_bright`。拆成 `retro`、`muted`、`bright` 会被 `audit_dataset.py` 的风格词黑名单拦截。校验脚本会先剔除完整的子标签 token 再扫描黑名单，所以整体写法是安全的。

---

## 3. 题材桶词表（与 `sources.csv` 的 `subject_bucket` 对齐）

| 桶 | 占比 | 常用英文词（供 BLIP / 视觉大模型初稿后人工校正） |
| --- | --- | --- |
| `person_outdoor` | 40% | `woman`, `figure`, `reading a book`, `sitting on a terrace`, `seaside`, `striped chair`, `courtyard`, `watering plants`, `white wall`, `strong sunlight`, `tree shadows` |
| `person_indoor` | 20% | `woman by a window`, `open shutters`, `table`, `a cup`, `an open book`, `vinyl record`, `guitar`, `half-length portrait`, `plain interior` |
| `landscape` | 20% | `white house`, `blue door`, `pale sky`, `dry grass`, `single tree`, `stone steps`, `empty beach`, `horizon`, `hard midday shadows` |
| `still_life` | 10% | `vase of flowers`, `ceramic bowl`, `scattered fruit`, `folded cloth`, `striped napkin`, `glass of water`, `top-down view`, `wooden table` |
| `ood_anchor` | 10% | `red bicycle`, `night market in Tokyo`, `lanterns`, `wet pavement`, `snowy cabin`, `cat on a windowsill`, `crowd`, `pine trees` |

`ood_anchor` 是**防内容泄漏的关键配额**：域外题材逼模型学风格而非题材，不要为了"风格更纯"把它删掉。

---

## 4. 禁用词黑名单（`audit_dataset.py` 自动拦截）

| 类别 | 禁用词（命中即判 caption 违规） |
| --- | --- |
| 媒介 / 技法 | `oil painting`, `painterly`, `impasto`, `brushstroke`, `brush stroke`, `thick paint`, `textured paint`, `watercolor`, `gouache`, `illustration`, `art style`, `in the style of` |
| 色彩 / 色板 | `muted`, `low saturation`, `desaturated`, `retro`, `vintage`, `morandi`, `pastel`, `vibrant` |
| 氛围 / 情绪 | `aesthetic`, `cozy`, `peaceful`, `serene`, `dreamy`, `nostalgic`, `healing`, `soft`, `gentle`, `warm glow`, `soft light` |
| 评价 | `beautiful`, `stunning`, `elegant`, `masterpiece`, `high quality` |
| 地域 / 署名 | `mediterranean`, `Guillermo Arias`, `inspired by`, `by the artist` |
| 中文对应 | `油画`, `厚涂`, `笔触`, `低饱和`, `复古`, `插画`, `治愈`, `松弛感`, `氛围感`, `高级感`, `温柔`, `慵懒`, `静谧`, `莫兰迪`, `水彩`, `水粉`, `印象派`, `颗粒感`, `画风` |

**判断口诀**：删掉这个词，画面内容还说得清吗？说得清 → 是风格词，删；说不清 → 是内容词，留。

例：`strong sunlight from the left` 删掉后不知道光从哪来 → **留**；`beautiful warm summer light` 里的 `beautiful warm` 删掉不影响内容 → **删**。

---

## 5. 标准范例（照抄结构，不要照抄内容）

```
# person_outdoor · retro_muted
mdtr_style retro_muted, a woman reading a book on a seaside terrace, striped chair, potted plant, white wall, strong sunlight from the left

# person_outdoor · candy_bright
mdtr_style candy_bright, a woman watering plants in a courtyard, green watering can, tiled floor, dappled tree shadows, blue sky

# person_indoor
mdtr_style retro_muted, a woman sitting at a table by an open window, a cup and an open book, half-open shutters, light falling on the wall

# landscape
mdtr_style retro_muted, a white house with a blue door against a pale sky, dry grass, a single tree, hard midday shadows

# still_life
mdtr_style candy_bright, a ceramic bowl with oranges, a striped napkin, a glass of water, plain background, top-down view

# ood_anchor（域外，测泛化）
mdtr_style, a red bicycle leaning against a white wall, midday sun, narrow street
mdtr_style retro_muted, a night market in Tokyo, lanterns, crowd, wet pavement
mdtr_style candy_bright, a cat sleeping on a windowsill, half-open shutters, a plant on the sill
```

> 无子标签的写法（`mdtr_style, ...`）允许少量存在，用于让模型学到"两支配色的公共部分"。但不要把某一支的图大量留空，否则旋钮失效。

---

## 6. 常见错误对照

| ❌ 改前 | 问题 | ✅ 改后 |
| --- | --- | --- |
| `a beautiful oil painting in muted retro colors, thick impasto, a woman on a terrace` | 无触发词 + 全是风格词 | `mdtr_style retro_muted, a woman on a terrace, striped chair, white wall, strong sunlight from the left` |
| `mdtr_style, impressionist painterly illustration, a woman reading` | 风格词会把风格挂到文字上；内容过短 | `mdtr_style, a woman reading a book on a seaside terrace, striped chair, potted plant, strong sunlight` |
| `mdtr_style retro, a vase of flowers` | 子标签被拆开；内容不足 8 词 | `mdtr_style retro_muted, a vase of flowers on a wooden table, scattered fruit, a folded cloth, window light` |
| `mdtr_style, a cozy healing summer afternoon by the sea` | 情绪词污染色板（§7.6 已知失败模式） | `mdtr_style, a woman sitting by the sea, a book on her lap, low stone wall, strong midday sunlight` |
| `mdtr_style, woman`（偷懒） | 场景词缺失 → 题材被当风格 | 写全主体 + 物件 + 场景 + 光线，见 §1 |

---

## 7. 标注工作流（S0）

1. `python scripts/init_dataset.py --sources configs/sources.csv`
   → 生成 `dataset/train/10_mdtr/<id>.<ext>` 与同名 `.txt` 骨架（含 `TODO`），以及 `dataset/dataset_manifest.csv`。
   **脚本默认不覆盖已存在的 caption**，重复运行安全。
2. 用 BLIP / WD-Tagger / 视觉大模型出内容初稿 → 按本手册删风格词、补齐场景词。
3. 逐条把 `.txt` 里的 `TODO` 全部替换掉。**任何残留 TODO 都会被 S0 门禁拦下。**
4. 抽检 100% caption（数据量 90–150 条，全检成本很低）：
   `python scripts/audit_dataset.py --dir dataset/train/10_mdtr --out audit_report.csv`
   要求输出 `caption 需修正 : 0`。
5. `python scripts/audit_manifest.py --sources configs/sources.csv --dataset dataset/train/10_mdtr`
   → 返回码 0 才准进 S1。

**留一份备份**：把定稿 caption 复制到 `dataset/captions/`，后续消融实验（如"触发词+内容 vs 纯触发词"）要能一键还原。

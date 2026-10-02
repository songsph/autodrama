# AutoDrama · 一站式 AI 短剧自动化流水线

输入一个创意 → 输出一集成片。全流程自动化，但在**三个决定成败的节点**留出人工拍板。

```
创意 → 分集大纲 → 角色卡(审核) → 场景卡(审核) → 剧本(审核)
     → 分镜 → 一致性审查 → 配音 → 首帧图 → 视频 → 合成 → 成片
```

---

## 核心理念

### 1. 一致性 = 不断减少模型的自由度

模型自由发挥得越少，人物和场景就越像"同一个人、同一个地方"。

`prompt_builder.py` 把每个镜头的 prompt 按**固定顺序**组装：

```
[角色外貌锚定词] + [当前服装/伤势] + [场景锚定词] + [光照与色调]
+ [景别] + [动作] + [运镜] + [情绪]
 ↑ 前四项是锁死的常量                ↑ 只有这四项随镜头变化
```

### 2. 三个必须由人拍板的审核关卡

| 关卡 | 为什么必须人工 | 确认后发生什么 |
|---|---|---|
| **剧本** | 决定故事走向，改这里最便宜 | 进入分镜阶段 |
| **人物图片** | 决定全剧人物一致性 | 存为 `golden_ref`，成为全剧参考锚点 |
| **场景图片** | 决定空间与光影一致性 | 存为 `establishing_shot`，后续镜头都从它派生 |

这三处一旦确认即被【锁定】，后续所有生成逐字复现，禁止自动改写。

### 3. 跨集连贯靠"世界状态表"

每集结束自动更新 `world`：谁在哪、受了什么伤、穿什么、道具在谁手上、还有哪些伏笔没回收。
写第 20 集时模型依然记得第 7 集"男主左手受伤"，不会让他突然用左手拎重物。

### 4. 短剧即代码

剧本、角色卡、分镜全部落成 JSON + Markdown，**可 git diff、可回滚、可复现**。
改一句台词就是一个 commit。这是把短剧项目放在 GitHub 上的真正意义。

---

## 快速开始（无需 GPU，先跑通流程）

### Windows：一键装到 D 盘（推荐）

```powershell
python scripts/setup_env.py                     # Windows 默认装到 D:\autodrama
python scripts/setup_env.py --root D:\autodrama # 或自定义
```

脚本会在 `D:\autodrama` 下建好 venv、pip 缓存、ffmpeg、模型目录，
**不往 C 盘写任何大文件**，并自动生成 `.env`。之后一律用这个 venv：

```powershell
D:\autodrama\venv\Scripts\python.exe -m autodrama status
```

> 脚本会自动尝试多个 pip 镜像源，ffmpeg 下载带断点续传；
> 若完整包下载失败，会自动回退用 pip 安装 `imageio-ffmpeg` 提供 ffmpeg。

### 其他环境

```bash
pip install -r requirements.txt

# 用 Mock 后端跑通全流程（不调 API、不占显存）
python -m autodrama init --name demo --idea "车祸后，她发现丈夫的记忆里住着另一个女人" --episodes 6
python -m autodrama run -p demo --ep 1 --yes      # --yes 跳过审核关卡
python -m autodrama status -p demo
```

接入 DeepSeek 真实生成：

```bash
# 安全做法：不落盘
read -s DEEPSEEK_API_KEY && export DEEPSEEK_API_KEY
# 或：cp .env.example .env 后填写（.env 已被 .gitignore 屏蔽）

python -m autodrama run -p demo --ep 1            # 去掉 --yes，逐个关卡人工确认
```

产出：

```
projects/demo/
├── project.json            # 全部结构化数据（可 diff）
├── README.md               # 自动生成的分镜表（可直接在 GitHub 浏览）
├── episodes/ep01_script.md
└── assets/{characters,scenes,keyframes,clips,audio}/
outputs/demo/ep01.mp4
```

---

## 在 AutoDL（5090 32G）上部署

```bash
bash setup_autodl.sh        # 装 ffmpeg、依赖、下载模型到数据盘
```

**必须知道的几件事：**

- **按模型批处理，不要逐镜头切换模型**。每次加载权重 1~3 分钟，24 个镜头逐一切就是纯烧钱。
  流水线已按阶段批处理（先出全部首帧图 → 再出全部视频 → 再配音）。
- **无卡开机调试**：写码、改 prompt、审剧本都在无卡模式（约 0.1 元/时）下做，只在跑生成时开 GPU。
- **跑完立刻关机**。关机数据全保留，下次开机 `--resume` 从断点继续。
- **配好环境先存镜像**，别手滑点"释放"（释放 = 全部清空，不可恢复）。
- **存镜像前删掉 `.env`**，否则 API Key 会跟着镜像发布出去。

成本参考（粗估）：一集 2 分钟 ≈ 24 个镜头 ≈ 1.5~3 小时 GPU ≈ 5~10 元。
MVP 阶段建议先用 480p，跑顺了再上 720p。

---

## 接入真实模型

每个渲染后端有三种模式，用环境变量切换（模型半年一换，架构要活下来）：

| 模式 | 说明 | 何时用 |
|---|---|---|
| `mock` | 离线占位图/静音，纯跑流程 | 开发调试、没有 GPU 时 |
| `local` | 内置实现：diffusers + IP-Adapter | 图像生成（已实现） |
| `worker` | 调 `scripts/video_worker.py`，**一次加载模型批量跑完整集** | 视频生成（推荐） |
| `command` | 调用你自己的命令模板 | 视频/配音/口型（适配任意模型） |

### 视频：worker 批量模式（推荐）

视频模型加载一次要 1~3 分钟。逐镜头起一个进程，一集 18 镜就多烧半小时 GPU 计费时间。
所以流水线把所有镜头打包成任务清单，交给 worker 一次加载、逐个生成：

```bash
export AUTODRAMA_VIDEO=worker
export AUTODRAMA_VIDEO_BACKEND=wan     # wan = diffusers 图生视频；h3 = MiniMax H3
export AUTODRAMA_VIDEO_MODEL=/root/autodl-tmp/models/Wan-AI/Wan2.2-I2V-A14B
```

`--backend h3` 走官方推理脚本（其接口经常变），可用命令模板覆盖：

```bash
export AUTODRAMA_H3_DIR=/root/autodl-tmp/repos/MiniMax-H3
export AUTODRAMA_H3_CMD='python $AUTODRAMA_H3_DIR/infer.py --image "{image}" --prompt "{prompt}" --out "{out}"'
```

AutoDL 上一键装好（含模型下载）：

```bash
bash setup_autodl.sh --gpu --video wan     # 或 --video h3
```

> H3 的 FP16 全量权重约 65G，32G 显存放不下，请选择 fp8/量化变体或开启 CPU offload；
> 想先跑通流程，建议用 `wan`（Wan2.2 I2V），接口稳定。

### 图像：local 模式

```bash
export AUTODRAMA_IMAGE=local
export AUTODRAMA_IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
export AUTODRAMA_IP_SCALE=0.85              # 角色参考图注入强度
export AUTODRAMA_SCENE_INIT_STRENGTH=0.35   # 用定场图做 img2img 底图
```

两条一致性通路在这里落地：
- **角色** → IP-Adapter 注入黄金参考图，保证"是同一个人"
- **场景** → 定场图当 img2img 底图，保证"是同一个地方、同样的光"

### 视频 / 配音 / 口型：command 模式

不用改代码，配一条命令模板即可接入任意模型：

```bash
export AUTODRAMA_VIDEO=command
export AUTODRAMA_VIDEO_CMD='python /root/h3/infer.py --img "{keyframe}" --last "{last_frame}" --prompt "{prompt}" --dur {duration} --out "{out}"'

export AUTODRAMA_TTS=command
export AUTODRAMA_TTS_CMD='python /root/CosyVoice/infer.py --text "{text}" --voice "{voice}" --emotion "{emotion}" --out "{out}"'
```

可用占位符：`{prompt} {negative} {char_ref} {scene_ref} {width} {height} {seed} {out}`
`{keyframe} {last_frame} {duration} {text} {voice} {emotion} {clip} {audio}`

| 环节 | 推荐模型 |
|---|---|
| 剧本/分镜 | DeepSeek（`llm.py` 已实现） |
| 首帧图 | FLUX / SDXL + IP-Adapter、PuLID |
| 视频 | MiniMax H3 / Wan2.2（用首尾帧 FLF2V） |
| 配音 | CosyVoice3 / IndexTTS-2 / Qwen3-TTS |
| 口型 | MuseTalk / LatentSync（可选） |
| 合成 | FFmpeg（`compose.py` 已实现） |

三个关键建议：
1. **图生视频优先**：先出静帧首帧图，再让它动起来。比纯文生视频一致性高一个量级。
2. **首尾帧串联**：`stage_video` 已自动把下一镜头的首帧作为本镜头的收束帧传入，衔接更自然。
3. **32G 显存装不下所有模型**，务必用独立进程逐阶段运行（每个阶段跑完即释放显存）。

---

## 命令一览

| 命令 | 作用 |
|---|---|
| `init` | 创建项目 |
| `outline` | 生成分集大纲 |
| `characters` | 生成角色卡 + 生成形象图（**审核点**） |
| `scenes` | 生成场景卡 + 生成定场图（**审核点**） |
| `script --ep N` | 写第 N 集剧本（**审核点**） |
| `shots --ep N` | 剧本拆分镜 |
| `check --ep N` | 一致性审查（找出矛盾与雷区） |
| `tts --ep N` | 配音，并以音频时长锁定镜头时长 |
| `keyframes --ep N` | 批量生成首帧静帧图 |
| `video --ep N` | 批量生成视频片段 |
| `compose --ep N` | 合成成片 + 更新世界状态与前情摘要 |
| `run --ep N --until X` | 自动跑到指定阶段 |
| `status` | 查看进度 |
| `ui` | 启动 Gradio 界面（审候选图、逐阶段运行） |

`keyframes` 支持 `--candidates N`：每个镜头出 N 张候选，用 face 相似度
自动挑最像角色的一张（需 `pip install -r requirements-gpu.txt`，未装则自动降级）。

每个阶段幂等且可单独重跑：**删掉不满意的产物文件，重跑该阶段即可**。

---

## 目录结构

```
autodrama/
├── consistency.py       # 一致性铁律（注入每一次 LLM 调用）
├── prompts.py           # 各阶段提示词 + 锁定资产/前情记忆注入
├── schema.py            # 数据模型（角色卡/场景卡/分镜/世界状态）
├── prompt_builder.py    # prompt 组装器 —— 一致性在这里落地
├── llm.py               # DeepSeek + Mock 兜底
├── providers.py         # 图像/视频/配音/口型 抽象层
├── stages.py            # 流水线各阶段（幂等、可断点续跑）
├── compose.py           # FFmpeg 合成（字幕/调色/音画对齐）
├── review.py            # 三个人工审核关卡
└── store.py             # 项目存储 + Markdown 导出
```

---

## 诚实的预期

| 场景 | 能做到吗 |
|---|---|
| 肩以上特写、近景对话 | ✅ 一致性很好 |
| 正反打 | ✅ 可以 |
| 全身动作、打斗 | ❌ 容易崩 |
| 三人以上同框 | ⚠️ 会串脸 → 拆成单人镜头 + 剪辑 |
| 单镜头 >5 秒 | ⚠️ 越长越崩 |

`consistency.py` 里内置了**技术规避清单**，模型写分镜时会主动绕开这些雷区。

目标定在"观众一眼认得出是同一个人"，而不是逐帧像素级一致。

---

## 安全提醒

- API Key 只走环境变量，**绝不写进代码**，`.env` 已被 `.gitignore` 屏蔽
- Key 一旦进过公开仓库，唯一正确的处理是**作废重新生成**，删文件没用（git 历史里还在）
- Gradio 若要暴露到公网必须加 `auth=()`，否则别人会用你的 GPU 和 Key
- 商用请注意各模型许可证（优先 Apache 2.0 / MIT），音色克隆不要用真人明星声音

---

## 路线图

- [x] 数据模型 + 一致性铁律 + 提示词体系
- [x] DeepSeek 接入 + 三审核关卡 + 分阶段流水线
- [x] FFmpeg 合成（字幕/调色/音画对齐）
- [x] 接入真实模型：图像 `local`（diffusers + IP-Adapter）、视频/配音/口型 `command` 模板
- [x] 镜头候选自动生成 + 人脸相似度自动选优
- [x] Gradio 可视化界面
- [x] Windows D 盘一键安装脚本
- [x] AutoDL 一键脚本：视频模型下载（Wan2.2 / MiniMax H3）+ 批量 worker
- [ ] 角色 LoRA 训练脚本
- [ ] BGM / 音效自动生成
- [ ] 视频片段自动打分（清晰度/闪烁/形变检测）

---

## License

MIT

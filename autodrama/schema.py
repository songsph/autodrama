"""AutoDrama 数据模型。

设计原则：一切中间产物都可序列化成 JSON / Markdown，可 git diff、可回滚、可复现。
带【锁定】标记的字段一旦人工确认，就禁止流水线自动改写 —— 这是一致性的根基。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# 允许的景别（限定枚举，避免模型自由发挥导致镜头语言失控）
SHOT_SIZES = [
    "大远景", "远景", "全景", "中景", "中近景", "近景", "特写", "大特写", "过肩", "POV",
]

# 允许的运镜
CAMERA_MOVES = [
    "固定", "缓慢推进", "缓慢拉远", "左摇", "右摇", "上摇", "下摇",
    "手持跟随", "环绕", "升降", "快速推近", "甩镜",
]

# 视觉生成的技术雷区（用于 prompt 生成时的自动规避提示）
TECH_RISKS = [
    "三人及以上同框（容易串脸）",
    "复杂手部动作（手指会崩）",
    "打斗、舞蹈等大幅度全身运动",
    "画面内出现文字、招牌、手机屏幕内容（AI 画不好字）",
    "大幅换装或发型突变",
    "超过 5 秒的长镜头（越长越易崩）",
]


class Character(BaseModel):
    """角色卡：人物一致性的唯一真相来源。"""
    id: str
    name: str
    role: str = "配角"
    identity: str = ""                        # 身份/职业/性格一句话
    appearance: str = ""                      # 【锁定】外貌锚定词，逐字复现进每个镜头
    negative: str = ""                        # 【锁定】负面词
    wardrobe: str = ""                        # 【锁定】默认服装造型
    golden_ref: Optional[str] = None          # 黄金参考图路径（人工选定）
    ref_images: List[str] = Field(default_factory=list)
    lora: Optional[str] = None                # 角色 LoRA（可选，主角专用）
    lora_weight: float = 0.8
    voice_id: Optional[str] = None            # TTS 音色 ID
    voice_sample: Optional[str] = None        # 音色克隆样本
    seed: int = 12345
    approved: bool = False                    # 人工审核关卡


class Scene(BaseModel):
    """场景卡：场景一致性的唯一真相来源。"""
    id: str
    name: str
    description: str = ""                     # 【锁定】场景锚定词
    time_of_day: str = ""                     # 清晨/正午/黄昏/深夜
    weather: str = ""
    light: str = ""                           # 光源方向与性质（决定光影一致）
    color_tone: str = ""                      # 色调（决定后期调色）
    negative: str = ""
    establishing_shot: Optional[str] = None   # 定场图：后续镜头都从它派生
    seed: int = 12345
    approved: bool = False                    # 人工审核关卡


class DialogueLine(BaseModel):
    character: str
    text: str
    emotion: str = "平静"
    audio: Optional[str] = None
    duration: float = 0.0


class Shot(BaseModel):
    """一个分镜 = 一个视频片段。"""
    id: str                                   # E01-S01
    index: int = 0
    scene_id: str = ""
    characters: List[str] = Field(default_factory=list)
    shot_size: str = "中景"
    camera: str = "固定"
    action: str = ""
    dialogue: List[DialogueLine] = Field(default_factory=list)
    duration: float = 3.0

    # ---- 生成产物 ----
    image_prompt: str = ""
    video_prompt: str = ""
    keyframe: Optional[str] = None            # 首帧静帧图
    candidates: List[str] = Field(default_factory=list)   # 候选图（用于自动选优）
    clip: Optional[str] = None                # 视频片段
    audio: Optional[str] = None               # 合并后的片段音频
    status: str = "pending"                   # pending / keyframe / clip / approved / rejected
    note: str = ""


class Episode(BaseModel):
    index: int
    title: str = ""
    synopsis: str = ""
    script_md: str = ""                       # 可读剧本（Markdown）
    script_lines: List[Dict[str, Any]] = Field(default_factory=list)  # 结构化台词单元
    shots: List[Shot] = Field(default_factory=list)
    summary: str = ""                         # 供下一集使用的前情摘要
    state_delta: Dict[str, Any] = Field(default_factory=dict)
    script_approved: bool = False             # 人工审核关卡：剧本
    status: str = "pending"                   # pending/scripted/shotted/keyframed/clipped/composed


class WorldState(BaseModel):
    """世界状态表：跨集连贯性的核心。每集结束后增量更新。"""
    timeline: str = "Day 1 早晨"
    characters: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    props: Dict[str, str] = Field(default_factory=dict)
    hooks: List[str] = Field(default_factory=list)      # 未回收的伏笔
    facts: List[str] = Field(default_factory=list)      # 已确立的关键事实


class Project(BaseModel):
    name: str
    title: str = ""
    logline: str = ""
    idea: str = ""
    genre: str = ""
    style: str = ""
    total_episodes: int = 6
    outline: str = ""
    characters: List[Character] = Field(default_factory=list)
    scenes: List[Scene] = Field(default_factory=list)
    episodes: List[Episode] = Field(default_factory=list)
    world: WorldState = Field(default_factory=WorldState)
    stage: str = "init"
    created_at: str = ""

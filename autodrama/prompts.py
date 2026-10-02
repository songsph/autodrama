"""提示词体系。

所有 LLM 调用都会自动注入 consistency.CONSISTENCY_RULES，
并把【已锁定的角色卡 / 场景卡 / 世界状态 / 前情摘要】拼进上下文，
让模型在"约束充分"的前提下创作，而不是自由发挥。
"""

from __future__ import annotations

import json
from typing import List

from .consistency import CONSISTENCY_RULES, REVIEW_CHECKLIST
from .schema import Project

SYSTEM_ROLE = """你是中国竖屏短剧行业的资深总编剧兼导演，擅长强钩子、快节奏、情绪浓度高的短剧创作。
你的产出必须可直接用于 AI 画面生成，因此必须严格遵守下面的一致性铁律与技术规避清单。
只输出要求的 JSON，不要输出任何解释性文字、Markdown 代码块符号或多余内容。"""


def _locked_context(project: Project) -> str:
    """把已锁定的资产拼成上下文，注入每一次创作调用。"""
    parts: List[str] = ["# 项目设定"]
    parts.append(f"- 剧名：{project.title}")
    parts.append(f"- 一句话梗概：{project.logline}")
    parts.append(f"- 原始创意：{project.idea}")
    parts.append(f"- 类型：{project.genre} / 风格：{project.style}")
    parts.append(f"- 总集数：{project.total_episodes}")

    if project.characters:
        parts.append("\n# 已锁定角色卡（外貌描述必须逐字复现，禁止改写）")
        for c in project.characters:
            parts.append(
                f"- {c.id}｜{c.name}（{c.role}）\n"
                f"  外貌：{c.appearance}\n"
                f"  默认服装：{c.wardrobe}\n"
                f"  负面词：{c.negative}"
            )

    if project.scenes:
        parts.append("\n# 已锁定场景卡（场景描述必须逐字复现，禁止改写）")
        for s in project.scenes:
            parts.append(
                f"- {s.id}｜{s.name}\n"
                f"  描述：{s.description}\n"
                f"  时间：{s.time_of_day}｜天气：{s.weather}\n"
                f"  光照：{s.light}\n"
                f"  色调：{s.color_tone}"
            )

    parts.append("\n# 世界状态表（不可违背）")
    parts.append(json.dumps(project.world.model_dump(), ensure_ascii=False, indent=2))
    return "\n".join(parts)


def _memory_context(project: Project, ep_index: int) -> str:
    """前情记忆：滚动摘要 + 上一集结尾。"""
    parts: List[str] = []
    sums = [e.summary for e in project.episodes if e.index < ep_index and e.summary]
    if sums:
        parts.append("# 前情摘要（前面所有已写集）")
        for i, s in enumerate(sums, 1):
            parts.append(f"- 第{i}集：{s}")
    prev = next((e for e in project.episodes if e.index == ep_index - 1), None)
    if prev and prev.shots:
        tail = prev.shots[-3:]
        parts.append("\n# 上一集结尾（本集必须无缝接续）")
        for sh in tail:
            d = sh.dialogue[0].text if sh.dialogue else "（无台词）"
            parts.append(f"- {sh.id}｜{sh.action}｜台词：{d}")
    if project.world.hooks:
        parts.append("\n# 未回收的伏笔（本集有机会就呼应）")
        for h in project.world.hooks:
            parts.append(f"- {h}")
    return "\n".join(parts)


def system_prompt(project: Project, ep_index: int = 0) -> str:
    return "\n\n".join([
        SYSTEM_ROLE,
        CONSISTENCY_RULES,
        _locked_context(project),
        _memory_context(project, ep_index) if ep_index else "",
    ])


# ------------------------------------------------------------------ 大纲

def build_outline_prompt(project: Project) -> str:
    return f"""请为下面这个创意设计一部 {project.total_episodes} 集的短剧大纲。

创意：{project.idea}
类型：{project.genre or '都市情感'}
风格：{project.style or '现实主义，偏冷色调'}

要求：
- 每集 1~2 分钟，节奏快，每集结尾必须留强钩子（悬念/反转/情绪爆点）
- 第 1 集必须在前 15 秒内抛出核心冲突
- 全剧有清晰的起承转合，最后一集完成所有伏笔回收
- 场景总数控制在 3~6 个（越少越省钱、一致性越好）
- 主要角色控制在 2~4 人（越少越不容易串脸）

严格输出如下 JSON：
{{
  "title": "剧名",
  "logline": "一句话梗概",
  "episodes": [
    {{"index": 1, "title": "集标题", "synopsis": "本集发生什么（80字内）", "hook": "结尾钩子"}}
  ]
}}"""


# ------------------------------------------------------------------ 角色卡

def build_character_prompt(project: Project) -> str:
    return f"""根据大纲，为这部剧设计角色卡。大纲如下：

{project.outline}

角色外貌描述词的写法要求（极其重要，直接决定 AI 出图是否像同一个人）：
- 必须包含：年龄段 + 性别/气质 + 脸型 + 发型发色 + 眉眼特征 + 鼻唇特征 + 体型 + 标志性服装 + 标志性配饰
- 用客观、可视觉化的名词，禁用"英俊""美丽""有气质"这类抽象形容词
- 不同角色之间的外貌特征必须差异明显（发色、发型、脸型、服装至少 3 项不同），防止 AI 串脸
- 全中文，控制在 60~100 字
- 同时给出该角色的负面词（要防止出现的特征）

严格输出如下 JSON：
{{
  "characters": [
    {{
      "id": "拼音_id，如 lin_wan",
      "name": "中文名",
      "role": "主角/配角/反派",
      "identity": "身份与性格一句话",
      "appearance": "外貌锚定词（60~100字，纯视觉描述）",
      "wardrobe": "默认服装造型（含颜色与材质）",
      "negative": "负面词"
    }}
  ]
}}"""


# ------------------------------------------------------------------ 场景卡

def build_scene_prompt(project: Project) -> str:
    return f"""根据大纲，为这部剧设计 3~6 个场景卡。大纲如下：

{project.outline}

已有人物：{', '.join(c.name for c in project.characters)}

场景描述词的写法要求：
- 必须包含：空间类型 + 方位布局（门窗/家具在哪一侧）+ 主要陈设 + 材质质感 + 环境细节
- 必须明确【光源方向与性质】（如"黄昏暖光从画面右侧窗户斜射"），这是光影一致的关键
- 必须明确【色调】（如"冷青色调"），这是后期统一调色的依据
- 全中文，控制在 60~100 字，纯视觉描述

严格输出如下 JSON：
{{
  "scenes": [
    {{
      "id": "S01",
      "name": "场景名",
      "description": "场景锚定词（60~100字）",
      "time_of_day": "清晨/正午/黄昏/深夜",
      "weather": "晴/阴/雨/雪",
      "light": "光源方向与性质",
      "color_tone": "色调描述",
      "negative": "负面词"
    }}
  ]
}}"""


# ------------------------------------------------------------------ 分集剧本

def build_script_prompt(project: Project, ep_index: int) -> str:
    ep = next((e for e in project.episodes if e.index == ep_index), None)
    outline_line = ""
    if ep:
        outline_line = f"\n本集大纲：{ep.title}｜{ep.synopsis}"
    return f"""请写第 {ep_index} 集的完整剧本。{outline_line}

要求：
- 全剧共 {project.total_episodes} 集，这是第 {ep_index} 集
- 本集时长 1~2 分钟，约 12~20 个台词单元
- 短剧节奏：开头 10 秒内必须抓住人，结尾必须有钩子
- 台词口语化、信息密度高，单句不超过 25 字（便于配音与字幕）
- 严格遵守世界状态表，不得让已受伤/已离场角色做出违背状态的举动
- 场景只能从已锁定的场景卡中选择
- 每个台词单元标注：场景 + 角色 + 台词 + 情绪

严格输出如下 JSON：
{{
  "title": "本集标题",
  "synopsis": "本集梗概",
  "script": [
    {{"scene_id": "S01", "character": "林晚", "text": "台词", "emotion": "压抑的决绝"}}
  ]
}}"""


# ------------------------------------------------------------------ 分镜

def build_shots_prompt(project: Project, ep_index: int) -> str:
    ep = next((e for e in project.episodes if e.index == ep_index), None)
    script_json = json.dumps(
        ep.script_lines if ep else [], ensure_ascii=False, indent=2,
    )

    return f"""请把第 {ep_index} 集的剧本拆成可执行的分镜表。

剧本：
{script_json}

可选景别：大远景/远景/全景/中景/中近景/近景/特写/大特写/过肩/POV
可选运镜：固定/缓慢推进/缓慢拉远/左摇/右摇/上摇/下摇/手持跟随/环绕/升降/快速推近/甩镜

硬性要求（违反即作废）：
1. 【景别决定时长上限，实测硬指标，违反必崩】
   大特写/特写 ≤ 2.5 秒｜近景/中近景 ≤ 4 秒｜中景 ≤ 5 秒｜全景/远景/过肩 ≤ 8 秒
   （人物一运动面部就漂移，特写超过 3 秒角色会变成另一个人。
     不要用长镜头拍完再剪开——后半段的人脸已经漂移了，必须逐段单独生成。）
   台词量按 4~5 字/秒 控制；无台词镜头也要有明确动作
2. 同一场戏的正反打必须保持同一轴线（在 action 里写明"人物位于画面左/右"）
3. 主动规避技术雷区：不得出现三人及以上同框、复杂手部动作、画面内文字、超 5 秒镜头
4. 遇多人对话必须拆成单人镜头 + 正反打
5. 每个场景的第一个镜头必须是能交代空间关系的定场镜头（用全景或远景）
6. action 必须是纯 visual 描述（摄像机能拍到的东西），不要写心理活动
7. 【最重要】action 里禁止重复任何外貌、服装、身份、场景环境描述
   ——角色外貌与场景环境由系统自动注入画面 prompt，你写了就是浪费篇幅且会导致输出被截断。
   action 只写三件事：谁在画面哪一侧 + 做什么动作 + 视线或运动方向。每条不超过 45 字。
8. 输出总量控制：全场 12~18 个镜头，不要为了"完整"而堆砌镜头

严格输出如下 JSON：
{{
  "shots": [
    {{
      "scene_id": "S01",
      "characters": ["lin_wan"],
      "shot_size": "中景",
      "camera": "缓慢推进",
      "action": "林晚站在画面右侧，右手紧握栏杆，回头看向画左，雨水顺下颌滑落",
      "duration": 3.5,
      "dialogue": [{{"character": "林晚", "text": "这次我不会再让你走。", "emotion": "压抑的决绝"}}]
    }}
  ]
}}"""


# ------------------------------------------------------------------ 状态更新 / 摘要 / 审查

def build_state_delta_prompt(project: Project, ep_index: int) -> str:
    ep = next((e for e in project.episodes if e.index == ep_index), None)
    shots_desc = "\n".join(
        f"- {sh.id}｜{sh.action}｜" + (sh.dialogue[0].text if sh.dialogue else "")
        for sh in (ep.shots if ep else [])
    )
    return f"""第 {ep_index} 集已经写完，分镜如下：

{shots_desc}

当前世界状态：
{json.dumps(project.world.model_dump(), ensure_ascii=False, indent=2)}

请输出本集造成的【世界状态变更】与【前情摘要】。

要求：
- state_delta 只写发生变化的部分，格式与现有状态一致
- 角色状态至少覆盖：位置、伤势、服装、情绪、与其他角色的关系
- 新增/回收的伏笔要写进 hooks_add / hooks_remove
- timeline 必须推进（写出本集结束时的时间点）
- summary 是给下一集看的，120 字以内，只写会影响后续剧情的关键事实

严格输出如下 JSON：
{{
  "timeline": "Day 3 22:40 雨",
  "characters": {{"lin_wan": {{"位置": "天台", "伤势": "左手掌擦伤", "服装": "米白风衣下摆湿透", "情绪": "决绝"}}}},
  "props": {{"怀表": "林晚持有，已摔裂"}},
  "hooks_add": ["陈默其实看见了那一幕"],
  "hooks_remove": ["林晚为何深夜出门"],
  "facts_add": ["林晚决定第二天去自首"],
  "summary": "本集摘要（120字内）"
}}"""


def build_consistency_check_prompt(project: Project, ep_index: int) -> str:
    ep = next((e for e in project.episodes if e.index == ep_index), None)
    shots_desc = "\n".join(
        f"- {sh.id}｜场景{sh.scene_id}｜景别{sh.shot_size}｜{sh.action}｜"
        + (sh.dialogue[0].text if sh.dialogue else "")
        for sh in (ep.shots if ep else [])
    )
    return f"""请审查第 {ep_index} 集的分镜表，找出一切破坏一致性与连贯性的问题。

分镜表：
{shots_desc}

{REVIEW_CHECKLIST}

严格输出如下 JSON：
{{
  "issues": [
    {{"shot_id": "E01-S07", "level": "严重/警告", "problem": "问题描述", "suggestion": "修改建议"}}
  ]
}}"""

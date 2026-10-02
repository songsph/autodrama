"""Prompt 组装器 —— 一致性在这里落地。

组装顺序严格固定（顺序本身会影响生成权重）：
    [角色外貌锚定词] + [当前服装/伤势状态] + [场景锚定词] + [光照与色调]
    + [景别] + [人物动作] + [运镜] + [画面情绪]

前四项是"锁死的常量"，只有后四项随镜头变化 —— 自由度被压到最小，
人物与场景的一致性因此大幅提升。
"""

from __future__ import annotations

from typing import List, Optional

from .consistency import DEFAULT_NEGATIVE, QUALITY_SUFFIX
from .schema import Character, Project, Scene, Shot


def _char_by_id(proj: Project, cid: str) -> Optional[Character]:
    return next((c for c in proj.characters if c.id == cid or c.name == cid), None)


def _scene_by_id(proj: Project, sid: str) -> Optional[Scene]:
    return next((s for s in proj.scenes if s.id == sid), None)


def _char_state(proj: Project, char: Character) -> str:
    """从世界状态表取角色当前的服装/伤势，保证跨集连续（如"左手包扎"）。"""
    st = proj.world.characters.get(char.id, {})
    if not st:
        return ""
    parts = []
    for k in ("服装", "伤势", "位置", "情绪"):
        v = st.get(k)
        if v and str(v) not in ("无", ""):
            parts.append(f"{k}：{v}")
    return "，".join(parts)


def build_image_prompt(proj: Project, shot: Shot) -> tuple[str, str]:
    """返回 (positive_prompt, negative_prompt)。"""
    seg: List[str] = []

    # 1. 角色外貌锚定词（逐字复现，禁止改写）
    for cid in shot.characters:
        c = _char_by_id(proj, cid)
        if not c:
            continue
        seg.append(c.appearance)
        state = _char_state(proj, c)
        if state:
            seg.append(state)
        # 状态表里若明确给了服装，优先用状态表的（可能已经换装/破损）
        wardrobe = proj.world.characters.get(c.id, {}).get("服装") or c.wardrobe
        if wardrobe:
            seg.append(f"穿着{wardrobe}")

    # 2. 场景锚定词
    sc = _scene_by_id(proj, shot.scene_id)
    if sc:
        seg.append(sc.description)
        if sc.time_of_day:
            seg.append(sc.time_of_day)
        if sc.weather:
            seg.append(sc.weather)
        if sc.light:
            seg.append(sc.light)
        if sc.color_tone:
            seg.append(sc.color_tone)

    # 3. 镜头变量（唯一允许变化的部分）
    seg.append(f"{shot.shot_size}构图")
    if shot.action:
        seg.append(shot.action)
    seg.append(f"运镜：{shot.camera}")
    if shot.dialogue:
        seg.append(f"角色正在说台词，情绪：{shot.dialogue[0].emotion}")

    seg.append(QUALITY_SUFFIX)

    # 4. 负面词合并：默认 + 角色 + 场景
    neg = [DEFAULT_NEGATIVE]
    for cid in shot.characters:
        c = _char_by_id(proj, cid)
        if c and c.negative:
            neg.append(c.negative)
    if sc and sc.negative:
        neg.append(sc.negative)

    return "，".join(x for x in seg if x), "，".join(neg)


def build_video_prompt(proj: Project, shot: Shot) -> str:
    """视频 prompt 只描述"运动"，画面内容由首帧图决定（关键：不让模型自由发挥）。"""
    sc = _scene_by_id(proj, shot.scene_id)
    seg: List[str] = []
    if shot.action:
        seg.append(shot.action)
    seg.append(f"镜头运动：{shot.camera}")
    if shot.dialogue:
        seg.append(f"人物口型在说话，表情：{shot.dialogue[0].emotion}")
    if sc and sc.light:
        seg.append(f"保持光照一致：{sc.light}")
    seg.append("画面稳定，无闪烁，无变形，电影质感")
    return "，".join(seg)


def ref_images_for(proj: Project, shot: Shot) -> tuple[List[str], Optional[str]]:
    """收集本镜头要用的参考图。

    返回 (角色黄金参考图列表, 场景定场图)：
    - 角色图 → 交给 IP-Adapter 注入身份，保证"是同一个人"
    - 定场图 → 交给 img2img 当底图，保证"是同一个地方、同样的光"
    """
    char_refs: List[str] = []
    for cid in shot.characters:
        c = _char_by_id(proj, cid)
        if c and c.golden_ref:
            char_refs.append(c.golden_ref)
    sc = _scene_by_id(proj, shot.scene_id)
    scene_ref = sc.establishing_shot if (sc and sc.establishing_shot) else None
    return char_refs, scene_ref

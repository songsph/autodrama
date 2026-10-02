"""流水线各阶段。每个阶段幂等、可单独重跑、支持断点续跑。

执行顺序（务必按模型批处理，不要逐镜头切换模型，否则加载权重的时间全是浪费）：
    outline → characters → scenes → script → shots → tts → keyframes → video → compose
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from . import compose, prompt_builder, prompts, providers, review, select, store
from .config import OUTPUT_DIR, cfg
from .llm import get_llm
from .schema import Character, DialogueLine, Episode, Project, Scene, Shot


def _print(msg: str) -> None:
    print(msg)


def _safe_image(fn, *a, **kw):
    """真实模型未接入时不阻塞流程，给出明确提示。"""
    try:
        return fn(*a, **kw)
    except NotImplementedError as e:
        print(f"[skip] {e}")
        return ""


# ------------------------------------------------------------------ 大纲

def stage_outline(proj: Project, auto: Optional[bool] = None) -> None:
    if proj.outline and proj.episodes:
        print("[skip] 大纲已存在")
        return
    llm = get_llm()
    data = llm.chat_json(prompts.system_prompt(proj),
                         prompts.build_outline_prompt(proj), mock_kind="outline")
    proj.title = data.get("title", proj.name)
    proj.logline = data.get("logline", "")
    proj.outline = "\n".join(
        f"第{e['index']}集《{e['title']}》{e['synopsis']}｜钩子：{e.get('hook','')}"
        for e in data.get("episodes", [])
    )
    proj.episodes = []
    for e in data.get("episodes", []):
        ep = store.get_episode(proj, int(e["index"]))
        ep.title = e.get("title", "")
        ep.synopsis = e.get("synopsis", "")
        ep.status = "pending"
    proj.stage = "outline"
    store.save(proj)
    _print(f"\n【大纲】{proj.title}｜{proj.logline}")
    review.confirm("审核点：分集大纲", proj.outline, auto=auto)


# ------------------------------------------------------------------ 角色卡 + 人物图

def stage_characters(proj: Project, auto: Optional[bool] = None, candidates: int = 3) -> None:
    if not proj.characters:
        llm = get_llm()
        data = llm.chat_json(prompts.system_prompt(proj),
                             prompts.build_character_prompt(proj), mock_kind="characters")
        proj.characters = [Character(**c) for c in data.get("characters", [])]

    ip = providers.get_image()
    for c in proj.characters:
        if c.approved and c.golden_ref:
            continue
        if not c.appearance:
            continue
        paths: List[str] = []
        for k in range(candidates):
            out = store.asset_path(proj, "characters", f"{c.id}_{k}.png")
            p, n = prompt_builder.build_image_prompt(
                proj, Shot(id=f"{c.id}_ref", scene_id="", characters=[c.id],
                           shot_size="中景", action="正面站立，平视镜头"))
            r = _safe_image(ip.generate, prompt=p, negative=n, char_refs=None,
                            scene_ref=None, width=cfg.width, height=cfg.height,
                            seed=c.seed + k, out=str(out))
            if r:
                paths.append(str(out))
        if not paths:
            continue
        c.ref_images = paths
        idx = review.choose(
            f"审核点：角色「{c.name}」形象图（选定后成为全剧黄金参考图）\n"
            f"外貌锚定词：{c.appearance}", paths, auto=auto)
        if idx == -2:
            c.ref_images = []
            print(f"角色 {c.name} 已标记为重生成，重跑本阶段即可")
            continue
        if idx >= 0:
            c.golden_ref = paths[idx]
            c.approved = True
    proj.stage = "characters"
    store.save(proj)


# ------------------------------------------------------------------ 场景卡 + 定场图

def stage_scenes(proj: Project, auto: Optional[bool] = None, candidates: int = 3) -> None:
    if not proj.scenes:
        llm = get_llm()
        data = llm.chat_json(prompts.system_prompt(proj),
                             prompts.build_scene_prompt(proj), mock_kind="scenes")
        proj.scenes = [Scene(**s) for s in data.get("scenes", [])]

    ip = providers.get_image()
    for s in proj.scenes:
        if s.approved and s.establishing_shot:
            continue
        paths: List[str] = []
        for k in range(candidates):
            out = store.asset_path(proj, "scenes", f"{s.id}_{k}.png")
            desc = "，".join(x for x in [s.description, s.time_of_day, s.weather,
                                         s.light, s.color_tone, "空镜，无人物"] if x)
            r = _safe_image(ip.generate, prompt=desc, negative=s.negative,
                            char_refs=None, scene_ref=None, width=cfg.width,
                            height=cfg.height, seed=s.seed + k, out=str(out))
            if r:
                paths.append(str(out))
        if not paths:
            continue
        idx = review.choose(
            f"审核点：场景「{s.name}」定场图（后续镜头都从它派生）\n{desc}",
            paths, auto=auto)
        if idx >= 0:
            s.establishing_shot = paths[idx]
            s.approved = True
    proj.stage = "scenes"
    store.save(proj)


# ------------------------------------------------------------------ 分集剧本

def stage_script(proj: Project, ep_index: int, auto: Optional[bool] = None,
                 max_retry: int = 3) -> Episode:
    ep = store.get_episode(proj, ep_index)
    llm = get_llm()
    for attempt in range(max_retry):
        data = llm.chat_json(prompts.system_prompt(proj, ep_index),
                             prompts.build_script_prompt(proj, ep_index),
                             mock_kind="script")
        ep.title = data.get("title", ep.title)
        ep.synopsis = data.get("synopsis", ep.synopsis)
        ep.script_lines = data.get("script", [])
        lines = "\n".join(
            f"[{l.get('scene_id')}] {l.get('character')}（{l.get('emotion')}）：{l.get('text')}"
            for l in ep.script_lines
        )
        ep.script_md = f"# 第 {ep_index} 集 {ep.title}\n\n{ep.synopsis}\n\n{lines}\n"
        if review.confirm(f"审核点：第 {ep_index} 集剧本", ep.script_md, auto=auto):
            break
        print(f"剧本被打回，重新生成（第 {attempt + 2} 次）…")
    ep.script_approved = True
    ep.status = "scripted"
    proj.stage = f"script@{ep_index}"
    store.save(proj)
    (store.project_dir(proj.name) / "episodes" / f"ep{ep_index:02d}_script.md").write_text(
        ep.script_md, encoding="utf-8")
    return ep


# ------------------------------------------------------------------ 分镜

def stage_shots(proj: Project, ep_index: int) -> Episode:
    ep = store.get_episode(proj, ep_index)
    if ep.shots:
        print("[skip] 分镜已存在（如需重做请删除后重跑）")
        return ep
    llm = get_llm()
    data = llm.chat_json(prompts.system_prompt(proj, ep_index),
                         prompts.build_shots_prompt(proj, ep_index), mock_kind="shots")
    shots: List[Shot] = []
    for i, s in enumerate(data.get("shots", []), 1):
        sid = f"E{ep_index:02d}-S{i:02d}"
        dlg = [DialogueLine(**d) for d in s.get("dialogue", [])]
        sh = Shot(
            id=sid, index=i, scene_id=s.get("scene_id", ""),
            characters=s.get("characters", []), shot_size=s.get("shot_size", "中景"),
            camera=s.get("camera", "固定"), action=s.get("action", ""),
            dialogue=dlg, duration=float(s.get("duration", 3.0)),
        )
        sh.image_prompt, _ = prompt_builder.build_image_prompt(proj, sh)
        sh.video_prompt = prompt_builder.build_video_prompt(proj, sh)
        shots.append(sh)
    ep.shots = shots
    ep.status = "shotted"
    proj.stage = f"shots@{ep_index}"
    store.save(proj)
    _print(f"[ok] 第 {ep_index} 集分镜 {len(shots)} 个镜头")
    return ep


# ------------------------------------------------------------------ 一致性审查

def stage_check(proj: Project, ep_index: int) -> list:
    ep = store.get_episode(proj, ep_index)
    llm = get_llm()
    data = llm.chat_json(prompts.system_prompt(proj, ep_index),
                         prompts.build_consistency_check_prompt(proj, ep_index),
                         mock_kind="check")
    issues = data.get("issues", [])
    if not issues:
        print("[ok] 一致性审查通过，未发现问题")
    else:
        print(f"[warn] 发现 {len(issues)} 个问题：")
        for it in issues:
            print(f"  - [{it.get('level')}] {it.get('shot_id')}：{it.get('problem')}"
                  f" → {it.get('suggestion')}")
    return issues


# ------------------------------------------------------------------ 配音（定镜头时长）

def _concat_audio(paths: List[Path], out: Path) -> Path:
    lst = out.parent / "audio_list.txt"
    lst.write_text("\n".join(f"file '{p.as_posix()}'" for p in paths), encoding="utf-8")
    import subprocess
    # 必须用 cfg.ffmpeg：Windows 整合包通常不进 PATH，硬编码 "ffmpeg" 会 WinError 2
    subprocess.run([cfg.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def stage_tts(proj: Project, ep_index: int) -> Episode:
    ep = store.get_episode(proj, ep_index)
    tts = providers.get_tts()
    total = 0.0
    for sh in ep.shots:
        if not sh.dialogue:
            continue
        if sh.audio and Path(sh.audio).exists():
            continue
        audios: List[Path] = []
        dur = 0.0
        for j, d in enumerate(sh.dialogue):
            ch = next((c for c in proj.characters if c.id == d.character
                       or c.name == d.character), None)
            out = store.asset_path(proj, "audio", f"{sh.id}_{j}.wav")
            _, seg = tts.synth(d.text, ch.voice_id if ch else None, d.emotion, str(out))
            d.duration = seg
            d.audio = str(out)
            audios.append(out)
            dur += seg
        merged = store.asset_path(proj, "audio", f"{sh.id}.wav")
        if len(audios) == 1 or not providers.has_ffmpeg():
            merged.write_bytes(audios[0].read_bytes())   # 无 ffmpeg 时降级：只保留首句
        else:
            _concat_audio(audios, merged)
        sh.audio = str(merged)
        # 音画同步：视频时长去迁就音频（先配音后视频的核心闭环）
        sh.duration = round(max(sh.duration, dur + 0.6), 2)
        total += sh.duration
    ep.status = "tts"
    proj.stage = f"tts@{ep_index}"
    store.save(proj)
    print(f"[ok] 配音完成，本集总时长约 {total:.1f}s")
    return ep


# ------------------------------------------------------------------ 首帧静帧图（批量）

def stage_keyframes(proj: Project, ep_index: int, candidates: Optional[int] = None) -> Episode:
    """批量生成首帧静帧图（按模型批处理，绝不逐镜头切换模型）。

    candidates > 1 时，每个镜头出 N 张候选，再用 face 相似度自动挑最像角色的一张。
    """
    ep = store.get_episode(proj, ep_index)
    ip = providers.get_image()
    k = candidates or cfg.candidates
    n = 0
    for sh in ep.shots:
        if sh.keyframe and Path(sh.keyframe).exists():
            continue
        char_refs, scene_ref = prompt_builder.ref_images_for(proj, sh)

        if k > 1:
            paths: List[str] = []
            for c in range(k):
                out = store.asset_path(proj, "keyframes", f"{sh.id}_{c}.png")
                r = _safe_image(ip.generate, sh.image_prompt, "", char_refs, scene_ref,
                                cfg.width, cfg.height, 12345 + sh.index * 100 + c, str(out))
                if r:
                    paths.append(r)
            if not paths:
                continue
            sh.candidates = paths
            if cfg.auto_pick and char_refs:
                idx, scores = select.pick_best(paths, char_refs[0])
                if scores and max(scores) >= 0:      # 全 -1 表示无人脸/未装依赖，不打扰用户
                    print(f"  {sh.id} 候选相似度 {[round(s, 3) for s in scores]} → 选 #{idx}")
            else:
                idx = 0
            sh.keyframe = paths[max(idx, 0)]
        else:
            out = store.asset_path(proj, "keyframes", f"{sh.id}.png")
            r = _safe_image(ip.generate, sh.image_prompt, "", char_refs, scene_ref,
                            cfg.width, cfg.height, 12345 + sh.index, str(out))
            if not r:
                continue
            sh.keyframe = r

        sh.status = "keyframe"
        n += 1

    ep.status = "keyframed"
    proj.stage = f"keyframes@{ep_index}"
    store.save(proj)
    print(f"[ok] 生成首帧图 {n} 张（不满意可删掉对应文件后重跑本阶段）")
    return ep


# ------------------------------------------------------------------ 视频（批量）

def stage_video(proj: Project, ep_index: int) -> Episode:
    ep = store.get_episode(proj, ep_index)
    vp = providers.get_video()
    shots = ep.shots

    # 先收集所有待生成镜头（含首尾帧串联：下一镜首帧作为本镜收束帧）
    pending = []
    for i, sh in enumerate(shots):
        if sh.clip and Path(sh.clip).exists():
            continue
        if not sh.keyframe:
            print(f"[skip] {sh.id} 缺少首帧图")
            continue
        last_frame = shots[i + 1].keyframe if i + 1 < len(shots) else None
        pending.append((sh, store.asset_path(proj, "clips", f"{sh.id}.mp4"), last_frame))

    n = 0
    if pending and hasattr(vp, "generate_batch"):
        # 批量：一次加载模型跑完所有镜头，省掉每镜头 1~3 分钟的重复加载
        print(f"[batch] 批量生成 {len(pending)} 个镜头（一次加载模型）")
        vp.generate_batch([
            {"out": str(out), "image": sh.keyframe, "last": last or "",
             "prompt": sh.video_prompt, "duration": sh.duration,
             "width": cfg.width, "height": cfg.height, "fps": cfg.fps}
            for sh, out, last in pending])
        for sh, out, _ in pending:
            if out.exists():
                sh.clip = str(out)
                sh.status = "clip"
                n += 1
    else:
        for sh, out, last in pending:
            vp.generate(sh.keyframe, sh.video_prompt, sh.duration, str(out), last)
            sh.clip = str(out)
            sh.status = "clip"
            n += 1
    ep.status = "clipped"
    proj.stage = f"video@{ep_index}"
    store.save(proj)
    print(f"[ok] 生成视频片段 {n} 段")
    return ep


# ------------------------------------------------------------------ 合成

def stage_compose(proj: Project, ep_index: int) -> Path:
    compose.require_ffmpeg()
    ep = store.get_episode(proj, ep_index)
    out_dir = OUTPUT_DIR / proj.name
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: List[Path] = []
    for sh in ep.shots:
        dst = store.asset_path(proj, "clips", f"{sh.id}_final.mp4")
        if not dst.exists():
            compose.compose_shot(sh, dst)
        clips.append(dst)
    final = out_dir / f"ep{ep_index:02d}.mp4"
    compose.concat(clips, final)
    ep.status = "composed"
    proj.stage = f"composed@{ep_index}"
    store.save(proj)
    print(f"[ok] 成片已生成：{final}")

    # 为下一集准备记忆：更新世界状态 + 生成前情摘要
    llm = get_llm()
    data = llm.chat_json(prompts.system_prompt(proj, ep_index),
                         prompts.build_state_delta_prompt(proj, ep_index),
                         mock_kind="state")
    w = proj.world
    w.timeline = data.get("timeline", w.timeline)
    for cid, st in (data.get("characters") or {}).items():
        w.characters.setdefault(cid, {}).update(st)
    for k, v in (data.get("props") or {}).items():
        w.props[k] = v
    for h in data.get("hooks_add", []):
        if h not in w.hooks:
            w.hooks.append(h)
    for h in data.get("hooks_remove", []):
        if h in w.hooks:
            w.hooks.remove(h)
    for f in data.get("facts_add", []):
        if f not in w.facts:
            w.facts.append(f)
    ep.summary = data.get("summary", "")
    ep.state_delta = data
    store.save(proj)
    print(f"[ok] 已更新世界状态与前情摘要（时间线：{w.timeline}）")
    return final


# ------------------------------------------------------------------ 一键跑完

PIPELINE = ["outline", "characters", "scenes", "script",
            "shots", "check", "tts", "keyframes", "video", "compose"]


def run_pipeline(proj: Project, ep_index: int, *, until: str = "compose",
                 auto: Optional[bool] = None) -> None:
    for name in PIPELINE:
        if name in ("outline", "characters", "scenes"):
            if name == "outline":
                stage_outline(proj, auto)
            elif name == "characters":
                stage_characters(proj, auto)
            else:
                stage_scenes(proj, auto)
        else:
            if name == "script":
                stage_script(proj, ep_index, auto)
            elif name == "shots":
                stage_shots(proj, ep_index)
            elif name == "check":
                stage_check(proj, ep_index)
            elif name == "tts":
                stage_tts(proj, ep_index)
            elif name == "keyframes":
                stage_keyframes(proj, ep_index)
            elif name == "video":
                stage_video(proj, ep_index)
            elif name == "compose":
                stage_compose(proj, ep_index)
        if name == until:
            break
        proj = store.load(proj.name)      # 每阶段后重新载入，保证状态最新

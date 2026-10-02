"""Gradio 可视化界面（可选）。

用途：查看进度、逐阶段运行、浏览候选图并人工拍板。
启动：python -m autodrama ui      （默认 127.0.0.1:7860）

若在 AutoDL 上远程访问，用 SSH 隧道：
    ssh -L 7860:127.0.0.1:7860 root@xxx.autodl.com -p 端口
若要暴露到公网，务必加 auth（在 launch 里加 auth=("user","pwd")），否则别人会用你的 GPU 和 API Key。
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from typing import List

from . import store
from .schema import Project


def _run(fn) -> str:
    """捕获阶段函数的 stdout，回显到界面。"""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            fn()
    except Exception as e:
        buf.write(f"\n[error] {e}")
    return buf.getvalue()


def _shot_rows(proj: Project, ep_index: int) -> List[List[str]]:
    ep = next((e for e in proj.episodes if e.index == ep_index), None)
    if not ep or not ep.shots:
        return []
    return [[sh.id, sh.scene_id, sh.shot_size, sh.camera, sh.action,
             sh.dialogue[0].text if sh.dialogue else "", f"{sh.duration}s", sh.status]
            for sh in ep.shots]


def build():
    import gradio as gr
    from . import stages

    projects = store.list_projects() or [""]

    def load(name: str):
        if not name:
            return "还没有项目", [], [], gr.update(choices=[]), gr.update(choices=[])
        proj = store.load(name)
        chars = [f"{c.id}｜{c.name}" for c in proj.characters]
        scenes = [f"{s.id}｜{s.name}" for s in proj.scenes]
        info = (f"### {proj.title or proj.name}\n{proj.logline}\n\n"
                f"阶段：{proj.stage}｜角色 {len(proj.characters)}｜场景 {len(proj.scenes)}\n\n"
                f"时间线：{proj.world.timeline}｜伏笔 {len(proj.world.hooks)} 个")
        return info, _shot_rows(proj, 1), [], gr.update(choices=chars), gr.update(choices=scenes)

    def shots_of(name: str, ep: int):
        if not name:
            return []
        return _shot_rows(store.load(name), int(ep))

    def do_stage(name: str, ep: int, stage: str, auto: bool):
        from .config import cfg
        cfg.auto_approve = bool(auto)      # 界面上勾选 = 跳过交互确认

        def once():
            proj = store.load(name)
            i = int(ep)
            {
                "outline": lambda: stages.stage_outline(proj, True if auto else None),
                "characters": lambda: stages.stage_characters(proj, True if auto else None),
                "scenes": lambda: stages.stage_scenes(proj, True if auto else None),
                "script": lambda: stages.stage_script(proj, i, True if auto else None),
                "shots": lambda: stages.stage_shots(proj, i),
                "check": lambda: stages.stage_check(proj, i),
                "tts": lambda: stages.stage_tts(proj, i),
                "keyframes": lambda: stages.stage_keyframes(proj, i),
                "video": lambda: stages.stage_video(proj, i),
                "compose": lambda: stages.stage_compose(proj, i),
            }[stage]()

        return _run(once)

    def show_char_candidates(name: str, char: str):
        if not name or not char:
            return []
        proj = store.load(name)
        cid = char.split("｜")[0]
        c = next((x for x in proj.characters if x.id == cid), None)
        if not c:
            return []
        return [(p, f"候选 {i}") for i, p in enumerate(c.ref_images)]

    def show_scene_candidates(name: str, scene: str):
        if not name or not scene:
            return []
        proj = store.load(name)
        sid = scene.split("｜")[0]
        s = next((x for x in proj.scenes if x.id == sid), None)
        if not s:
            return []
        import glob, os
        d = store.project_dir(name) / "assets" / "scenes"
        files = sorted(glob.glob(str(d / f"{sid}_*.png")))
        return [(f, os.path.basename(f)) for f in files]

    def pick_char(name: str, char: str, idx: int):
        proj = store.load(name)
        cid = char.split("｜")[0]
        c = next((x for x in proj.characters if x.id == cid), None)
        if not c or not c.ref_images:
            return "没有可选择的候选图"
        c.golden_ref = c.ref_images[int(idx) % len(c.ref_images)]
        c.approved = True
        store.save(proj)
        return f"✅ 角色 {c.name} 黄金参考图已锁定：{c.golden_ref}"

    def pick_scene(name: str, scene: str, idx: int):
        import glob
        proj = store.load(name)
        sid = scene.split("｜")[0]
        s = next((x for x in proj.scenes if x.id == sid), None)
        if not s:
            return "场景不存在"
        d = store.project_dir(name) / "assets" / "scenes"
        files = sorted(glob.glob(str(d / f"{sid}_*.png")))
        if not files:
            return "没有可选择的候选图"
        s.establishing_shot = files[int(idx) % len(files)]
        s.approved = True
        store.save(proj)
        return f"✅ 场景 {s.name} 定场图已锁定：{s.establishing_shot}"

    with gr.Blocks(title="AutoDrama · AI 短剧流水线") as demo:
        gr.Markdown("# AutoDrama · 一站式 AI 短剧流水线\n"
                    "创意 → 大纲 → 角色卡 → 场景卡 → 剧本 → 分镜 → 配音 → 首帧图 → 视频 → 成片")

        with gr.Row():
            proj = gr.Dropdown(choices=projects, value=projects[0], label="项目")
            ep = gr.Number(value=1, label="集数", precision=0)
            auto = gr.Checkbox(value=True, label="跳过交互确认（界面模式下建议开启）")
            refresh = gr.Button("刷新")

        info = gr.Markdown()

        gr.Markdown("---")

        with gr.Row():
            b1 = gr.Button("outline")
            b2 = gr.Button("characters")
            b3 = gr.Button("scenes")
            b4 = gr.Button("script")
            b5 = gr.Button("shots")
        with gr.Row():
            b6 = gr.Button("check")
            b7 = gr.Button("tts")
            b8 = gr.Button("keyframes")
            b9 = gr.Button("video")
            b10 = gr.Button("compose")

        logbox = gr.Textbox(lines=18, label="运行日志", autoscroll=True)

        for btn, name in [(b1, "outline"), (b2, "characters"), (b3, "scenes"),
                          (b4, "script"), (b5, "shots"), (b6, "check"),
                          (b7, "tts"), (b8, "keyframes"), (b9, "video"), (b10, "compose")]:
            btn.click(lambda n, e, a, nm=name: do_stage(n, e, nm, a),
                      [proj, ep, auto], [logbox])

        gr.Markdown("---\n## 审核关卡：人工拍板（决定全剧一致性）")

        with gr.Tab("角色形象图"):
            with gr.Row():
                char_dd = gr.Dropdown(choices=[], label="角色")
                char_idx = gr.Number(value=0, label="候选序号", precision=0)
                char_btn = gr.Button("设为黄金参考图")
            char_gallery = gr.Gallery(label="候选图", columns=4, height=320)
            char_out = gr.Markdown()
            char_dd.change(show_char_candidates, [proj, char_dd], [char_gallery])
            char_btn.click(pick_char, [proj, char_dd, char_idx], [char_out])
            char_btn.click(show_char_candidates, [proj, char_dd], [char_gallery])

        with gr.Tab("场景定场图"):
            with gr.Row():
                scene_dd = gr.Dropdown(choices=[], label="场景")
                scene_idx = gr.Number(value=0, label="候选序号", precision=0)
                scene_btn = gr.Button("设为定场图")
            scene_gallery = gr.Gallery(label="候选图", columns=4, height=320)
            scene_out = gr.Markdown()
            scene_dd.change(show_scene_candidates, [proj, scene_dd], [scene_gallery])
            scene_btn.click(pick_scene, [proj, scene_dd, scene_idx], [scene_out])
            scene_btn.click(show_scene_candidates, [proj, scene_dd], [scene_gallery])

        with gr.Tab("分镜表"):
            table = gr.Dataframe(
                headers=["镜头", "场景", "景别", "运镜", "动作", "台词", "时长", "状态"],
                datatype=["str"] * 8, row_count=20, col_count=(8, "fixed"))
            gr.Button("刷新分镜表").click(shots_of, [proj, ep], [table])

        refresh.click(load, [proj], [info, table, char_gallery, char_dd, scene_dd])
        proj.change(load, [proj], [info, table, char_gallery, char_dd, scene_dd])
        ep.change(shots_of, [proj, ep], [table])

    return demo


def launch(share: bool = False, port: int = 7860) -> None:
    build().launch(server_name="127.0.0.1", server_port=port, share=share)

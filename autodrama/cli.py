"""AutoDrama 命令行入口。

典型用法（AutoDL 上）：
    python -m autodrama init --name yuye --idea "车祸后她发现丈夫记忆里有另一个女人" --episodes 6
    python -m autodrama run -p yuye --ep 1            # 一键跑到成片
    python -m autodrama run -p yuye --ep 1 --until shots   # 只跑到分镜，人工确认后再继续
    python -m autodrama status -p yuye
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from . import store
from .config import cfg
from .schema import Project


def _get_project(args) -> Project:
    return store.find_project(getattr(args, "project", None))


def cmd_init(args) -> None:
    proj = store.create_project(args.name, args.idea, genre=args.genre,
                                style=args.style, total_episodes=args.episodes)
    print(f"[ok] 项目已创建：{store.project_dir(proj.name)}")
    print("下一步：python -m autodrama run -p "
          f"{proj.name} --ep 1 --until shots")


def cmd_status(args) -> None:
    proj = _get_project(args)
    print(f"项目：{proj.name}｜{proj.title}｜阶段：{proj.stage}")
    print(f"角色 {len(proj.characters)} 个｜场景 {len(proj.scenes)} 个｜"
          f"已写集数 {len(proj.episodes)}")
    print(f"未回收伏笔：{len(proj.world.hooks)}｜时间线：{proj.world.timeline}")
    for c in proj.characters:
        print(f"  - 角色 {c.name}：{'✅已锁定' if c.approved else '⏳待审核'}"
              f"｜参考图 {c.golden_ref or '—'}")
    for s in proj.scenes:
        print(f"  - 场景 {s.name}：{'✅已锁定' if s.approved else '⏳待审核'}")
    for ep in proj.episodes:
        done = sum(1 for sh in ep.shots if sh.status in ("clip", "approved"))
        print(f"  - 第{ep.index}集《{ep.title}》剧本："
              f"{'✅' if ep.script_approved else '⏳'}｜镜头 {done}/{len(ep.shots)}｜{ep.status}")


def cmd_run(args) -> None:
    from . import stages
    proj = _get_project(args)
    stages.run_pipeline(proj, args.ep, until=args.until, auto=args.yes or None)


def _single_stage(fn, args, needs_ep: bool):
    proj = _get_project(args)
    if needs_ep:
        fn(proj, args.ep, auto=args.yes or None) if fn.__name__.startswith("stage_script") \
            else fn(proj, args.ep)
    else:
        fn(proj, auto=args.yes or None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autodrama", description="AutoDrama：一站式 AI 短剧自动化流水线")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_, ep=False):
        s = sub.add_parser(name, help=help_)
        s.add_argument("-p", "--project", help="项目名（省略则使用最近的项目）")
        s.add_argument("--yes", action="store_true", help="跳过人工审核关卡（调试用）")
        if ep:
            s.add_argument("--ep", type=int, default=1, help="集数，默认 1")
        return s

    c = sub.add_parser("init", help="创建项目")
    c.add_argument("--name", required=True)
    c.add_argument("--idea", required=True, help="你的创意，一句话或一段话")
    c.add_argument("--genre", default="都市情感")
    c.add_argument("--style", default="现实主义，偏冷色调")
    c.add_argument("--episodes", type=int, default=6)

    add("status", "查看项目进度")
    add("outline", "生成分集大纲")
    add("characters", "生成角色卡并选定人物参考图（审核点）")
    add("scenes", "生成场景卡并选定定场图（审核点）")
    add("script", "生成某一集剧本（审核点）", ep=True)
    add("shots", "把剧本拆成分镜表", ep=True)
    add("check", "一致性审查", ep=True)
    add("tts", "配音并锁定镜头时长", ep=True)
    k = add("keyframes", "批量生成首帧静帧图（每个镜头 N 张候选，自动挑最像的）", ep=True)
    k.add_argument("--candidates", type=int, default=None,
                   help="每个镜头候选数量，默认取 AUTODRAMA_CANDIDATES（默认 3）")
    add("video", "批量生成视频片段", ep=True)
    add("compose", "合成成片并更新世界状态", ep=True)

    u = sub.add_parser("ui", help="启动 Gradio 可视化界面")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--share", action="store_true", help="生成公网穿透链接（注意鉴权与 GPU 费用）")

    r = add("run", "按顺序自动执行多个阶段", ep=True)
    r.add_argument("--until", default="compose",
                   choices=["outline", "characters", "scenes", "script", "shots",
                            "check", "tts", "keyframes", "video", "compose"])

    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "init":
            cmd_init(args)
        elif args.cmd == "status":
            cmd_status(args)
        elif args.cmd == "run":
            cmd_run(args)
        elif args.cmd == "ui":
            from .ui import launch
            launch(share=args.share, port=args.port)
        else:
            from . import stages
            proj = _get_project(args)
            auto = args.yes or None
            ep = getattr(args, "ep", 1)
            {
                "outline": lambda: stages.stage_outline(proj, auto),
                "characters": lambda: stages.stage_characters(proj, auto),
                "scenes": lambda: stages.stage_scenes(proj, auto),
                "script": lambda: stages.stage_script(proj, ep, auto),
                "shots": lambda: stages.stage_shots(proj, ep),
                "check": lambda: stages.stage_check(proj, ep),
                "tts": lambda: stages.stage_tts(proj, ep),
                "keyframes": lambda: stages.stage_keyframes(
                    proj, ep, getattr(args, "candidates", None)),
                "video": lambda: stages.stage_video(proj, ep),
                "compose": lambda: stages.stage_compose(proj, ep),
            }[args.cmd]()
        return 0
    except Exception as e:
        print(f"[error] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

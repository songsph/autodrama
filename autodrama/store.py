"""项目存储：JSON 落盘，支持 git diff、断点续跑与人工回滚。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import PROJECTS_DIR
from .schema import Episode, Project


def project_dir(name: str) -> Path:
    return PROJECTS_DIR / name


def create_project(name: str, idea: str, *, genre: str = "", style: str = "",
                   total_episodes: int = 6) -> Project:
    p = project_dir(name)
    for sub in ("assets/characters", "assets/scenes", "assets/keyframes",
                "assets/clips", "assets/audio", "episodes"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    proj = Project(
        name=name, idea=idea, genre=genre, style=style,
        total_episodes=total_episodes, created_at=datetime.now().isoformat(),
    )
    save(proj)
    return proj


def save(proj: Project) -> None:
    d = project_dir(proj.name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "project.json").write_text(
        json.dumps(proj.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    # 同时导出人类可读的 Markdown，方便在 GitHub 上直接浏览
    (d / "README.md").write_text(to_markdown(proj), encoding="utf-8")


def load(name: str) -> Project:
    f = project_dir(name) / "project.json"
    if not f.exists():
        raise FileNotFoundError(f"项目不存在：{name}（先用 init 创建）")
    return Project.model_validate_json(f.read_text(encoding="utf-8"))


def list_projects() -> list[str]:
    return sorted(p.name for p in PROJECTS_DIR.iterdir() if (p / "project.json").exists())


def get_episode(proj: Project, index: int) -> Episode:
    ep = next((e for e in proj.episodes if e.index == index), None)
    if ep is None:
        ep = Episode(index=index)
        proj.episodes.append(ep)
        proj.episodes.sort(key=lambda x: x.index)
    return ep


def asset_path(proj: Project, kind: str, filename: str) -> Path:
    d = project_dir(proj.name) / "assets" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def to_markdown(proj: Project) -> str:
    """导出可读文档，让剧本/角色/分镜可以直接在 GitHub 上审阅。"""
    L = [f"# {proj.title or proj.name}", "",
         f"> {proj.logline}", "",
         f"- 类型：{proj.genre}｜风格：{proj.style}｜集数：{proj.total_episodes}",
         f"- 当前阶段：{proj.stage}", ""]
    if proj.characters:
        L += ["## 角色卡", "", "| ID | 名称 | 角色 | 外貌锚定词 | 参考图 | 审核 |",
              "|---|---|---|---|---|---|"]
        for c in proj.characters:
            L.append(f"| {c.id} | {c.name} | {c.role} | {c.appearance[:40]}… | "
                     f"{c.golden_ref or '—'} | {'✅' if c.approved else '⏳'} |")
        L.append("")
    if proj.scenes:
        L += ["## 场景卡", "", "| ID | 名称 | 时间 | 光照 | 定场图 | 审核 |", "|---|---|---|---|---|---|"]
        for s in proj.scenes:
            L.append(f"| {s.id} | {s.name} | {s.time_of_day} | {s.light} | "
                     f"{s.establishing_shot or '—'} | {'✅' if s.approved else '⏳'} |")
        L.append("")
    for ep in proj.episodes:
        L += [f"## 第 {ep.index} 集 {ep.title}", "", ep.synopsis, ""]
        if ep.shots:
            L += ["| 镜头 | 场景 | 景别 | 运镜 | 动作 | 台词 | 时长 | 状态 |",
                  "|---|---|---|---|---|---|---|---|"]
            for sh in ep.shots:
                d = sh.dialogue[0].text if sh.dialogue else "—"
                L.append(f"| {sh.id} | {sh.scene_id} | {sh.shot_size} | {sh.camera} | "
                         f"{sh.action[:24]}… | {d} | {sh.duration}s | {sh.status} |")
            L.append("")
    return "\n".join(L)


def find_project(name: Optional[str] = None) -> Project:
    """未指定项目名时，取唯一/最近的项目。"""
    names = list_projects()
    if not names:
        raise RuntimeError("还没有任何项目，请先执行：python -m autodrama init ...")
    if name:
        return load(name)
    return load(names[-1])

"""人工审核关卡（Human-in-the-Loop）。

三个必须由用户拍板的节点：
    1. 剧本          —— 决定故事走向，改这里最便宜
    2. 主要人物图片  —— 决定全剧人物一致性，选定后成为"黄金参考图"
    3. 场景图片      —— 决定全剧空间与光影一致性，选定后成为"定场图"

这三个节点一旦确认，对应字段即被【锁定】，后续所有镜头逐字复现，禁止自动改写。
"""

from __future__ import annotations

from typing import List, Optional

from .config import cfg

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    _console = Console()
except Exception:                      # rich 缺失时退化成纯文本
    _console = None
    Panel = None
    Prompt = None


def _print(text: str, title: str = "") -> None:
    if _console and Panel:
        _console.print(Panel(text, title=title or None, border_style="cyan"))
    else:
        print(f"\n===== {title} =====\n{text}\n")


def ask(prompt: str, default: str = "y") -> str:
    if Prompt:
        return Prompt.ask(prompt, default=default).strip().lower()
    return input(f"{prompt} [{default}] ").strip().lower() or default


def confirm(title: str, content: str, *, auto: Optional[bool] = None) -> bool:
    """展示内容并等待用户确认。auto=True 时直接通过（调试用）。"""
    _print(content, title)
    if auto is None:
        auto = cfg.auto_approve
    if auto:
        print(f"[auto] {title} 已自动通过")
        return True
    while True:
        ans = ask("是否通过？(y=通过 / n=打回重做 / e=手动编辑文件后继续)", "y")
        if ans in ("y", "yes", ""):
            return True
        if ans == "n":
            return False
        if ans == "e":
            input("请直接编辑文件，改完后按回车继续…")
            return True


def choose(title: str, items: List[str], *, auto: Optional[bool] = None) -> int:
    """从候选列表中挑选一张（用于选黄金参考图 / 定场图）。"""
    if not items:
        return -1
    body = "\n".join(f"  [{i}] {p}" for i, p in enumerate(items))
    _print(body, title)
    if auto is None:
        auto = cfg.auto_approve
    if auto:
        print(f"[auto] {title} 自动选择第 0 项")
        return 0
    while True:
        ans = ask(f"选择序号 (0-{len(items) - 1})，r=全部重生成", "0")
        if ans == "r":
            return -2
        if ans.isdigit() and 0 <= int(ans) < len(items):
            return int(ans)
        print("输入无效，请重试。")

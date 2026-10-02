"""FFmpeg 合成：音画对齐、烧录字幕、统一调色、拼接成片。

关键设计：
- 音画同步以【音频时长】为准，视频去迁就音频（先配音后视频的顺序在这里闭环）
- 每个片段统一编码参数（分辨率/帧率/像素格式/音频编码），保证 concat 可直接拼接
- 统一调色是"一致性兜底大招"，能救回不同批次生成画面的色差
- ffmpeg 路径来自 cfg（可指向 D 盘安装位置，无需加入 PATH）
- ffmpeg 报错信息必须可见，否则排错无从下手
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional

from .config import ROOT, cfg
from .schema import Shot

FFMPEG = cfg.ffmpeg
FFPROBE = cfg.ffprobe
CACHE = ROOT / ".cache"


def _run(cmd: List[str]) -> None:
    """执行 ffmpeg/ffprobe，失败时把 stderr 抛出来（Windows 下务必忽略解码错误）。

    固定 cwd 为仓库根目录，这样字体可以用相对路径，避开 Windows 盘符冒号
    在 ffmpeg 滤镜里无法正确转义的坑。
    """
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT),
                       encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(
            f"命令失败（{r.returncode}）：{' '.join(cmd[:6])} …\n"
            f"{r.stderr[-2000:]}"
        )


def ensure_font() -> str:
    """把系统字体复制成仓库内的相对路径，供 drawtext 使用。

    Windows 的 `C:/...` 盘符冒号在 ffmpeg 滤镜里转义极不稳定，
    改用相对路径 + 固定 cwd 彻底规避。
    """
    if not cfg.font:
        return ""
    src = Path(cfg.font)
    if ":" not in str(src):                 # Linux 绝对路径可直接用
        return str(src)
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / "font.ttf"
    if not dst.exists():
        import shutil
        shutil.copy2(src, dst)
    return ".cache/font.ttf"


def _escape_font(path: str) -> str:
    """相对路径无需转义；绝对路径里的反斜杠统一换成正斜杠。"""
    if ":" in path:
        return path.replace("\\", "/").replace(":", "\\:")
    return path.replace("\\", "/")


def _escape_text(text: str) -> str:
    return (text.replace("\\", "\\\\").replace(":", "\\:")
                .replace("'", "\\'").replace("%", "\\%"))


def _wrap(text: str, per_line: int = 14, max_lines: int = 2) -> List[str]:
    lines = [text[i:i + per_line] for i in range(0, len(text), per_line)]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    return lines


def subtitle_filter(text: str) -> str:
    """生成底部居中字幕滤镜（中文必须指定字体文件）。"""
    font = ensure_font()
    if not text or not font:
        return ""
    font = _escape_font(font)
    lines = _wrap(text)
    n = len(lines)
    filters = []
    base_y = cfg.height - 240
    for i, line in enumerate(lines):
        y = base_y + i * 62 - (0 if n == 1 else 31)
        filters.append(
            f"drawtext=fontfile={font}:text='{_escape_text(line)}'"
            f":fontcolor=white:fontsize=52:borderw=3:bordercolor=black@0.7"
            f":x=(w-text_w)/2:y={y}"
        )
    return ",".join(filters)


def _grade_filter() -> str:
    """统一调色：修正不同批次生成的色差。"""
    return "eq=contrast=1.04:saturation=1.06:brightness=0.01"


def silent_audio(duration: float, out: Path) -> Path:
    _run([FFMPEG, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
          "-t", f"{duration:.2f}", "-c:a", "aac", str(out)])
    return out


def probe_duration(path: Path) -> float:
    """优先 ffprobe；没有 ffprobe 时用 ffmpeg -i 解析 Duration（兼容精简版）。"""
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="ignore")
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        pass
    try:
        r = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True,
                           text=True, encoding="utf-8", errors="ignore")
        m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return 0.0


def compose_shot(shot: Shot, out: Path) -> Path:
    """把单个镜头的画面 + 音频 + 字幕合成为统一规格的片段。"""
    src = Path(shot.clip) if shot.clip else Path(shot.keyframe)
    if not src or not src.exists():
        raise FileNotFoundError(f"镜头 {shot.id} 缺少画面素材（clip/keyframe）")

    duration = float(shot.duration)
    audio = Path(shot.audio) if shot.audio and Path(shot.audio).exists() else None
    if audio is not None:
        adur = probe_duration(audio)
        if adur > 0.05:
            duration = max(duration, adur + 0.6)     # 台词后留 0.6s 呼吸
        else:
            audio = None                             # 空/无效音频视为无声

    tmp = out.parent / f"{shot.id}_raw.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    is_image = src.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    inp = ["-loop", "1", "-i", str(src)] if is_image else ["-i", str(src)]
    vf = (f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=increase,"
          f"crop={cfg.width}:{cfg.height},fps={cfg.fps},") + _grade_filter()
    sub = subtitle_filter(shot.dialogue[0].text if shot.dialogue else "")
    if sub:
        vf += "," + sub

    # 选项必须放在【所有输入之后】：ffmpeg 会把 -vf/-c:v 之类的选项归属于
    # 其后出现的那个文件，若音频 -i 排在后面，滤镜会被误当成音频的输入选项而报错。
    cmd = [FFMPEG, "-y", *inp]
    if audio is not None:
        cmd += ["-i", str(audio)]
    cmd += ["-t", f"{duration:.2f}", "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p"]
    if audio is not None:
        # 视频自带音轨时也要以配音为准，故显式 map
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
    cmd += ["-r", str(cfg.fps), str(tmp)]
    _run(cmd)

    # 没有配音就补静音，保证 concat 时所有片段都有音轨
    if audio is None:
        audio = silent_audio(duration, out.parent / f"{shot.id}_sil.m4a")

    _run([FFMPEG, "-y", "-i", str(tmp), "-i", str(audio),
          "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)])
    tmp.unlink(missing_ok=True)
    return out


def concat(clips: List[Path], out: Path, bgm: Optional[Path] = None) -> Path:
    """拼接成片。所有片段已统一编码参数，可直接用 concat demuxer。"""
    if not clips:
        raise ValueError("没有可拼接的片段")
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.parent / "concat.txt"
    lst.write_text("\n".join(f"file '{c.as_posix()}'" for c in clips), encoding="utf-8")

    cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if bgm and Path(bgm).exists():
        cmd += ["-i", str(bgm), "-filter_complex",
                "[1:a]volume=0.18[a1];[0:a][a1]amix=inputs=2:duration=first[a]",
                "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-c", "copy"]
    cmd += [str(out)]
    _run(cmd)
    return out


def require_ffmpeg() -> None:
    from shutil import which
    if Path(FFMPEG).is_file() or which(FFMPEG) is not None:
        return
    raise RuntimeError(
        "未检测到 ffmpeg。\n"
        "Windows：运行 python scripts/setup_env.py（自动装到 D 盘），\n"
        "或在 .env 中设置 AUTODRAMA_BIN_DIR 指向 ffmpeg 所在目录。\n"
        "Linux/AutoDL：apt install ffmpeg"
    )

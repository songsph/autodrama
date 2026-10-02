"""全局配置：一切路径可配置，密钥不落盘、不进代码。

D 盘策略：Windows 下软件、模型权重、缓存默认全部放 D:/autodrama，不占系统盘。
可用 AUTODRAMA_DATA_ROOT 覆盖；AutoDL 上改为 /root/autodl-tmp 即可。
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

IS_WIN = platform.system() == "Windows"


def env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def default_data_root() -> Path:
    """Windows 默认 D 盘（不占系统盘），Linux/AutoDL 默认仓库内。"""
    v = env("AUTODRAMA_DATA_ROOT")
    if v:
        return Path(v)
    if IS_WIN and Path("D:/").exists():
        return Path("D:/autodrama")
    return ROOT / ".data"


DATA_ROOT = default_data_root()
PROJECTS_DIR = Path(env("AUTODRAMA_PROJECTS_DIR", str(ROOT / "projects")))
OUTPUT_DIR = Path(env("AUTODRAMA_OUTPUTS_DIR", str(ROOT / "outputs")))

# ---- 让所有模型/缓存下载都落到 D 盘（必须在导入 torch/diffusers 之前设置）----
for _k, _sub in (("HF_HOME", "cache/hf"), ("MODELSCOPE_CACHE", "cache/modelscope"),
                 ("TORCH_HOME", "cache/torch"), ("XDG_CACHE_HOME", "cache/xdg")):
    os.environ.setdefault(_k, str(DATA_ROOT / _sub))


def guess_font() -> str:
    """中文字幕必须指定字体文件，否则 ffmpeg 画不出汉字。"""
    for p in (
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ):
        if Path(p).exists():
            return p
    return ""


def _find_tool(name: str) -> str:
    """查找外部可执行文件：优先 AUTODRAMA_BIN_DIR，其次 PATH，最后常见 D 盘安装位。"""
    bin_dir = env("AUTODRAMA_BIN_DIR")
    if bin_dir:
        p = Path(bin_dir) / (name + (".exe" if IS_WIN else ""))
        if p.exists():
            return str(p)
    found = shutil.which(name)
    if found:
        return found
    for base in (DATA_ROOT / "tools" / "ffmpeg" / "bin", Path("D:/tools/ffmpeg/bin")):
        p = base / (name + (".exe" if IS_WIN else ""))
        if p.exists():
            return str(p)
    return name


class Config:
    # ---- 路径 ----
    data_root = DATA_ROOT
    model_dir = Path(env("AUTODRAMA_MODEL_DIR", str(DATA_ROOT / "models")))
    ffmpeg = _find_tool("ffmpeg")
    ffprobe = _find_tool("ffprobe")

    # ---- LLM ----
    llm_backend = env("AUTODRAMA_LLM", "deepseek")
    deepseek_key = env("DEEPSEEK_API_KEY")
    deepseek_base = env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model = env("DEEPSEEK_MODEL", "deepseek-flash")
    # 推理类模型（deepseek-flash / v4-pro）会消耗大量 reasoning tokens，
    # 上限给小了会导致 content 被截断成空串 → JSON 解析失败。务必给足。
    llm_max_tokens = int(env("DEEPSEEK_MAX_TOKENS", "16384"))
    # 推理模型思考时容易"跑偏"，创意阶段留 0.8，结构化输出阶段用低温度
    llm_temperature = float(env("DEEPSEEK_TEMPERATURE", "0.8"))

    # ---- 渲染后端：mock / local / command ----
    image_backend = env("AUTODRAMA_IMAGE", "mock")
    video_backend = env("AUTODRAMA_VIDEO", "mock")
    tts_backend = env("AUTODRAMA_TTS", "mock")
    lipsync_backend = env("AUTODRAMA_LIPSYNC", "mock")

    # ---- 图像模型（local 模式）----
    image_model = env("AUTODRAMA_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
    ip_adapter_repo = env("AUTODRAMA_IP_ADAPTER", "h94/IP-Adapter")
    ip_adapter_scale = float(env("AUTODRAMA_IP_SCALE", "0.85"))
    scene_init_strength = float(env("AUTODRAMA_SCENE_INIT_STRENGTH", "0.35"))
    steps = int(env("AUTODRAMA_STEPS", "28"))
    guidance = float(env("AUTODRAMA_GUIDANCE", "7.0"))

    # ---- 命令模板（command 模式，适配任意模型）----
    image_cmd = env("AUTODRAMA_IMAGE_CMD")
    video_cmd = env("AUTODRAMA_VIDEO_CMD")
    tts_cmd = env("AUTODRAMA_TTS_CMD")
    lipsync_cmd = env("AUTODRAMA_LIPSYNC_CMD")

    # ---- 视频参数（竖屏短剧）----
    width = int(env("AUTODRAMA_WIDTH", "720"))
    height = int(env("AUTODRAMA_HEIGHT", "1280"))
    fps = int(env("AUTODRAMA_FPS", "24"))

    # ---- 候选自动选优 ----
    candidates = int(env("AUTODRAMA_CANDIDATES", "3"))
    auto_pick = env("AUTODRAMA_AUTO_PICK", "1") == "1"

    font = env("AUTODRAMA_FONT") or guess_font()
    use_lipsync = env("AUTODRAMA_USE_LIPSYNC", "0") == "1"
    auto_approve = env("AUTODRAMA_AUTO_APPROVE", "0") == "1"


cfg = Config()

for _d in (PROJECTS_DIR, OUTPUT_DIR, cfg.model_dir):
    _d.mkdir(parents=True, exist_ok=True)

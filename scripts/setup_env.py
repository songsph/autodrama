"""AutoDrama 一键环境安装（Windows 默认全部装到 D 盘，不占系统盘）。

做的事：
    1. 在 D 盘建目录结构（models / cache / tools / venv）
    2. 建 Python 虚拟环境并装依赖（pip 缓存也放 D 盘）
    3. 下载 ffmpeg / ffprobe 到 D 盘（合成成片必需）
    4. 生成 .env（指向 D 盘路径）

用法：
    python scripts/setup_env.py                      # Windows 默认 D:\\autodrama
    python scripts/setup_env.py --root D:\\autodrama  # 自定义
    python scripts/setup_env.py --skip-ffmpeg
    python scripts/setup_env.py --gpu                # 同时装 GPU 渲染依赖
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
# 依次尝试，某个源被限流/403 时自动回退
MIRRORS = [
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple/",
    "https://pypi.org/simple",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def default_root() -> str:
    if os.name == "nt" and Path("D:/").exists():
        return "D:/autodrama"
    return str(REPO / ".data")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def make_dirs(root: Path) -> None:
    for sub in ("", "venv", "models", "cache/pip", "cache/hf", "tools"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    log(f"[ok] 目录结构：{root}")


def make_venv(root: Path) -> Path:
    venv = root / "venv"
    exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not exe.exists():
        log("[..] 创建虚拟环境（D 盘）")
        run([sys.executable, "-m", "venv", str(venv)])
    return exe


def pip_install(py: Path, args: list[str], env: dict, optional: bool = False) -> bool:
    for mirror in MIRRORS:
        log(f"[..] pip install {' '.join(args[:2])} …（源：{mirror}）")
        r = subprocess.run([str(py), "-m", "pip", "install", *args, "-i", mirror], env=env)
        if r.returncode == 0:
            return True
        log(f"[warn] 该源失败（{r.returncode}），换下一个")
    if optional:
        log("[warn] 可选依赖安装失败，可稍后手动安装")
        return False
    raise RuntimeError(f"所有镜像源均安装失败：{args}")


def install_deps(py: Path, gpu: bool) -> None:
    env = dict(os.environ, PIP_CACHE_DIR=str(py.parent.parent / "cache" / "pip"))
    pip_install(py, ["--upgrade", "pip"], env, optional=True)   # 失败也不影响后续
    log("[..] 安装核心依赖")
    pip_install(py, ["-r", str(REPO / "requirements.txt")], env)
    if gpu:
        log("[..] 安装 GPU 渲染依赖（较大）")
        pip_install(py, ["-r", str(REPO / "requirements-gpu.txt")], env, optional=True)
    log("[ok] Python 依赖安装完成")


def _download(url: str, dest: Path, retries: int = 6) -> bool:
    """带断点续传的下载，断网后自动重试。"""
    import time
    from urllib.request import Request, urlopen

    for attempt in range(retries):
        resume = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={resume}-"} if resume else {}
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as r:
                if resume and r.status != 206:        # 服务器不支持续传
                    resume = 0
                total = int(r.headers.get("Content-Length", 0)) + resume
                with open(dest, "ab" if resume else "wb") as f:
                    done = resume
                    while True:
                        chunk = r.read(512 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total and done % (8 * 1024 * 1024) < 512 * 1024:
                            log(f"    {done / 1048576:.0f} / {total / 1048576:.0f} MB")
            if zipfile.is_zipfile(dest):
                return True
            log("    文件不完整，重试…")
        except Exception as e:
            log(f"    第 {attempt + 1} 次失败：{e}")
            time.sleep(3)
    return False


def _ffmpeg_from_pip(root: Path, py: Path) -> str:
    """兜底：用 pip 装 imageio-ffmpeg（约 30MB，国内源快），取出其中的 ffmpeg.exe。"""
    log("[..] 回退方案：pip 安装 imageio-ffmpeg")
    for mirror in MIRRORS:
        r = subprocess.run([str(py), "-m", "pip", "install", "imageio-ffmpeg", "-i", mirror])
        if r.returncode == 0:
            break
    else:
        return ""
    try:
        import imageio_ffmpeg
        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as e:
        log(f"[warn] imageio-ffmpeg 不可用：{e}")
        return ""
    bin_dir = root / "tools" / "ffmpeg" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    # imageio-ffmpeg 的 exe 名形如 ffmpeg-win64-v7.1.exe，统一重命名，便于查找
    target = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    shutil.copy2(src, target)
    log(f"[ok] ffmpeg（精简版）就绪：{bin_dir}")
    return str(bin_dir)


def install_ffmpeg(root: Path, py: Path) -> str:
    tools = root / "tools"
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    found = list(tools.rglob(exe_name))
    if found:
        log(f"[ok] ffmpeg 已存在：{found[0].parent}")
        return str(found[0].parent)

    if os.name != "nt":
        log("[skip] 非 Windows，请用包管理器安装：apt install ffmpeg")
        return ""

    zip_path = tools / "ffmpeg.zip"
    log(f"[..] 下载 ffmpeg（约 110MB）→ {zip_path}")
    if _download(FFMPEG_URL, zip_path):
        log("[..] 解压中")
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tools / "ffmpeg")
            zip_path.unlink(missing_ok=True)
            found = list((tools / "ffmpeg").rglob("ffmpeg.exe"))
            if found:
                log(f"[ok] ffmpeg 就绪：{found[0].parent}")
                return str(found[0].parent)
        except Exception as e:
            log(f"[warn] 解压失败：{e}")

    log(f"[warn] 自动下载未完成，请稍后重试或手动下载 {FFMPEG_URL}")
    return _ffmpeg_from_pip(root, py)


def write_env(root: Path, bin_dir: str) -> None:
    env_file = REPO / ".env"
    if env_file.exists():
        text = env_file.read_text(encoding="utf-8")
        if bin_dir and "AUTODRAMA_BIN_DIR" not in text:
            env_file.write_text(
                text.rstrip() + f"\nAUTODRAMA_BIN_DIR={Path(bin_dir).as_posix()}\n",
                encoding="utf-8")
            log("[ok] .env 已补充 AUTODRAMA_BIN_DIR")
        else:
            log("[skip] .env 已存在，未覆盖")
        return
    # 注释一律用 ASCII，避免 Windows GBK 控制台下出现乱码
    lines = [
        "# Generated by scripts/setup_env.py -- all data stays on the data drive",
        f"AUTODRAMA_DATA_ROOT={root.as_posix()}",
        f"AUTODRAMA_MODEL_DIR={(root / 'models').as_posix()}",
    ]
    if bin_dir:
        lines.append(f"AUTODRAMA_BIN_DIR={Path(bin_dir).as_posix()}")
    lines += [
        "",
        "# DeepSeek key: prefer NOT writing it to disk.",
        "#   Linux:   read -s DEEPSEEK_API_KEY && export DEEPSEEK_API_KEY",
        "DEEPSEEK_API_KEY=",
    ]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"[ok] 已生成 {env_file}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=default_root())
    ap.add_argument("--skip-ffmpeg", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="同时安装 GPU 渲染依赖")
    args = ap.parse_args()

    root = Path(args.root)
    make_dirs(root)
    py = make_venv(root)
    install_deps(py, args.gpu)
    bin_dir = "" if args.skip_ffmpeg else install_ffmpeg(root, py)
    write_env(root, bin_dir)

    log("")
    log("完成。以后用这个 Python 运行（所有包装在 D 盘）：")
    log(f"  {py} -m autodrama status")
    log(f"  {py} -m autodrama ui")
    return 0


if __name__ == "__main__":
    sys.exit(main())

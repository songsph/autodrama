#!/usr/bin/env python3
"""AutoDrama 视频生成 worker —— 一次加载模型，批量跑完一整集。

为什么需要它：
  视频模型加载一次要 1~3 分钟。如果每个镜头都起一个新进程，
  一集 18 个镜头光加载就要烧掉半小时 GPU 计费时间。
  所以流水线把所有镜头打包成 tasks.json，由本脚本一次加载、逐个生成。

三种后端（用 AUTODRAMA_VIDEO_BACKEND 选择）：
  wan  —— diffusers 的 Wan 系列（Wan2.1/2.2 I2V），接口最稳定，推荐先用它跑通
  h3   —— MiniMax H3，调用官方推理脚本（脚本接口常变，故支持命令模板覆盖）
  cmd  —— 完全自定义命令模板 AUTODRAMA_VIDEO_CMD

用法：
  # 批量（流水线调用）
  python scripts/video_worker.py --batch tasks.json
  # 单条（手工试跑）
  python scripts/video_worker.py --image a.png --prompt "..." --out o.mp4 --duration 3.0

tasks.json 每项字段：
  out(必填) image(必填) last prompt negative duration width height fps seed
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

FPS = int(os.environ.get("AUTODRAMA_FPS", "24"))
WIDTH = int(os.environ.get("AUTODRAMA_WIDTH", "720"))
HEIGHT = int(os.environ.get("AUTODRAMA_HEIGHT", "1280"))


# ------------------------------------------------------------------ 通用

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _save_video(frames, out: str, fps: int) -> None:
    """把帧序列存成 mp4。优先用 diffusers 自带工具，其次 imageio。"""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, out, fps=fps)
        return
    except Exception:
        pass
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(out, fps=fps) as w:
            for f in frames:
                w.append_data(f)
        return
    except Exception:
        pass
    raise RuntimeError(
        f"无法写出视频 {out}：缺少 imageio-ffmpeg。\n"
        f"请先安装：pip install imageio imageio-ffmpeg"
    )


def _num_frames(duration: float, fps: int) -> int:
    """多数视频模型要求帧数满足 4n+1，这里做对齐。"""
    n = max(1, int(round(float(duration) * fps)))
    return (n - 1) // 4 * 4 + 1


# ------------------------------------------------------------------ Wan（diffusers）

def run_wan(tasks: list, model_dir: str) -> None:
    import torch
    from PIL import Image

    try:
        from diffusers import WanImageToVideoPipeline as Pipe
    except ImportError:
        raise RuntimeError(
            "当前 diffusers 版本没有 WanImageToVideoPipeline。\n"
            "升级即可：pip install -U diffusers transformers accelerate"
        )

    print(f"[wan] 加载模型：{model_dir}")
    pipe = Pipe.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    try:
        pipe.enable_model_cpu_offload()      # 32G 显存也建议开，留余量给峰值
    except Exception:
        pipe.to("cuda")

    steps = int(_env("AUTODRAMA_STEPS", "28"))
    guidance = float(_env("AUTODRAMA_GUIDANCE", "7.0"))

    for i, t in enumerate(tasks, 1):
        out = t["out"]
        if Path(out).exists():
            print(f"[{i}/{len(tasks)}] 跳过（已存在）{out}")
            continue
        try:
            img = Image.open(t["image"]).convert("RGB")
            fps = int(t.get("fps", FPS))
            nf = _num_frames(t.get("duration", 3.0), fps)
            print(f"[{i}/{len(tasks)}] 生成 {out}（{nf} 帧）")
            res = pipe(
                image=img,
                prompt=t.get("prompt", ""),
                negative_prompt=t.get("negative", ""),
                num_frames=nf,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator("cuda").manual_seed(int(t.get("seed", 0)) or 0),
            )
            _save_video(res.frames[0], out, fps)
        except Exception as e:
            print(f"[error] {out} 生成失败：{e}")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ------------------------------------------------------------------ MiniMax H3

def _find_h3_script(repo: str) -> str:
    """H3 官方仓库的推理脚本名会变，这里按常见名字探测。"""
    if _env("AUTODRAMA_H3_SCRIPT"):
        return _env("AUTODRAMA_H3_SCRIPT")
    if repo and Path(repo).is_dir():
        for name in ("inference.py", "infer.py", "generate.py", "run.py", "demo.py",
                     "sample.py", "scripts/inference.py", "scripts/infer.py"):
            p = Path(repo) / name
            if p.is_file():
                return str(p)
    return ""


def run_h3(tasks: list) -> None:
    repo = _env("AUTODRAMA_H3_DIR", "")
    model_dir = _env("AUTODRAMA_H3_MODEL", "") or _env("AUTODRAMA_MODEL_DIR", "")
    script = _find_h3_script(repo)
    tpl = _env("AUTODRAMA_H3_CMD") or _env("AUTODRAMA_VIDEO_CMD")

    if not tpl and not script:
        raise RuntimeError(
            "找不到 H3 推理脚本。\n"
            "请设置 AUTODRAMA_H3_DIR（官方仓库目录，setup_autodl.sh --video h3 会自动 clone），\n"
            "或直接给出命令模板 AUTODRAMA_H3_CMD，例如：\n"
            "  export AUTODRAMA_H3_CMD='python /root/h3/infer.py --image \"{image}\" "
            "--prompt \"{prompt}\" --duration {duration} --out \"{out}\"'"
        )

    print(f"[h3] repo={repo or '-'} model={model_dir or '-'} script={script or '模板模式'}")
    for i, t in enumerate(tasks, 1):
        out = t["out"]
        if Path(out).exists():
            print(f"[{i}/{len(tasks)}] 跳过（已存在）{out}")
            continue
        kw = dict(
            image=t["image"], last=t.get("last", "") or t["image"],
            prompt=t.get("prompt", ""), negative=t.get("negative", ""),
            duration=t.get("duration", 3.0), out=out,
            width=int(t.get("width", WIDTH)), height=int(t.get("height", HEIGHT)),
            fps=int(t.get("fps", FPS)), seed=int(t.get("seed", 0)),
            model=model_dir,
        )
        try:
            if tpl:
                cmd = shlex.split(tpl.format(**kw))
            else:
                cmd = [sys.executable, script,
                       "--image", kw["image"], "--prompt", kw["prompt"],
                       "--duration", str(kw["duration"]), "--out", out]
                if kw["last"] and kw["last"] != kw["image"]:
                    cmd += ["--last", kw["last"]]
            print(f"[{i}/{len(tasks)}] {' '.join(cmd)[:160]}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[error] {out} 生成失败：{e}")
            traceback.print_exc()


# ------------------------------------------------------------------ 自定义命令

def run_cmd(tasks: list) -> None:
    tpl = _env("AUTODRAMA_VIDEO_CMD")
    if not tpl:
        raise RuntimeError("AUTODRAMA_VIDEO_CMD 未配置")
    for i, t in enumerate(tasks, 1):
        out = t["out"]
        if Path(out).exists():
            print(f"[{i}/{len(tasks)}] 跳过（已存在）{out}")
            continue
        cmd = shlex.split(tpl.format(
            image=t["image"], keyframe=t["image"], last=t.get("last", "") or t["image"],
            last_frame=t.get("last", "") or t["image"],
            prompt=t.get("prompt", ""), duration=t.get("duration", 3.0), out=out,
            width=int(t.get("width", WIDTH)), height=int(t.get("height", HEIGHT)),
            fps=int(t.get("fps", FPS)), seed=int(t.get("seed", 0))))
        print(f"[{i}/{len(tasks)}] {' '.join(cmd)[:160]}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[error] {out} 生成失败：{e}")


# ------------------------------------------------------------------ 入口

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", help="任务清单 JSON 路径")
    ap.add_argument("--backend", default=_env("AUTODRAMA_VIDEO_BACKEND", "wan"))
    ap.add_argument("--model", default=_env("AUTODRAMA_VIDEO_MODEL", ""))
    ap.add_argument("--image"), ap.add_argument("--last", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--duration", type=float, default=3.0)
    args = ap.parse_args()

    tasks: list
    if args.batch:
        tasks = json.loads(Path(args.batch).read_text(encoding="utf-8"))
    elif args.image and args.out:
        tasks = [{"out": args.out, "image": args.image, "last": args.last,
                  "prompt": args.prompt, "duration": args.duration}]
    else:
        ap.error("需要 --batch 或同时提供 --image 与 --out")
        return

    backend = (args.backend or "wan").lower()
    model_dir = args.model or _env("AUTODRAMA_VIDEO_MODEL") or \
        str(Path(_env("AUTODRAMA_MODEL_DIR", "models")) / (
            "Wan2.2-I2V-A14B" if backend == "wan" else "MiniMax-H3"))

    print(f"[worker] backend={backend} model={model_dir} tasks={len(tasks)}")
    if backend == "wan":
        run_wan(tasks, model_dir)
    elif backend == "h3":
        run_h3(tasks)
    else:
        run_cmd(tasks)

    done = sum(1 for t in tasks if Path(t["out"]).exists())
    print(f"[worker] 完成 {done}/{len(tasks)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""AutoDrama 配音 worker —— 一次加载 TTS 模型，批量合成一整集所有台词。

不用它也能跑：每个镜头起一次推理脚本即可。但 TTS 模型加载同样要几十秒，
一集几十句台词逐句起进程会非常慢，所以和视频一样走批量入口。

两种后端（AUTODRAMA_TTS_BACKEND）：
  cosyvoice —— CosyVoice2/3 官方代码，零样本音色克隆（推荐）
  cmd       —— 完全自定义命令模板 AUTODRAMA_TTS_CMD

用法：
  python scripts/tts_worker.py --batch tasks.json
  python scripts/tts_worker.py --text "你好" --voice /path/ref.wav --out o.wav

tasks.json 每项：out(必填) text(必填) voice speaker emotion
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


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# ------------------------------------------------------------------ CosyVoice

def run_cosyvoice(tasks: list) -> None:
    repo = _env("AUTODRAMA_COSYVOICE_DIR", "")
    model = _env("AUTODRAMA_COSYVOICE_MODEL", "") or _env("AUTODRAMA_MODEL_DIR", "")

    if repo and Path(repo).is_dir():
        sys.path.insert(0, repo)
        if Path(repo, "third_party", "Matcha-TTS").is_dir():
            sys.path.insert(0, str(Path(repo, "third_party", "Matcha-TTS")))

    tts = None
    last_err = ""
    # 各版本类名不一致（CosyVoice / CosyVoice2 / CosyVoice3），逐个尝试
    for mod_name, cls_name in (("cosyvoice.cli.cosyvoice", "CosyVoice"),
                               ("cosyvoice.cli.cosyvoice", "CosyVoice2"),
                               ("cosyvoice.cli.cosyvoice", "CosyVoice3")):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            tts = getattr(mod, cls_name)(model, load_trt=False, fp16=True)
            print(f"[cosyvoice] 已加载 {cls_name}｜model={model}")
            break
        except Exception as e:
            last_err = f"{cls_name}: {e}"
    if tts is None:
        raise RuntimeError(
            "CosyVoice 加载失败。请确认：\n"
            "  1) 已 clone 官方代码：git clone https://github.com/FunAudioLLM/CosyVoice\n"
            "     并设置 AUTODRAMA_COSYVOICE_DIR 指向该目录\n"
            "  2) 已下载权重并设置 AUTODRAMA_COSYVOICE_MODEL\n"
            "  3) 已装依赖：pip install -r requirements.txt（官方仓库内）\n"
            "若实在跑不通，可改用命令模板：AUTODRAMA_TTS_CMD\n"
            f"最后错误：{last_err}"
        )

    for i, t in enumerate(tasks, 1):
        out = t["out"]
        if Path(out).exists():
            print(f"[{i}/{len(tasks)}] 跳过（已存在）{out}")
            continue
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        text = t.get("text", "").strip()
        if not text:
            continue
        try:
            # 有参考音频走零样本克隆（音色一致），否则用预置音色
            sample = t.get("voice") or ""
            if sample and Path(sample).exists() and hasattr(tts, "inference_zero_shot"):
                gen = tts.inference_zero_shot(
                    text, t.get("speaker") or "角色", sample, stream=False)
            elif hasattr(tts, "inference_sft"):
                gen = tts.inference_sft(text, t.get("speaker") or "中文女", stream=False)
            else:
                gen = tts.inference_zero_shot(text, t.get("speaker") or "角色",
                                              sample or "", stream=False)
            _write_wav(gen, out, tts)
            print(f"[{i}/{len(tasks)}] {out}")
        except Exception as e:
            print(f"[error] {out} 合成失败：{e}")
            traceback.print_exc()


def _write_wav(gen, out: str, tts) -> None:
    """官方返回的是生成器，拼成整段后写 wav。"""
    import numpy as np
    import soundfile as sf
    chunks = []
    for item in gen:
        wav = item.get("tts_speech", item) if isinstance(item, dict) else item
        if hasattr(wav, "numpy"):
            wav = wav.numpy()
        chunks.append(np.asarray(wav).squeeze())
    if not chunks:
        raise RuntimeError("模型没有返回音频")
    data = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    sr = int(getattr(tts, "sample_rate", 22050))
    sf.write(out, data, sr)


# ------------------------------------------------------------------ 自定义命令

def run_cmd(tasks: list) -> None:
    tpl = _env("AUTODRAMA_TTS_CMD")
    if not tpl:
        raise RuntimeError("AUTODRAMA_TTS_CMD 未配置")
    for i, t in enumerate(tasks, 1):
        out = t["out"]
        if Path(out).exists():
            print(f"[{i}/{len(tasks)}] 跳过（已存在）{out}")
            continue
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cmd = shlex.split(tpl.format(
            text=t.get("text", ""), voice=t.get("voice", ""),
            speaker=t.get("speaker", ""), emotion=t.get("emotion", ""), out=out))
        print(f"[{i}/{len(tasks)}] {' '.join(cmd)[:160]}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"[error] {out} 合成失败：{e}")


# ------------------------------------------------------------------ 入口

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", help="任务清单 JSON 路径")
    ap.add_argument("--backend", default=_env("AUTODRAMA_TTS_BACKEND", "cosyvoice"))
    ap.add_argument("--text", default="")
    ap.add_argument("--voice", default="")
    ap.add_argument("--speaker", default="中文女")
    ap.add_argument("--emotion", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.batch:
        tasks = json.loads(Path(args.batch).read_text(encoding="utf-8"))
    elif args.text and args.out:
        tasks = [{"out": args.out, "text": args.text, "voice": args.voice,
                  "speaker": args.speaker, "emotion": args.emotion}]
    else:
        ap.error("需要 --batch 或同时提供 --text 与 --out")
        return

    print(f"[tts] backend={args.backend} tasks={len(tasks)}")
    if args.backend == "cosyvoice":
        run_cosyvoice(tasks)
    else:
        run_cmd(tasks)
    print(f"[tts] 完成 {sum(1 for t in tasks if Path(t['out']).exists())}/{len(tasks)}")


if __name__ == "__main__":
    main()

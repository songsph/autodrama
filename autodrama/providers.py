"""渲染后端抽象层：图像 / 视频 / 配音 / 口型。

三种后端，用环境变量切换：
    mock     离线占位，跑通流程用（默认）
    local    内置 diffusers 实现（目前图像已实现）
    command  调用外部命令模板 —— 适配任意模型，无需改代码

设计要点：
1. 一致性靠"参考图注入"：角色用 IP-Adapter 锚定，场景用定场图做 img2img 底图。
2. 32G 显存装不下所有模型，务必按阶段批处理，一个阶段结束即释放显存。
"""

from __future__ import annotations

import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from .config import cfg


def has_ffmpeg() -> bool:
    from shutil import which
    p = Path(cfg.ffmpeg)
    return p.is_file() or which(cfg.ffmpeg) is not None


def run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_cmd_template(tpl: str, **kw) -> None:
    """把命令模板中的占位符替换后执行。

    可用占位符：prompt negative char_ref scene_ref width height seed out
                keyframe duration last_frame text voice emotion clip audio
    """
    cmd = shlex.split(tpl.format(**kw))
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------ 图像

class ImageProvider(ABC):
    """生成分镜首帧静帧图。

    char_refs : 角色黄金参考图（用 IP-Adapter 注入身份，保证人物一致）
    scene_ref : 场景定场图（作为 img2img 底图，保证空间与光影一致）
    """

    @abstractmethod
    def generate(self, prompt: str, negative: str = "",
                 char_refs: Optional[List[str]] = None, scene_ref: Optional[str] = None,
                 width: int = 720, height: int = 1280, seed: int = 0,
                 out: str = "") -> str:
        ...


class MockImageProvider(ImageProvider):
    """用 PIL 画占位图，把 prompt 写进画面，方便离线验证流水线。"""

    def generate(self, prompt: str, negative: str = "",
                 char_refs: Optional[List[str]] = None, scene_ref: Optional[str] = None,
                 width: int = 720, height: int = 1280, seed: int = 0,
                 out: str = "") -> str:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (width, height), (26, 28, 34))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, width, 120], fill=(40, 44, 54))
        try:
            font = ImageFont.truetype(cfg.font, 26)
            small = ImageFont.truetype(cfg.font, 18)
        except Exception:
            font = small = ImageFont.load_default()
        d.text((24, 30), Path(out).stem, fill=(255, 208, 96), font=font)
        d.text((24, 78), f"角色参考 {len(char_refs or [])}｜场景参考 {'有' if scene_ref else '无'}"
                         f"｜seed {seed}", fill=(150, 160, 180), font=small)
        y = 160
        for i in range(0, len(prompt), 18):
            d.text((24, y), prompt[i:i + 18], fill=(225, 228, 235), font=small)
            y += 30
            if y > height - 60:
                break
        d.text((24, height - 46), "MOCK IMAGE — 接入真实模型后替换",
               fill=(120, 130, 150), font=small)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        return out


class LocalImageProvider(ImageProvider):
    """diffusers 实现：SDXL + IP-Adapter（角色锚定）+ 定场图 img2img（场景派生）。

    显存参考（5090 32G 完全够）：SDXL fp16 约 10~12G；加 IP-Adapter 几乎不增加显存。
    想更省显存可设置 AUTODRAMA_IMAGE_MODEL 为量化版，或改用 SD1.5 系列。
    """

    _pipe = None

    def _load(self):
        if LocalImageProvider._pipe is not None:
            return LocalImageProvider._pipe
        import torch
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.image_model, torch_dtype=torch.float16, use_safetensors=True)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            try:
                pipe.enable_model_cpu_offload()      # 显存紧张时的保险开关
            except Exception:
                pass
        try:
            pipe.load_ip_adapter(cfg.ip_adapter_repo, subfolder="sdxl_models",
                                 weight_name="ip-adapter_sdxl.bin")
            print("[info] IP-Adapter 已加载（角色一致性开启）")
        except Exception as e:
            print(f"[warn] IP-Adapter 加载失败，人物一致性会下降：{e}")
        LocalImageProvider._pipe = pipe
        return pipe

    def generate(self, prompt: str, negative: str = "",
                 char_refs: Optional[List[str]] = None, scene_ref: Optional[str] = None,
                 width: int = 720, height: int = 1280, seed: int = 0,
                 out: str = "") -> str:
        import torch
        from PIL import Image

        pipe = self._load()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kw = dict(prompt=prompt, negative_prompt=negative,
                  num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
                  generator=torch.Generator(device).manual_seed(seed))

        # 角色一致性：IP-Adapter 注入黄金参考图
        try:
            if char_refs:
                ref = Image.open(char_refs[0]).convert("RGB").resize((1024, 1024))
                kw["ip_adapter_image"] = ref
                pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
            elif hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(0.0)
        except Exception as e:
            print(f"[warn] IP-Adapter 注入失败：{e}")

        Path(out).parent.mkdir(parents=True, exist_ok=True)

        # 场景一致性：以定场图为底图做 img2img，保证空间/光影/色调同源
        if scene_ref and Path(scene_ref).exists() and cfg.scene_init_strength > 0:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            init = Image.open(scene_ref).convert("RGB").resize((width, height))
            i2i = StableDiffusionXLImg2ImgPipeline(**pipe.components)
            img = i2i(image=init, strength=cfg.scene_init_strength, **kw).images[0]
        else:
            img = pipe(width=width, height=height, **kw).images[0]

        img.save(out)
        return out


class CommandImageProvider(ImageProvider):
    """命令模板模式：适配任意图像模型/整合包。

    例（.env）：
    AUTODRAMA_IMAGE_CMD=python D:/models/flux/infer.py --prompt "{prompt}"
        --neg "{negative}" --ref "{char_ref}" --w {width} --h {height}
        --seed {seed} --out "{out}"
    """

    def generate(self, prompt: str, negative: str = "",
                 char_refs: Optional[List[str]] = None, scene_ref: Optional[str] = None,
                 width: int = 720, height: int = 1280, seed: int = 0,
                 out: str = "") -> str:
        if not cfg.image_cmd:
            raise RuntimeError("AUTODRAMA_IMAGE_CMD 未配置")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        run_cmd_template(cfg.image_cmd, prompt=prompt, negative=negative,
                         char_ref=(char_refs or [""])[0], scene_ref=scene_ref or "",
                         width=width, height=height, seed=seed, out=out)
        return out


# ------------------------------------------------------------------ 视频

class VideoProvider(ABC):
    """图生视频：只负责"让首帧图动起来"，画面内容由首帧图锁死。"""

    @abstractmethod
    def generate(self, keyframe: str, prompt: str, duration: float,
                 out: str = "", last_frame: Optional[str] = None) -> str:
        ...


class MockVideoProvider(VideoProvider):
    def generate(self, keyframe: str, prompt: str, duration: float,
                 out: str = "", last_frame: Optional[str] = None) -> str:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if has_ffmpeg():
            run([cfg.ffmpeg, "-y", "-loop", "1", "-i", keyframe, "-t", f"{duration}",
                 "-r", str(cfg.fps), "-vf",
                 f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=increase,"
                 f"crop={cfg.width}:{cfg.height}",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
        else:
            Path(out).write_bytes(b"")   # 没 ffmpeg 时只占位，流水线仍可跑通
        return out


class CommandVideoProvider(VideoProvider):
    """命令模板模式：适配 MiniMax H3 / Wan2.2 等任意视频模型。

    例（AutoDL）：
    AUTODRAMA_VIDEO_CMD=python /root/h3/infer.py --img "{keyframe}"
        --last "{last_frame}" --prompt "{prompt}" --dur {duration} --out "{out}"

    强烈建议使用【首尾帧模式】：--last 传入下一镜头的首帧，镜头衔接最自然。
    """

    def generate(self, keyframe: str, prompt: str, duration: float,
                 out: str = "", last_frame: Optional[str] = None) -> str:
        if not cfg.video_cmd:
            raise RuntimeError(
                "AUTODRAMA_VIDEO_CMD 未配置。\n"
                "例：python /root/h3/infer.py --img \"{keyframe}\" --last \"{last_frame}\" "
                "--prompt \"{prompt}\" --dur {duration} --out \"{out}\""
            )
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        run_cmd_template(cfg.video_cmd, keyframe=keyframe, last_frame=last_frame or "",
                         prompt=prompt, duration=duration, out=out,
                         width=cfg.width, height=cfg.height)
        return out


# ------------------------------------------------------------------ 配音

class TTSProvider(ABC):
    @abstractmethod
    def synth(self, text: str, voice_id: Optional[str], emotion: str,
              out: str = "") -> Tuple[str, float]:
        """返回 (音频路径, 时长秒)。时长用于反推镜头时长，保证音画同步。"""
        ...


class MockTTSProvider(TTSProvider):
    def synth(self, text: str, voice_id: Optional[str], emotion: str,
              out: str = "") -> Tuple[str, float]:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        dur = max(1.0, min(5.0, len(text) * 0.25))
        if has_ffmpeg():
            run([cfg.ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                 "-t", f"{dur:.2f}", out])
        else:
            Path(out).write_bytes(b"")
        return out, dur


class CommandTTSProvider(TTSProvider):
    """命令模板模式：适配 CosyVoice3 / IndexTTS-2 / Qwen3-TTS。

    例：
    AUTODRAMA_TTS_CMD=python /root/CosyVoice/infer.py --text "{text}"
        --voice "{voice}" --emotion "{emotion}" --out "{out}"
    """

    def synth(self, text: str, voice_id: Optional[str], emotion: str,
              out: str = "") -> Tuple[str, float]:
        if not cfg.tts_cmd:
            raise RuntimeError("AUTODRAMA_TTS_CMD 未配置")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        run_cmd_template(cfg.tts_cmd, text=text, voice=voice_id or "",
                         emotion=emotion, out=out)
        dur = 0.0
        if Path(out).exists() and has_ffmpeg():
            r = subprocess.run(
                [cfg.ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(out)], capture_output=True, text=True)
            try:
                dur = float(r.stdout.strip())
            except ValueError:
                dur = 0.0
        return out, dur or max(1.0, min(5.0, len(text) * 0.25))


# ------------------------------------------------------------------ 口型

class LipsyncProvider(ABC):
    @abstractmethod
    def apply(self, clip: str, audio: str, out: str = "") -> str:
        ...


class MockLipsyncProvider(LipsyncProvider):
    def apply(self, clip: str, audio: str, out: str = "") -> str:
        return clip


class CommandLipsyncProvider(LipsyncProvider):
    """命令模板模式：适配 MuseTalk / LatentSync（画质不稳可关闭）。"""

    def apply(self, clip: str, audio: str, out: str = "") -> str:
        if not cfg.lipsync_cmd:
            raise RuntimeError("AUTODRAMA_LIPSYNC_CMD 未配置")
        run_cmd_template(cfg.lipsync_cmd, clip=clip, audio=audio, out=out)
        return out


# ------------------------------------------------------------------ 工厂

def get_image() -> ImageProvider:
    return {"mock": MockImageProvider, "local": LocalImageProvider,
            "command": CommandImageProvider}.get(cfg.image_backend, MockImageProvider)()


def get_video() -> VideoProvider:
    # 视频模型生态变化快，统一走 command 模板（local 视作 command 的别名）
    if cfg.video_backend in ("local", "command"):
        return CommandVideoProvider()
    return MockVideoProvider()


def get_tts() -> TTSProvider:
    if cfg.tts_backend in ("local", "command"):
        return CommandTTSProvider()
    return MockTTSProvider()


def get_lipsync() -> LipsyncProvider:
    if cfg.lipsync_backend in ("local", "command"):
        return CommandLipsyncProvider()
    return MockLipsyncProvider()

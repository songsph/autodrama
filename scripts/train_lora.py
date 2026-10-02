#!/usr/bin/env python3
"""角色 LoRA 训练 —— 主角一致性的最强手段。

什么时候值得训？
  主角（出场镜头多、有特写）值得；只出场几镜的配角用角色卡 + IP-Adapter 就够。
  15~30 张同一个人、背景干净、多角度/多表情的图，训 15~30 分钟即可。

数据来源（两种）：
  1) 从项目读：--project demo --character lin_wan
     自动收集 projects/demo/assets/characters/lin_wan_*.png 与 golden_ref，
     用角色卡里的【锁定】外貌词作为训练 prompt —— 保证 LoRA 学到的就是角色卡描述的那个人。
  2) 手动指定：--name lin_wan --images img1.png img2.png ... --prompt "外貌描述"

输出：<lora 目录>/<角色id>.safetensors，并自动写回角色卡的 lora 字段（--update-project）。
之后出图时流水线会自动为含该角色的镜头启用 LoRA。

用法：
  python scripts/train_lora.py --project demo --character lin_wan
  python scripts/train_lora.py --project demo --character lin_wan --steps 1500 --rank 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from PIL import Image


# ------------------------------------------------------------------ 数据

def collect_from_project(project: str, character: str) -> tuple:
    """从项目资产里收集该角色的参考图与训练 prompt。"""
    root = Path("projects") / project
    pj = root / "project.json"
    if not pj.exists():
        raise FileNotFoundError(f"找不到项目：{pj}")

    data = json.loads(pj.read_text(encoding="utf-8"))
    ch = next((c for c in data.get("characters", [])
               if c.get("id") == character or c.get("name") == character), None)
    if ch is None:
        raise ValueError(f"项目里没有角色 {character}，可选："
                         f"{[c.get('id') for c in data.get('characters', [])]}")

    adir = root / "assets" / "characters"
    imgs = sorted(set(
        list(adir.glob(f"{ch['id']}_*.png")) + list(adir.glob(f"{ch['id']}_*.jpg"))
        + ([Path(ch["golden_ref"])] if ch.get("golden_ref") and
           Path(ch["golden_ref"]).exists() else [])
    ))
    if not imgs:
        raise FileNotFoundError(
            f"没有找到 {character} 的参考图。请先跑 keyframes 阶段生成候选图，"
            f"或手动准备 15~30 张同一个人、背景干净的图后用 --images 指定。")

    # 训练 prompt = 角色卡锁定外貌 + 默认服装，与出图时的锚定词同源
    prompt = " ".join(x for x in (ch.get("appearance", ""), ch.get("wardrobe", "")) if x)
    return [str(p) for p in imgs], (prompt or f"a photo of {ch.get('name', character)}"), ch


def make_dataset(images: list, prompt: str, size: int):
    """极简数据集：所有图共用一个 prompt（角色 LoRA 的标准做法）。"""
    from torch.utils.data import Dataset
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class DS(Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, i):
            img = Image.open(images[i]).convert("RGB")
            return {"pixel_values": tf(img), "prompt": prompt}

    return DS()


# ------------------------------------------------------------------ 训练

def train(images: list, prompt: str, out: str, base: str, steps: int, rank: int,
          lr: float, size: int, batch: int, grad_accum: int, seed: int) -> None:
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from peft import LoraConfig

    torch.manual_seed(seed)
    print(f"[lora] 底模 {base}")
    print(f"[lora] 图片 {len(images)} 张｜prompt：{prompt[:60]}…")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base, torch_dtype=torch.float16, use_safetensors=True)
    pipe.to("cuda")
    vae, unet = pipe.vae, pipe.unet
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    enc1, enc2 = pipe.text_encoder, pipe.text_encoder_2
    for p in list(vae.parameters()) + list(enc1.parameters()) + list(enc2.parameters()):
        p.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)
    for n, p in unet.named_parameters():
        if "lora" in n:
            p.requires_grad_(True)
            p.data = p.data.float()          # LoRA 参数用 fp32 训，稳定性更好

    trainable = [p for p in unet.parameters() if p.requires_grad]
    print(f"[lora] 可训练参数 {sum(p.numel() for p in trainable) / 1e6:.1f}M")
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    noise_sched = DDPMScheduler.from_config(base, subfolder="scheduler")
    ds = make_dataset(images, prompt, size)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2)

    def encode(texts):
        """SDXL 双文本编码器 → (prompt_embeds, pooled_embeds)"""
        t1 = tok1(texts, padding="max_length", max_length=tok1.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to("cuda")
        t2 = tok2(texts, padding="max_length", max_length=tok2.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to("cuda")
        e1 = enc1(t1, output_hidden_states=True).hidden_states[-2]
        o2 = enc2(t2, output_hidden_states=True)
        e2 = o2.hidden_states[-2]
        return torch.cat([e1, e2], dim=-1), o2[0]

    print(f"[lora] 开始训练 {steps} 步…")
    step, done = 0, False
    total = max(1, steps * grad_accum)
    while not done:
        for batch_data in dl:
            px = batch_data["pixel_values"].to("cuda", dtype=torch.float16)
            with torch.no_grad():
                latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
                pe, pooled = encode(list(batch_data["prompt"]))
                noise = torch.randn_like(latents)
                bs = latents.shape[0]
                ts = torch.randint(0, noise_sched.config.num_train_timesteps,
                                   (bs,), device="cuda").long()
                noisy = noise_sched.add_noise(latents, noise, ts)
                # SDXL 必须传的尺寸条件：(原w, 原h, 裁y, 裁x, 目标w, 目标h)
                tids = torch.tensor([size, size, 0, 0, size, size],
                                    device="cuda").unsqueeze(0).repeat(bs, 1)

            pred = unet(noisy, ts, encoder_hidden_states=pe, added_cond_kwargs={
                "text_embeds": pooled, "time_ids": tids}).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"   step {step}/{steps}  loss {loss.item():.4f}")
                if step >= steps:
                    done = True
                    break
        if step == 0:
            raise RuntimeError("数据为空，无法训练")

    # 保存：先由 pipeline 导出标准 LoRA，再拷成 <角色>.safetensors
    tmp_dir = Path(out).parent / "_tmp_lora"
    pipe.save_lora_weights(str(tmp_dir))
    src = next(tmp_dir.glob("*.safetensors"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[lora] 已保存：{out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="项目名（配合 --character 自动取素材）")
    ap.add_argument("--character", help="角色 id 或名字")
    ap.add_argument("--name", help="手动模式下的角色 id")
    ap.add_argument("--images", nargs="*", default=[], help="训练图路径")
    ap.add_argument("--prompt", default="", help="训练 prompt（默认用角色卡外貌词）")
    ap.add_argument("--out", default="", help="输出 safetensors 路径")
    ap.add_argument("--base", default=os.environ.get(
        "AUTODRAMA_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--update-project", action="store_true",
                    help="训练后把路径写回角色卡的 lora 字段")
    args = ap.parse_args()

    ch = None
    if args.project and args.character:
        images, prompt, ch = collect_from_project(args.project, args.character)
        name = ch.get("id") or args.character
        prompt = args.prompt or prompt
    else:
        if not args.images:
            ap.error("需要提供 --project/--character 或 --images")
            return
        images, prompt, name = args.images, args.prompt, (args.name or "character")
        if not prompt:
            ap.error("手动模式必须用 --prompt 给出外貌描述")

    lora_dir = os.environ.get("AUTODRAMA_LORA_DIR") or \
        str(Path(os.environ.get("AUTODRAMA_MODEL_DIR", "models")) / "loras")
    out = args.out or str(Path(lora_dir) / f"{name}.safetensors")

    if len(images) < 8:
        print(f"[warn] 只有 {len(images)} 张图，建议 15~30 张，否则 LoRA 容易过拟合或学不像")

    train(images, prompt, out, args.base, args.steps, args.rank, args.lr,
          args.size, args.batch, args.grad_accum, args.seed)

    if args.update_project and ch is not None and args.project:
        pj = Path("projects") / args.project / "project.json"
        data = json.loads(pj.read_text(encoding="utf-8"))
        for c in data.get("characters", []):
            if c.get("id") == ch.get("id"):
                c["lora"] = out
        pj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[lora] 已写回角色卡：{ch.get('id')} -> {out}")


if __name__ == "__main__":
    main()

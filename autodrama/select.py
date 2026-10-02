"""镜头候选自动选优。

每个镜头生成 N 张候选，用人脸相似度跟角色"黄金参考图"比对，自动挑最像的那张。
这是提升人物一致性性价比最高的一招：完全自动化，不增加人工负担。

依赖：insightface + onnxruntime（requirements-gpu.txt）。
未安装或图里没人脸时自动降级为"取第一张"，流程不会中断。
模型缓存目录指向 cfg.model_dir（D 盘），不占系统盘。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .config import cfg

_face_app = None
_unavailable = False


def _get_app():
    global _face_app, _unavailable
    if _unavailable:
        return None
    if _face_app is None:
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            print(f"[info] 未安装 insightface，跳过自动选优：{e}")
            _unavailable = True
            return None
        try:
            import torch
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if torch.cuda.is_available() else ["CPUExecutionProvider"])
            ctx = 0 if torch.cuda.is_available() else -1
        except Exception:
            providers, ctx = ["CPUExecutionProvider"], -1
        root = str(cfg.model_dir / "insightface")
        _face_app = FaceAnalysis(name="buffalo_l", root=root, providers=providers)
        _face_app.prepare(ctx_id=ctx, det_size=(640, 640))
    return _face_app


def embed(path: str):
    """提取图中最大人脸的特征向量；无人脸返回 None。"""
    app = _get_app()
    if app is None:
        return None
    try:
        import numpy as np
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))
        faces = app.get(img)
        if not faces:
            return None
        # 取面积最大的人脸
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding
    except Exception as e:
        print(f"[warn] 人脸特征提取失败 {path}：{e}")
        return None


def pick_best(candidates: List[str], reference: Optional[str]) -> Tuple[int, List[float]]:
    """从候选中挑出与参考图最像的一张，返回 (索引, 各候选得分)。"""
    if not candidates:
        return -1, []
    if not reference:
        return 0, [0.0] * len(candidates)
    try:
        import numpy as np
    except Exception:
        return 0, [0.0] * len(candidates)

    ref_emb = embed(reference)
    scores: List[float] = []
    for c in candidates:
        emb = embed(c)
        scores.append(float(np.dot(ref_emb, emb)) if (emb is not None and ref_emb is not None)
                      else -1.0)
    if all(s < 0 for s in scores):        # 参考图或候选里没人脸 → 降级
        return 0, scores
    return int(np.argmax(scores)), scores

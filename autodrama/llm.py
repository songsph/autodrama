"""LLM 层：DeepSeek 接入 + Mock 兜底（无 API Key 也能跑通全流程）。"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict

from .config import cfg


def _scan_objects(text: str) -> list:
    """扫描出文本中同层级所有完整的 {...} 片段（忽略字符串内的括号）。"""
    objs, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                objs.append(text[start:i + 1])
                start = -1
    return objs


def extract_json(text: str) -> Dict[str, Any]:
    """从模型输出里稳健地抠出 JSON。

    模型偶尔会：包代码块、加前后缀、或者被 max_tokens 截断。
    截断时不能整段作废——抢救出所有完整的记录，能救多少算多少。
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退化方案 1：截取第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # 退化方案 2：输出被截断时，抢救数组里所有完整的元素
    for key in ("shots", "script", "characters", "scenes", "episodes"):
        kp = text.find(f'"{key}"')
        if kp < 0:
            continue
        ap = text.find("[", kp)
        if ap < 0:
            continue
        items = []
        for frag in _scan_objects(text[ap + 1:]):
            try:
                obj = json.loads(frag)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                items.append(obj)
        if items:
            print(f"[warn] 模型输出被截断，已抢救出 {len(items)} 条完整记录"
                  f"（可调大 DEEPSEEK_MAX_TOKENS 避免）")
            return {key: items}

    # 退化方案 3：单个对象
    for frag in _scan_objects(text):
        try:
            obj = json.loads(frag)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"无法从模型输出中解析 JSON：\n{text[:500]}")


class LLMProvider(ABC):
    @abstractmethod
    def chat_json(self, system: str, user: str, *, mock_kind: str = "",
                  temperature: float = 0.8) -> Dict[str, Any]:
        ...


class DeepSeekLLM(LLMProvider):
    """DeepSeek API（OpenAI 兼容协议）。API Key 只从环境变量读取。"""

    def __init__(self) -> None:
        if not cfg.deepseek_key:
            raise RuntimeError(
                "未设置 DEEPSEEK_API_KEY。\n"
                "安全做法（不落盘）：read -s DEEPSEEK_API_KEY && export DEEPSEEK_API_KEY\n"
                "或复制 .env.example 为 .env 后填入（已被 .gitignore 屏蔽）。"
            )
        from openai import OpenAI
        self.client = OpenAI(api_key=cfg.deepseek_key, base_url=cfg.deepseek_base)
        self.model = cfg.deepseek_model

    def chat_json(self, system: str, user: str, *, mock_kind: str = "",
                  temperature: float = 0.8) -> Dict[str, Any]:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=cfg.llm_max_tokens,     # 推理模型必须给足，否则 content 为空
        )
        try:
            resp = self.client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs)
        except Exception:
            # 部分网关不支持 response_format，退化后再试一次
            resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        if not content:
            # 推理模型偶尔把答案留在 reasoning_content 里
            content = (getattr(msg, "reasoning_content", "") or "").strip()
        if not content:
            raise RuntimeError(
                f"模型返回内容为空（可能被 max_tokens 截断，"
                f"当前 {cfg.llm_max_tokens}，请调大 DEEPSEEK_MAX_TOKENS）")
        return extract_json(content)


# ------------------------------------------------------------------ Mock

_MOCK: Dict[str, Any] = {
    "outline": {
        "title": "《雨夜回声》",
        "logline": "一场车祸后，她发现丈夫的记忆里住着另一个女人。",
        "episodes": [
            {"index": 1, "title": "错位", "synopsis": "林晚在医院醒来，丈夫陈默叫她另一个名字。",
             "hook": "陈默手机里存着一张陌生女人的照片"},
            {"index": 2, "title": "旧照", "synopsis": "林晚翻出结婚照，发现照片角落有第三只手。",
             "hook": "照片背面写着一行陌生的字"},
        ],
    },
    "characters": {
        "characters": [
            {"id": "lin_wan", "name": "林晚", "role": "主角",
             "identity": "28岁，车祸幸存者，外表冷静内心执拗",
             "appearance": "28岁东亚女性，鹅蛋脸，黑色齐肩LOB发型中分，细眉，琥珀色瞳孔，"
                           "鼻梁高挺，唇形偏薄，身材纤瘦，左眉尾有一道浅疤",
             "wardrobe": "米白色双排扣风衣，内搭浅灰高领毛衣",
             "negative": "浓妆，卷发，染发，夸张耳饰，双眼皮贴"},
            {"id": "chen_mo", "name": "陈默", "role": "主角",
             "identity": "32岁，林晚的丈夫，心事重重",
             "appearance": "32岁东亚男性，方脸，黑色寸头，浓眉，单眼皮，下颌线分明，"
                           "身形高大肩宽，右手背有一块陈旧烫伤疤痕",
             "wardrobe": "深灰色羊毛大衣，黑色高领衫",
             "negative": "长发，胡须，戴眼镜，瘦弱"},
        ],
    },
    "scenes": {
        "scenes": [
            {"id": "S01", "name": "医院病房",
             "description": "医院单人病房，病床靠画面右侧，床头柜在右，窗户在画面左侧，"
                            "白色墙面，浅灰地板，天花板日光灯",
             "time_of_day": "深夜", "weather": "雨",
             "light": "冷白色日光灯从正上方打光，窗外雨夜反光从左侧进入",
             "color_tone": "冷青色调，低饱和", "negative": "暖光，杂乱，多人"},
            {"id": "S02", "name": "家中客厅",
             "description": "老旧公寓客厅，米色布艺沙发靠画面左侧，茶几居中，"
                            "电视墙在右侧，木质地板，窗外可见雨",
             "time_of_day": "深夜", "weather": "雨",
             "light": "暖黄台灯从画面左侧打光，其余区域偏暗",
             "color_tone": "暖黄色调，高对比", "negative": "明亮，现代装修，多人"},
        ],
    },
    "script": {
        "title": "错位",
        "synopsis": "林晚在医院醒来，丈夫叫她另一个名字。",
        "script": [
            {"scene_id": "S01", "character": "林晚", "text": "陈默，我这是怎么了？", "emotion": "虚弱困惑"},
            {"scene_id": "S01", "character": "陈默", "text": "别动，医生说你需要休息。", "emotion": "紧张克制"},
            {"scene_id": "S01", "character": "林晚", "text": "你刚才叫我什么？", "emotion": "警觉"},
            {"scene_id": "S01", "character": "陈默", "text": "……我叫你晚晚啊。", "emotion": "心虚"},
            {"scene_id": "S02", "character": "林晚", "text": "那这个人是谁？", "emotion": "压抑的愤怒"},
            {"scene_id": "S02", "character": "陈默", "text": "我不认识她。", "emotion": "回避"},
        ],
    },
    "shots": {
        "shots": [
            {"scene_id": "S01", "characters": [], "shot_size": "全景", "camera": "缓慢推进",
             "action": "空镜，病房全景，病床位于画面右侧，窗外雨夜，冷白灯光",
             "duration": 3.0, "dialogue": []},
            {"scene_id": "S01", "characters": ["lin_wan"], "shot_size": "特写", "camera": "固定",
             "action": "林晚位于画面中心偏右，睫毛颤动，缓缓睁眼，视线看向画左",
             "duration": 3.0,
             "dialogue": [{"character": "林晚", "text": "陈默，我这是怎么了？", "emotion": "虚弱困惑"}]},
            {"scene_id": "S01", "characters": ["chen_mo"], "shot_size": "近景", "camera": "缓慢推进",
             "action": "陈默位于画面左侧，身体前倾，右手按住床沿，眼神躲闪",
             "duration": 3.5,
             "dialogue": [{"character": "陈默", "text": "别动，医生说你需要休息。", "emotion": "紧张克制"}]},
            {"scene_id": "S01", "characters": ["lin_wan"], "shot_size": "大特写", "camera": "固定",
             "action": "林晚眼睛特写，瞳孔骤缩，眉头收紧",
             "duration": 2.5,
             "dialogue": [{"character": "林晚", "text": "你刚才叫我什么？", "emotion": "警觉"}]},
            {"scene_id": "S01", "characters": ["chen_mo"], "shot_size": "近景", "camera": "固定",
             "action": "陈默位于画面左侧，喉结滚动，短暂沉默后开口",
             "duration": 3.0,
             "dialogue": [{"character": "陈默", "text": "……我叫你晚晚啊。", "emotion": "心虚"}]},
            {"scene_id": "S02", "characters": ["lin_wan"], "shot_size": "中近景", "camera": "手持跟随",
             "action": "林晚位于画面右侧，举起手机屏幕朝向画外，手腕微微发抖，面部在暗处",
             "duration": 3.5,
             "dialogue": [{"character": "林晚", "text": "那这个人是谁？", "emotion": "压抑的愤怒"}]},
            {"scene_id": "S02", "characters": ["chen_mo"], "shot_size": "过肩", "camera": "固定",
             "action": "过林晚肩部拍摄，陈默位于画面纵深处，背对光源，侧脸轮廓模糊",
             "duration": 3.0,
             "dialogue": [{"character": "陈默", "text": "我不认识她。", "emotion": "回避"}]},
        ],
    },
    "state": {
        "timeline": "Day 1 深夜 雨后",
        "characters": {
            "lin_wan": {"位置": "家中客厅", "伤势": "头部包扎，右臂擦伤", "服装": "米白风衣",
                        "情绪": "压抑的愤怒", "关系": "怀疑陈默"},
            "chen_mo": {"位置": "家中客厅", "伤势": "无", "服装": "深灰大衣",
                        "情绪": "心虚回避", "关系": "隐瞒林晚"},
        },
        "props": {"陌生女人照片": "陈默手机内，林晚已见过"},
        "hooks_add": ["照片上的女人到底是谁", "陈默为何叫错名字"],
        "hooks_remove": [],
        "facts_add": ["林晚出了车祸并失忆片段", "陈默手机里有陌生女人照片"],
        "summary": "林晚车祸醒来，丈夫陈脱口叫出另一个名字，林晚在其手机中发现陌生女人照片，"
                   "陈默否认认识她。两人关系出现裂痕，林晚开始暗中调查。",
    },
    "check": {"issues": []},
}


class MockLLM(LLMProvider):
    """离线兜底：不调 API 也能把流水线跑通，用于开发调试与 CI。"""

    def chat_json(self, system: str, user: str, *, mock_kind: str = "",
                  temperature: float = 0.8) -> Dict[str, Any]:
        if mock_kind in _MOCK:
            return json.loads(json.dumps(_MOCK[mock_kind]))  # 深拷贝，避免污染
        raise ValueError(f"MockLLM 缺少 mock_kind={mock_kind!r} 的样例数据")


def get_llm() -> LLMProvider:
    if cfg.llm_backend == "mock":
        return MockLLM()
    try:
        return DeepSeekLLM()
    except Exception as e:                      # 缺 key / 网络不通时自动降级
        print(f"[warn] DeepSeek 不可用，降级为 Mock：{e}")
        return MockLLM()

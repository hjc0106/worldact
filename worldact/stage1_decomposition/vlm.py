from __future__ import annotations

import base64
import io
import json
from typing import Any

import httpx
import torch
from PIL import Image
from omegaconf import OmegaConf

from worldact.config import DEFAULT_VLM_SYSTEM_PROMPT, DEFAULT_VLM_USER_PROMPT
from worldact.data.gaussians import RenderedView
from worldact.data.objects import ObjectMemoryState


def _view_to_data_url(view: RenderedView, max_side: int = 768) -> str:
    arr = (view.rgb.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
    image = Image.fromarray(arr)
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _render_prompt_template(template: str, values: dict[str, int]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def _chat_completions_url(api_base: str) -> str:
    base = str(api_base).rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_api_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err is not None:
            return str(err)
    return response.text.strip() or f"HTTP {response.status_code}"


def _vision_payload_unsupported(message: str) -> bool:
    lowered = message.lower()
    return "image_url" in lowered or "unknown variant" in lowered or "multimodal" in lowered and "not support" in lowered


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _find_json_array(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("VLM response was empty; expected a JSON array")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text)
        if isinstance(value, list):
            return text
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    while start >= 0:
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("[", start + 1)
            continue
        if isinstance(value, list):
            return text[start : start + end]
        start = text.find("[", start + 1)

    snippet = text[:500].replace("\n", "\\n")
    raise ValueError(f"VLM response did not contain a JSON array. Response starts with: {snippet!r}")


class VisionNotSupportedError(RuntimeError):
    """Raised when the configured chat endpoint rejects image_url content blocks."""


class VLMClient:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        if not cfg.api_base or not cfg.api_key:
            raise ValueError("VLM api_base/api_key are required. Use --objects to bypass VLM.")

    def analyze_keyframes(self, views: list[RenderedView]) -> list[ObjectMemoryState]:
        system, text = self._build_prompts(num_keyframes=len(views))
        text = (
            f"{text}\n\n"
            "Return exactly one JSON array and nothing else. "
            "The first non-whitespace character must be '[' and the last must be ']'."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        image_max_side = int(getattr(self.cfg, "image_max_side", 768))
        content.extend({"type": "image_url", "image_url": {"url": _view_to_data_url(v, image_max_side)}} for v in views)
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
        }
        url = _chat_completions_url(str(self.cfg.api_base))
        last_error: Exception | None = None
        for _ in range(self.cfg.max_retries):
            try:
                response = httpx.post(
                    url,
                    headers={"Authorization": f"Bearer {self.cfg.api_key}"},
                    json=payload,
                    timeout=self.cfg.timeout,
                )
                if response.status_code >= 400:
                    api_error = _extract_api_error(response)
                    if _vision_payload_unsupported(api_error):
                        raise VisionNotSupportedError(
                            "Configured VLM endpoint does not accept vision/image_url payloads. "
                            "DeepSeek chat completions are text-only. Bypass VLM with "
                            "`--objects ...` / `OBJECTS=...`, or point stage1.vlm.api_base to a "
                            f"vision-capable model (e.g. OpenAI gpt-4o). API error: {api_error}"
                        )
                    response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                content_text = _message_content_to_text(message.get("content"))
                if not content_text.strip():
                    content_text = _message_content_to_text(message.get("reasoning_content"))
                return self._parse_inventory(content_text)
            except VisionNotSupportedError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                    api_error = _extract_api_error(exc.response)
                    if _vision_payload_unsupported(api_error):
                        raise VisionNotSupportedError(
                            "Configured VLM endpoint does not accept vision/image_url payloads. "
                            "Bypass VLM with `--objects ...`, or use a vision-capable api_base. "
                            f"API error: {api_error}"
                        ) from exc
        raise RuntimeError(f"VLM object discovery failed: {last_error}") from last_error

    def _build_prompts(self, *, num_keyframes: int | None = None) -> tuple[str, str]:
        prompts = getattr(self.cfg, "prompts", None)
        if prompts is None:
            prompts = {}
        if OmegaConf.is_config(prompts):
            prompt_cfg = OmegaConf.to_container(prompts, resolve=True)
        elif isinstance(prompts, dict):
            prompt_cfg = prompts
        else:
            prompt_cfg = {
                "system": getattr(prompts, "system", ""),
                "user": getattr(prompts, "user", ""),
            }

        system_template = str(prompt_cfg.get("system") or DEFAULT_VLM_SYSTEM_PROMPT).strip()
        user_template = str(prompt_cfg.get("user") or DEFAULT_VLM_USER_PROMPT).strip()
        values = {
            "max_objects": int(getattr(self.cfg, "max_objects", 20)),
            "num_keyframes": int(num_keyframes if num_keyframes is not None else getattr(self.cfg, "num_keyframes", 0)),
        }
        return _render_prompt_template(system_template, values), _render_prompt_template(user_template, values)

    def _parse_inventory(self, content: str) -> list[ObjectMemoryState]:
        data = json.loads(_find_json_array(content))
        if not isinstance(data, list):
            raise ValueError("VLM response must be a JSON array")
        objects: list[ObjectMemoryState] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            portable = bool(item.get("portable", True))
            recognizability = str(item.get("recognizability", "precise"))
            name = str(item.get("name", "")).strip()
            if not portable or recognizability == "unrecognizable" or not name:
                continue
            objects.append(
                ObjectMemoryState(
                    name=name,
                    category=str(item.get("category", name)),
                    recognizability=recognizability,
                    is_portable=portable,
                    text_prompt=str(item.get("text_prompt", name)).strip() or name,
                )
            )
        return objects


def manual_memory_states(prompts: list[str]) -> list[ObjectMemoryState]:
    return [
        ObjectMemoryState(name=p, category=p, recognizability="precise", is_portable=True, text_prompt=p)
        for p in prompts
    ]

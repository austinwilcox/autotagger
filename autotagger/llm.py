"""Local-LLM client. Speaks both OpenAI-compatible chat and Ollama's native API.

Any of these work out of the box:

    Ollama        --llm-url http://localhost:11434            (native, default)
    LM Studio     --llm-url http://localhost:1234/v1          (OpenAI-compatible)
    llama.cpp     --llm-url http://localhost:8080/v1
    vLLM / TGI    --llm-url http://host:8000/v1
    Anything else that exposes /v1/chat/completions

The native Ollama path is preferred when available because its `format` field
takes a JSON Schema and constrains decoding to it — a 4B model that would
otherwise wander into prose returns valid JSON every time. The OpenAI path falls
back to `response_format: json_object` plus tolerant parsing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
# Ollama's thinking models wrap reasoning in <think>...</think> before the answer.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(Exception):
    pass


@dataclass
class LLMConfig:
    url: str = "http://localhost:11434"
    model: str = "qwen3:latest"
    api: str = "auto"            # "auto" | "ollama" | "openai"
    api_key: str | None = None
    temperature: float = 0.0     # determinism matters more than flair here
    timeout: float = 180.0
    num_ctx: int | None = 8192   # Ollama only; prompts here run ~2-4k tokens
    max_tokens: int = 2048
    think: bool = False          # hybrid models (qwen3, gpt-oss) reason before answering


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base = config.url.rstrip("/")
        self.api = config.api if config.api != "auto" else self._detect_api()
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self.client = httpx.Client(timeout=config.timeout, headers=headers)

    def _detect_api(self) -> str:
        """An OpenAI-compatible base URL ends in /v1; everything else we assume is Ollama."""
        return "openai" if self.base.endswith("/v1") else "ollama"

    def health(self) -> tuple[bool, str]:
        """Check the endpoint is up and the model exists. Returns (ok, message)."""
        try:
            if self.api == "ollama":
                r = self.client.get(f"{self.base}/api/tags", timeout=10)
                r.raise_for_status()
                names = [m["name"] for m in r.json().get("models", [])]
                if self.config.model not in names and not any(
                    n.split(":")[0] == self.config.model.split(":")[0] for n in names
                ):
                    return False, (
                        f"model {self.config.model!r} not found. Available: "
                        + (", ".join(names) or "(none)")
                    )
                return True, f"ollama ok, model {self.config.model}"
            r = self.client.get(f"{self.base}/models", timeout=10)
            r.raise_for_status()
            return True, f"openai-compatible endpoint ok at {self.base}"
        except httpx.HTTPError as exc:
            return False, f"cannot reach {self.base}: {exc}"

    # -- completion --------------------------------------------------------

    def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        retries: int = 2,
    ) -> dict[str, Any]:
        """Get a JSON object back, repairing and retrying if the model misbehaves."""
        convo = list(messages)
        last_error = ""
        for attempt in range(retries + 1):
            raw = self._chat(convo, schema)
            parsed = _extract_json(raw)
            if parsed is not None:
                missing = [k for k in schema.get("required", []) if k not in parsed]
                if not missing:
                    return parsed
                last_error = f"missing required key(s): {', '.join(missing)}"
            else:
                last_error = "response was not valid JSON"

            if attempt == retries:
                break
            log.debug("LLM retry %s: %s", attempt + 1, last_error)
            convo = convo + [
                {"role": "assistant", "content": raw[:2000]},
                {
                    "role": "user",
                    "content": (
                        f"That response was rejected: {last_error}. "
                        "Reply with ONE valid JSON object and nothing else — no prose, "
                        "no markdown fences. Required keys: "
                        + ", ".join(schema.get("required", []))
                    ),
                },
            ]
        raise LLMError(f"LLM did not return usable JSON after {retries + 1} attempts: {last_error}")

    def _chat(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        if self.api == "ollama":
            return self._chat_ollama(messages, schema)
        return self._chat_openai(messages, schema)

    def _chat_ollama(self, messages, schema) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            # Constrained decoding against the schema: the reason this path is preferred.
            "format": schema,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if self.config.num_ctx:
            payload["options"]["num_ctx"] = self.config.num_ctx
        # Hybrid reasoning models (qwen3, gpt-oss, deepseek-r1) emit a long
        # `thinking` block that counts against num_predict, which truncates the
        # JSON before it is finished. The decision procedure in the prompt is
        # already an explicit step-by-step, so free-form reasoning buys nothing
        # here and costs a lot of tokens.
        if not self.config.think:
            payload["think"] = False

        try:
            r = self.client.post(f"{self.base}/api/chat", json=payload)
            if r.status_code == 400 and "think" in payload:
                # Model has no thinking mode; ollama rejects the field outright.
                payload.pop("think")
                r = self.client.post(f"{self.base}/api/chat", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc

        body = r.json()
        message = body.get("message") or {}
        content = message.get("content") or ""
        if not content and body.get("done_reason") == "length":
            raise LLMError(
                f"{self.config.model} hit the {self.config.max_tokens}-token limit before "
                "emitting any JSON. Raise --llm-max-tokens, or pick a model with a "
                "shorter reasoning preamble."
            )
        return content

    def _chat_openai(self, messages, schema) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            r = self.client.post(f"{self.base}/chat/completions", json=payload)
            if r.status_code == 400:
                # Some servers reject response_format outright; the prompt already
                # demands bare JSON, so drop it and rely on parsing.
                payload.pop("response_format", None)
                r = self.client.post(f"{self.base}/chat/completions", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"chat completion failed: {exc}") from exc
        choices = r.json().get("choices") or []
        if not choices:
            raise LLMError("chat completion returned no choices")
        return choices[0].get("message", {}).get("content", "")

    def close(self) -> None:
        self.client.close()


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of whatever the model actually emitted."""
    if not text:
        return None
    cleaned = _THINK.sub("", text)
    cleaned = _FENCE.sub("", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} span in the output.
    start = cleaned.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    return None

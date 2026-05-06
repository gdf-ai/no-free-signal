"""Thin shim over Bedrock's Converse API for non-Anthropic models.

The existing LLM controller uses the official ``anthropic.AnthropicBedrock``
SDK, which only handles Claude model IDs. Nova and Llama models go through
Bedrock's standard Converse API via boto3. This module wraps that API in
the same minimal surface the controller already calls — ``client.messages
.create(model=..., system=..., messages=..., max_tokens=..., temperature
=...)`` returning an object with ``.content[0].text`` — so the call site
doesn't have to branch on provider.

Used for ``us.amazon.nova-*`` and ``us.meta.llama-*`` model ids. Anthropic
models continue to use the dedicated SDK.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _BedrockContentBlock:
    text: str


@dataclass
class _BedrockResponse:
    content: list[_BedrockContentBlock]


class _BedrockMessages:
    """Mimics ``anthropic.AnthropicBedrock().messages``."""

    def __init__(self, runtime: Any):
        self._runtime = runtime

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        system: str,
        messages: list[dict[str, Any]],
    ) -> _BedrockResponse:
        # Converse expects ``content`` as a list of typed blocks.
        converse_messages: list[dict[str, Any]] = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                blocks = [{"text": content}]
            elif isinstance(content, list):
                blocks = content
            else:
                blocks = [{"text": str(content)}]
            converse_messages.append({"role": m["role"], "content": blocks})

        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        resp = self._runtime.converse(**kwargs)
        text = ""
        try:
            blocks = resp["output"]["message"]["content"]
            for blk in blocks:
                if "text" in blk:
                    text += blk["text"]
        except (KeyError, IndexError, TypeError):
            text = ""
        return _BedrockResponse(content=[_BedrockContentBlock(text=text)])


class BedrockClient:
    """Drop-in replacement for ``AnthropicBedrock`` covering Nova / Llama."""

    def __init__(self, region: str | None = None):
        import boto3
        kwargs: dict[str, Any] = {}
        if region:
            kwargs["region_name"] = region
        self._runtime = boto3.client("bedrock-runtime", **kwargs)
        self.messages = _BedrockMessages(self._runtime)

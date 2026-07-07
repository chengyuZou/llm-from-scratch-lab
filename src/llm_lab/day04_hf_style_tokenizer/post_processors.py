from __future__ import annotations

from typing import Protocol

from llm_lab.day04_hf_style_tokenizer.core import ByteToken, Encoding
from llm_lab.day04_hf_style_tokenizer.models import BPEModel


class PostProcessor(Protocol):
    def process(self, encoding: Encoding, model: BPEModel) -> Encoding:
        ...


class NoPostProcessor:
    def process(self, encoding: Encoding, model: BPEModel) -> Encoding:
        return encoding


class TemplatePostProcessor:
    """
    简化版 TemplateProcessing。

    prefix/suffix 中的 token 都作为 special token 原子插入。
    """

    def __init__(self, prefix: list[str] | None = None, suffix: list[str] | None = None):
        self.prefix = prefix or []
        self.suffix = suffix or []

    def process(self, encoding: Encoding, model: BPEModel) -> Encoding:
        prefix_tokens = [token.encode("utf-8") for token in self.prefix]
        suffix_tokens = [token.encode("utf-8") for token in self.suffix]

        for token in self.prefix + self.suffix:
            model.add_special_token(token)

        tokens: list[ByteToken] = prefix_tokens + encoding.tokens + suffix_tokens
        ids = model.ids_for_tokens(tokens)

        return Encoding(
            ids=ids,
            tokens=tokens,
            offsets=[None] * len(prefix_tokens) + encoding.offsets + [None] * len(suffix_tokens),
            special_tokens_mask=[1] * len(prefix_tokens)
            + encoding.special_tokens_mask
            + [1] * len(suffix_tokens),
            normalized_text=encoding.normalized_text,
            pre_tokens=encoding.pre_tokens,
        )

from __future__ import annotations

import unicodedata
from typing import Protocol


class Normalizer(Protocol):
    def normalize(self, text: str) -> str:
        ...


class IdentityNormalizer:
    def normalize(self, text: str) -> str:
        return text


class NFKCNormalizer:
    """
    Unicode NFKC normalization。

    真实 tokenizer 会认真处理 offset alignment。Day04 先只保留文本变换本身。
    """

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)


class StripNormalizer:
    def normalize(self, text: str) -> str:
        return text.strip()


class SequenceNormalizer:
    def __init__(self, normalizers: list[Normalizer]):
        self.normalizers = normalizers

    def normalize(self, text: str) -> str:
        normalized = text
        for normalizer in self.normalizers:
            normalized = normalizer.normalize(normalized)
        return normalized

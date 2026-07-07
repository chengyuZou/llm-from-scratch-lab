from __future__ import annotations

import re
from typing import Protocol

from llm_lab.day04_hf_style_tokenizer.core import PreToken


class PreTokenizer(Protocol):
    def pre_tokenize(self, text: str, base_offset: int = 0) -> list[PreToken]:
        ...


class ByteLevelPreTokenizer:
    """
    保留空白片段的简化 byte-level pre-tokenizer。

    真实 GPT-2/tiktoken 还会做 byte-to-unicode 映射和复杂正则。
    这里先保证 roundtrip，不丢空格。
    """

    _PATTERN = re.compile(r"\s+|\S+", re.UNICODE)

    def pre_tokenize(self, text: str, base_offset: int = 0) -> list[PreToken]:
        return [
            PreToken(
                text=match.group(),
                offset=(base_offset + match.start(), base_offset + match.end()),
            )
            for match in self._PATTERN.finditer(text)
        ]


class SpecialAwarePreTokenizer:
    """
    把 added special tokens 作为原子片段保留下来，再让普通文本走 base pre-tokenizer。

    这是工业 tokenizer 很关键的一层：<|im_start|> 不能被 BPE 拆成 '<', '|', 'im'...
    """

    def __init__(self, base: PreTokenizer, special_tokens: list[str]):
        self.base = base
        self.special_tokens = sorted(special_tokens, key=len, reverse=True)

    def pre_tokenize(self, text: str, base_offset: int = 0) -> list[PreToken]:
        if not self.special_tokens:
            return self.base.pre_tokenize(text, base_offset)

        pieces: list[PreToken] = []
        cursor = 0

        while cursor < len(text):
            match = self._find_next_special(text, cursor)

            if match is None:
                pieces.extend(self.base.pre_tokenize(text[cursor:], base_offset + cursor))
                break

            start, end, token = match

            if start > cursor:
                pieces.extend(self.base.pre_tokenize(text[cursor:start], base_offset + cursor))

            pieces.append(
                PreToken(
                    text=token,
                    offset=(base_offset + start, base_offset + end),
                    is_special=True,
                )
            )
            cursor = end

        return pieces

    def _find_next_special(self, text: str, start_at: int) -> tuple[int, int, str] | None:
        best: tuple[int, int, str] | None = None

        for token in self.special_tokens:
            start = text.find(token, start_at)
            if start == -1:
                continue

            end = start + len(token)
            candidate = (start, end, token)

            if best is None or candidate[0] < best[0] or (
                candidate[0] == best[0] and len(candidate[2]) > len(best[2])
            ):
                best = candidate

        return best

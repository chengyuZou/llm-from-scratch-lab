from __future__ import annotations

from dataclasses import dataclass


ByteToken = bytes
Offset = tuple[int, int]


@dataclass(frozen=True)
class PreToken:
    """
    pre-tokenizer 输出的一个文本片段。

    is_special=True 表示这个片段是 added special token，不能再交给 BPE 拆碎。
    """

    text: str
    offset: Offset
    is_special: bool = False


@dataclass(frozen=True)
class Encoding:
    """
    类似 Hugging Face tokenizers 的 Encoding，但只保留 Day04 需要观察的字段。
    """

    ids: list[int]
    tokens: list[ByteToken]
    offsets: list[Offset | None]
    special_tokens_mask: list[int]
    normalized_text: str
    pre_tokens: list[PreToken]

    def token_hex(self) -> list[str]:
        return [token.hex(" ") for token in self.tokens]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

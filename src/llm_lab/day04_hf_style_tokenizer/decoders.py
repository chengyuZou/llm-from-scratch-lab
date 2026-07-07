from __future__ import annotations

from llm_lab.day04_hf_style_tokenizer.core import ByteToken, Encoding


class ByteLevelDecoder:
    def decode(self, encoding: Encoding, skip_special_tokens: bool = False) -> str:
        tokens: list[ByteToken] = []

        for token, is_special in zip(encoding.tokens, encoding.special_tokens_mask):
            if skip_special_tokens and is_special:
                continue
            tokens.append(token)

        return b"".join(tokens).decode("utf-8")

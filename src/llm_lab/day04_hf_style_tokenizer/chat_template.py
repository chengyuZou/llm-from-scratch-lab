from __future__ import annotations

from llm_lab.day04_hf_style_tokenizer.core import ChatMessage


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"


QWEN_LIKE_SPECIAL_TOKENS = [
    END_OF_TEXT,
    IM_START,
    IM_END,
]


class QwenLikeChatTemplate:
    """
    Qwen 系 chat template 的教学版。

    真实 tokenizer_config.json 里的 Jinja template 更完整，会处理 tool call 等更多角色。
    这里先保留最关键的 role boundary 和 generation prompt。
    """

    def render(self, messages: list[ChatMessage], add_generation_prompt: bool = True) -> str:
        chunks: list[str] = []

        for message in messages:
            chunks.append(f"{IM_START}{message.role}\n{message.content}{IM_END}\n")

        if add_generation_prompt:
            chunks.append(f"{IM_START}assistant\n")

        return "".join(chunks)

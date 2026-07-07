from __future__ import annotations

from rich.console import Console
from rich.table import Table

from llm_lab.day04_hf_style_tokenizer.chat_template import QwenLikeChatTemplate
from llm_lab.day04_hf_style_tokenizer.core import ChatMessage
from llm_lab.day04_hf_style_tokenizer.factory import build_qwen_like_tokenizer


console = Console()


def print_encoding(title: str, text: str) -> None:
    tokenizer = build_qwen_like_tokenizer()
    encoding = tokenizer.encode(text)

    table = Table(title=title)
    table.add_column("field")
    table.add_column("value")

    table.add_row("text", repr(text))
    table.add_row("pre_tokens", repr(encoding.pre_tokens))
    table.add_row("ids", repr(encoding.ids))
    table.add_row("token_hex", repr(encoding.token_hex()))
    table.add_row("special_mask", repr(encoding.special_tokens_mask))
    table.add_row("decoded_keep", repr(tokenizer.decode(encoding, skip_special_tokens=False)))
    table.add_row("decoded_skip", repr(tokenizer.decode(encoding, skip_special_tokens=True)))

    console.print(table)


def main() -> None:
    console.rule("[bold cyan]Day04 HF-style Tokenizer Pipeline[/bold cyan]")

    print_encoding("Plain Encoding", "hello 你好 Agent🤖")

    chat_template = QwenLikeChatTemplate()
    prompt = chat_template.render(
        [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="解释 tokenizer。"),
        ],
        add_generation_prompt=True,
    )

    print_encoding("Qwen-like Chat Template Encoding", prompt)


if __name__ == "__main__":
    main()

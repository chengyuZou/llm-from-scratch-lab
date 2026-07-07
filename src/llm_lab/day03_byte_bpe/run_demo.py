from __future__ import annotations

from rich.console import Console
from rich.table import Table

from llm_lab.day03_byte_bpe.byte_bpe import (
    ByteLevelBPETokenizer,
    byte_tokens_to_text,
    format_byte_token,
    text_to_byte_tokens,
)


console = Console()


TEXTS = [
    "hello",
    "你好",
    "hello你好",
    "😀",
    "Agent🤖",
]

BPE_CORPUS = [
    "hello",
    "hello你好",
    "你好",
    "Agent🤖",
]


def print_byte_roundtrip() -> None:
    table = Table(title="UTF-8 Byte Roundtrip")
    table.add_column("text")
    table.add_column("utf8_hex")
    table.add_column("byte_tokens_hex")
    table.add_column("formatted")
    table.add_column("decoded")

    for text in TEXTS:
        byte_tokens = text_to_byte_tokens(text)
        formatted = [
            format_byte_token(token)
            for token in byte_tokens
        ]
        decoded = byte_tokens_to_text(byte_tokens)

        table.add_row(
            repr(text),
            text.encode("utf-8").hex(" "),
            repr([format_byte_token(token) for token in byte_tokens]),
            repr(formatted),
            repr(decoded),
        )

    console.print(table)


def print_training_steps(tokenizer: ByteLevelBPETokenizer) -> None:
    table = Table(title="Byte-level BPE Merge Steps")
    table.add_column("step", justify="right")
    table.add_column("best_pair")
    table.add_column("frequency", justify="right")
    table.add_column("new_token")
    table.add_column("splits")

    for step in tokenizer.training_steps:
        table.add_row(
            str(step.step),
            repr(tuple(format_byte_token(token) for token in step.best_pair)),
            str(step.frequency),
            repr(format_byte_token(step.new_token)),
            repr([
                [format_byte_token(token) for token in tokens]
                for tokens in step.splits
            ]),
        )

    console.print(table)


def print_tokenization_examples(tokenizer: ByteLevelBPETokenizer) -> None:
    table = Table(title="Byte-level BPE Tokenization Examples")
    table.add_column("text")
    table.add_column("tokens(hex)")
    table.add_column("ids")
    table.add_column("decoded")

    for text in TEXTS:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        table.add_row(
            repr(text),
            repr([format_byte_token(token) for token in tokens]),
            repr(ids),
            repr(decoded),
        )

    console.print(table)


def main() -> None:
    console.rule("[bold cyan]Day03 Byte-level BPE[/bold cyan]")

    print_byte_roundtrip()

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(BPE_CORPUS, num_merges=8)

    print_training_steps(tokenizer)
    print_tokenization_examples(tokenizer)


if __name__ == "__main__":
    main()

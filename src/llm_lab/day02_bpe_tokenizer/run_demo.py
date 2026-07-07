from __future__ import annotations

from rich.console import Console
from rich.table import Table

from llm_lab.day02_bpe_tokenizer.naive_bpe import NaiveBPETokenizer


console = Console()


CORPUS = [
    "low",
    "lower",
    "lowest",
]


TEST_TEXTS = [
    "low",
    "lower",
    "lowest",
    "slow",
    "tokenizer",
]


def print_training_steps(tokenizer: NaiveBPETokenizer) -> None:
    """
    展示每一轮 BPE merge 的过程。

    这个函数只负责展示，不负责训练算法。
    你实现 train() 后，self.training_steps 应该有数据。
    """
    table = Table(title="BPE Merge Steps")
    table.add_column("step", justify="right")
    table.add_column("best_pair")
    table.add_column("frequency", justify="right")
    table.add_column("new_token")
    table.add_column("splits")

    for step in tokenizer.training_steps:
        table.add_row(
            str(step.step),
            repr(step.best_pair),
            str(step.frequency),
            repr(step.new_token),
            repr(step.splits),
        )

    console.print(table)


def print_tokenization_examples(tokenizer: NaiveBPETokenizer, texts: list[str]) -> None:
    """
    展示训练完成后，新文本如何被已有 merge rules 编码。
    """
    table = Table(title="BPE Tokenization Examples")
    table.add_column("text")
    table.add_column("tokens")
    table.add_column("ids")
    table.add_column("decoded")

    for text in texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)

        table.add_row(
            repr(text),
            repr(tokens),
            repr(ids),
            repr(decoded),
        )

    console.print(table)


def main() -> None:
    console.rule("[bold cyan]Day02 Naive BPE[/bold cyan]")

    tokenizer = NaiveBPETokenizer()

    tokenizer.train(CORPUS, num_merges=5)

    print_training_steps(tokenizer)
    print_tokenization_examples(tokenizer, TEST_TEXTS)


if __name__ == "__main__":
    main()

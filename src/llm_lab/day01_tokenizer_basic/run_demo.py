from rich.console import Console
from rich.table import Table

from llm_lab.day01_tokenizer_basic.char_tokenizer import CharTokenizer
from llm_lab.day01_tokenizer_basic.word_tokenizer import WordTokenizer

console = Console()

# Day01 用一小段固定 corpus 构建 vocab。
# 它故意包含中文、英文、LLM 术语和代码，方便观察不同 tokenizer 的取舍。
CORPUS = """
我正在学习大模型。
LLM tokenizers convert text into token ids.
Tokenizer is the first gate of a language model.
def get_user_profile(user_id): return db.query(user_id)
"""

# 测试文本
TEST_TEXTS = [
    "我正在学习大模型。",
    "LLM tokenizer",
    "tokenizer",
    "tokenization",
    "def get_user_profile(user_id): return db.query(user_id)",
    "Hello, world!",
]


def inspect_tokenizer(name: str, tokenizer: CharTokenizer | WordTokenizer, texts: list[str]) -> None:
    # rich.Table 只负责把观察结果排成表格，不参与 tokenizer 逻辑。
    console.rule(f"[bold cyan]{name}[/bold cyan]")
    console.print(f"Vocab size: {tokenizer.vocab_size}\n")

    table = Table(title=name)
    table.add_column("text")
    table.add_column("tokens")
    table.add_column("ids")
    table.add_column("id_tokens")
    table.add_column("decoded_keep")
    table.add_column("decoded_skip")

    for text in texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text, add_special_tokens=True)
        id_tokens = tokenizer.convert_ids_to_tokens(ids)
        # keep 用来观察结构 token；skip 用来观察展示给人的正文。
        decoded_keep = tokenizer.decode(ids, skip_special_tokens=False)
        decoded_skip = tokenizer.decode(ids, skip_special_tokens=True)

        table.add_row(
            repr(text),
            repr(tokens),
            repr(ids),
            repr(id_tokens),
            repr(decoded_keep),
            repr(decoded_skip),
        )
    console.print(table)


def main() -> None:
    # 两个 tokenizer 使用同一份 corpus，方便横向比较 token 粒度和 OOV 行为。
    char_tokenizer = CharTokenizer(CORPUS)
    word_tokenizer = WordTokenizer(CORPUS)

    inspect_tokenizer("Char Tokenizer", char_tokenizer, TEST_TEXTS)
    inspect_tokenizer("Word Tokenizer", word_tokenizer, TEST_TEXTS)

if __name__ == "__main__":
    main()
    

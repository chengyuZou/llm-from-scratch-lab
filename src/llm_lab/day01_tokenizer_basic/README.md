# Day 01: Tokenizer Basic

今天只做 tokenizer 的最小闭环，不碰 Hugging Face，也不碰 BPE。

目标不是实现一个现代 LLM tokenizer，而是先把 tokenizer 的骨架拆清楚：

```text
text
  -> tokenize(text)
tokens
  -> convert tokens to ids
token ids
  -> convert ids to tokens
tokens
  -> decode(ids)
text
```

现代 tokenizer 再复杂，也绕不开这个闭环。后面的 BPE、Byte-level BPE、SentencePiece、Qwen tokenizer、DeepSeek tokenizer、GLM tokenizer，本质上都是在这个骨架上增加更强的切分规则、更大的词表、更严格的特殊 token 协议和更快的工程实现。

## Module Split

Day01 只包含三个文件：

```text
day01_tokenizer_basic/
├── char_tokenizer.py
├── word_tokenizer.py
└── run_demo.py
```

职责边界：

| File | Role |
| --- | --- |
| `char_tokenizer.py` | 字符级 tokenizer。用于观察“几乎不 OOV，但序列更长”的问题。 |
| `word_tokenizer.py` | 粗糙 word-level tokenizer。用于观察“序列更短，但容易 `<unk>`，decode 难还原”的问题。 |
| `run_demo.py` | 对比两个 tokenizer 在中文、英文、代码、未知词上的行为。 |

不要在 Day01 引入 BPE、Hugging Face、模型权重、embedding 或训练逻辑。Day01 的边界必须保持很小。

## Facade Interface

两个 tokenizer 都应该暴露同一组接口。

```python
class TokenizerLike:
    pad_token: str
    unk_token: str
    bos_token: str
    eos_token: str

    pad_token_id: int
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int

    @property
    def vocab_size(self) -> int: ...

    def tokenize(self, text: str) -> list[str]: ...

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]: ...

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str: ...

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]: ...

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...
```

这里的 `TokenizerLike` 只是 README 里的接口说明，不需要你今天真的写成抽象基类。等后面多个 tokenizer 实现变多了，再考虑抽公共协议。

## Data Flow

```mermaid
flowchart LR
    A["corpus"] --> B["build vocab"]
    B --> C["token_to_id"]
    B --> D["id_to_token"]
    E["input text"] --> F["tokenize"]
    F --> G["tokens"]
    G --> H["encode"]
    H --> I["token ids"]
    I --> J["decode"]
    J --> K["output text"]
```

核心点：

1. `corpus` 只负责构建词表。
2. `tokenize(text)` 只负责切文本，不负责转 id。
3. `encode(text)` 负责 `text -> tokens -> ids`。
4. `decode(ids)` 负责 `ids -> tokens -> text`。
5. 未知 token 必须映射到 `<unk>`。
6. `add_special_tokens=True` 时，`encode` 应该在首尾加 `<bos>` 和 `<eos>`。

## Special Tokens

Day01 固定使用这四个 special tokens：

| Token | Meaning |
| --- | --- |
| `<pad>` | padding token，用于未来批处理时补齐长度。Day01 暂时只保留概念。 |
| `<unk>` | unknown token，表示词表里不存在的字符或片段。 |
| `<bos>` | begin of sequence，序列开始。 |
| `<eos>` | end of sequence，序列结束。 |

建议固定顺序：

```python
special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
```

这样 id 稳定，测试也更容易写。

## CharTokenizer

字符级 tokenizer 的规则非常简单：

```text
"tokenizer" -> ["t", "o", "k", "e", "n", "i", "z", "e", "r"]
"我喜欢LLM" -> ["我", "喜", "欢", "L", "L", "M"]
```

实现逻辑：

1. 从 `corpus` 中收集所有出现过的字符。
2. 按稳定顺序构建 vocab。
3. 每个字符映射到一个 id。
4. 输入里遇到 corpus 中没出现过的字符时，映射为 `<unk>`。

需要观察的 tradeoff：

| Advantage | Cost |
| --- | --- |
| 不容易遇到未知词，只要字符见过就能编码。 | 英文、代码、长词会被切得很碎，token 序列变长。 |
| decode 通常更容易还原原文。 | 完全没有词根、词片段、代码结构的概念。 |

注意：如果一个汉字没有出现在 `corpus` 中，它仍然会变成 `<unk>`。字符级 tokenizer 不是永远没有未知 token。

## WordTokenizer

word-level tokenizer 这一天只做粗糙版本。

推荐正则：

```python
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
```

大致效果：

```text
"Hello, world!" -> ["Hello", ",", "world", "!"]
"def get_user_profile(user_id):" -> ["def", "get_user_profile", "(", "user_id", ")", ":"]
```

它的问题要故意暴露出来：

| Problem | Example |
| --- | --- |
| 词表外词会变成 `<unk>`。 | corpus 有 `token`，输入 `tokenization`，可能直接 OOV。 |
| decode 很难完美恢复空格。 | `Hello, world!` 可能还原成 `Hello , world !`。 |
| 对中文不友好。 | `\w+` 在 Python Unicode 正则下可能把连续中文当成一个片段。 |
| 对代码结构理解很浅。 | `get_user_profile` 只是一个正则片段，不是真正的语义切分。 |

这里不要为了“看起来聪明”去修太多规则。Day01 的目的就是让你看到 word-level tokenizer 为什么不够用。

## Demo Requirements

`run_demo.py` 应该构造一个小 corpus，至少包含：

1. 中文句子。
2. 英文句子。
3. LLM/tokenizer 技术词。
4. 一小段代码。

建议测试文本：

```python
TEST_TEXTS = [
    "我正在学习大模型。",
    "LLM tokenizer",
    "tokenizer",
    "tokenization",
    "def get_user_profile(user_id): return db.query(user_id)",
    "Hello, world!",
]
```

demo 输出至少包含：

| Column | Meaning |
| --- | --- |
| `text` | 原始输入文本。 |
| `tokens` | `tokenize(text)` 的结果。 |
| `ids` | `encode(text, add_special_tokens=True)` 的结果。 |
| `decoded` | `decode(ids)` 的结果。 |

推荐用 `rich.Table` 输出，方便肉眼对比。

运行命令：

```powershell
python -m llm_lab.day01_tokenizer_basic.run_demo
```

如果还没有安装为 editable package，需要先在项目根目录执行：

```powershell
pip install -e .
```

## What To Observe

跑 demo 时不要只看“有没有报错”，要看这些现象：

1. `CharTokenizer` 的 token 数通常比 `WordTokenizer` 多。
2. `CharTokenizer` 对 `tokenization` 会拆成很多字符。
3. `WordTokenizer` 遇到 corpus 里没有的完整词，容易得到 `<unk>`。
4. `WordTokenizer.decode()` 可能破坏原始空格和标点布局。
5. `<bos>` 和 `<eos>` 出现在 ids 中，但默认 decode 时应该被跳过。
6. 如果不跳过 special tokens，decode 结果应该能看到 `<bos>` / `<eos>` 这些结构标记。

## Tests To Add

Day01 至少需要这些测试：

```text
tests/
├── test_char_tokenizer.py
└── test_word_tokenizer.py
```

测试点：

1. `CharTokenizer.encode()` 和 `CharTokenizer.decode()` 对已知字符基本可逆。
2. `CharTokenizer` 遇到 unknown char 时会产生 `unk_token_id`。
3. `encode(add_special_tokens=True)` 会添加 `bos_token_id` 和 `eos_token_id`。
4. `decode(skip_special_tokens=True)` 会跳过 special tokens。
5. `WordTokenizer.tokenize()` 能切出英文、标点、括号、代码 identifier。
6. `WordTokenizer` 对不在 vocab 里的 token 会使用 `<unk>`。

运行：

```powershell
pytest
```

## Review Checklist

写完代码后，我会按这个清单 review：

1. API 是否和 README 的 facade 一致。
2. special token id 是否稳定。
3. unknown token 的处理是否明确。
4. `encode` 和 `decode` 是否职责清晰。
5. `WordTokenizer.decode()` 是否没有假装完美还原。
6. demo 是否能清楚暴露 char-level 和 word-level 的优缺点。
7. 测试是否覆盖 happy path 和 OOV path。

## Connection To Real Tokenizers

Day01 的 tokenizer 很朴素，但它已经对应真实 tokenizer 的核心概念：

| Day01 Concept | Real Tokenizer Concept |
| --- | --- |
| `token_to_id` | vocab |
| `id_to_token` | reverse vocab |
| `encode` | text to input ids |
| `decode` | output ids to text |
| `<bos>` / `<eos>` | sequence boundary tokens |
| `<unk>` | out-of-vocabulary fallback |
| `vocab_size` | embedding table row count |

后面 Day02 开始手写 BPE，才会进入现代 LLM tokenizer 的主线：

```text
统计相邻 pair
  -> 选择最高频 pair
  -> merge 成新 token
  -> 保存 merge rules
  -> 用 merge rules encode 新文本
```

Day04 再去看 Qwen、DeepSeek、GLM 的 tokenizer 文件，届时你会发现工业实现不是魔法，只是把这些基础概念做到了更大、更快、更严格。

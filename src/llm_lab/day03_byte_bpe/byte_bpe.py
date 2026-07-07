from __future__ import annotations

from dataclasses import dataclass


ByteToken = bytes
BytePair = tuple[ByteToken, ByteToken]
ByteTokenSequence = list[ByteToken]
ByteSplits = list[ByteTokenSequence]


@dataclass(frozen=True)
class ByteMergeStep:
    """
    记录一次 byte-level BPE merge。

    Day03 先保留这个结构，等实现 train() 后给 demo 展示用。
    """

    step: int
    best_pair: BytePair
    frequency: int
    new_token: ByteToken
    splits: ByteSplits


def text_to_byte_tokens(text: str) -> ByteTokenSequence:
    """
    把文本转成单 byte token 列表。

    Example:
        "A" -> [b"A"]
        "你" -> [b"\\xe4", b"\\xbd", b"\\xa0"]

    实现提示:
        1. raw = text.encode("utf-8")
        2. 遍历 raw 里的每个整数 byte
        3. 用 bytes([value]) 包成单 byte token
    """
    return [
        bytes([value])
        for value in text.encode("utf-8")
    ]


def byte_tokens_to_text(tokens: ByteTokenSequence) -> str:
    """
    把 byte token 列表拼回文本。

    注意:
        不要逐 token decode。
        单个 byte 可能不是合法 UTF-8 字符。

    正确流程:
        raw = b"".join(tokens)
        return raw.decode("utf-8")
    """
    return b"".join(tokens).decode("utf-8")


def format_byte_token(token: ByteToken) -> str:
    """
    把 byte token 格式化成人类容易读的形式。

    Example:
        b"A" -> "41"
        b"\\xe4" -> "e4"
        b"low" -> "6c 6f 77"

    这个函数只用于 demo 展示，不参与 tokenizer 核心逻辑。
    """
    return token.hex(" ")


def build_initial_byte_splits(corpus: list[str]) -> ByteSplits:
    """
    把 corpus 中每个文本切成 byte token sequence。

    Example:
        ["A", "你"] -> [[b"A"], [b"\\xe4", b"\\xbd", b"\\xa0"]]
    """
    return [
        text_to_byte_tokens(text)
        for text in corpus
    ]


def compute_byte_pair_freqs(splits: ByteSplits) -> dict[BytePair, int]:
    """
    统计相邻 byte token pair 的频率。

    这和 Day02 的 compute_pair_freqs 是同一件事，只是 token 类型从 str 变成 bytes。
    """
    pair_freqs: dict[BytePair, int] = {}

    for tokens in splits:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + 1

    return pair_freqs


def select_best_byte_pair(pair_freqs: dict[BytePair, int]) -> tuple[BytePair, int] | None:
    """
    选择最高频 byte pair。

    tie-break 和 Day02 保持一致:
        1. frequency 降序。
        2. pair 字典序升序。
    """
    if not pair_freqs:
        return None

    pair, frequency = min(
        pair_freqs.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return pair, frequency


def merge_byte_pair_in_sequence(tokens: ByteTokenSequence, pair: BytePair) -> ByteTokenSequence:
    """
    在单个 byte token sequence 中合并目标 pair。

    命中 pair 时，新 token 是 pair[0] + pair[1]。
    """
    merged_tokens: ByteTokenSequence = []
    i = 0

    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            merged_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1

    return merged_tokens


def merge_byte_pair(splits: ByteSplits, pair: BytePair) -> ByteSplits:
    """
    在整个 corpus splits 中合并目标 byte pair。
    """
    return [
        merge_byte_pair_in_sequence(tokens, pair)
        for tokens in splits
    ]


class ByteLevelBPETokenizer:
    """
    教学版 byte-level BPE tokenizer。

    Day03 的核心差异:
        Day02 token 是 str。
        Day03 token 是 bytes。

    如果初始化时把 256 个 single-byte token 都放进 vocab，
    那么任意 UTF-8 文本都可以落到 byte tokens 上，减少 <unk>。
    """

    def __init__(self) -> None:
        self.token_to_id: dict[ByteToken, int] = {}
        self.id_to_token: dict[int, ByteToken] = {}
        self.merges: list[BytePair] = []
        self.training_steps: list[ByteMergeStep] = []

        self._reset_vocab()

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def train(self, corpus: list[str], num_merges: int) -> None:
        """
        学习 byte-level BPE merge rules。

        等 Step 1 的 byte roundtrip 写完后再实现。
        """
        self._reset_vocab()
        self.merges = []
        self.training_steps = []

        splits = build_initial_byte_splits(corpus)

        for step in range(1, num_merges + 1):
            pair_freqs = compute_byte_pair_freqs(splits)
            best = select_best_byte_pair(pair_freqs)

            if best is None:
                break

            best_pair, frequency = best
            splits = merge_byte_pair(splits, best_pair)
            new_token = best_pair[0] + best_pair[1]

            self.merges.append(best_pair)
            self._add_token_to_vocab(new_token)
            self.training_steps.append(
                ByteMergeStep(
                    step=step,
                    best_pair=best_pair,
                    frequency=frequency,
                    new_token=new_token,
                    splits=splits,
                )
            )

    def tokenize(self, text: str) -> list[ByteToken]:
        """
        text -> byte tokens -> apply learned byte merges。

        注意:
            这里和 Day02 一样，不能重新学习 merge。
        """
        tokens = text_to_byte_tokens(text)

        for pair in self.merges:
            tokens = merge_byte_pair_in_sequence(tokens, pair)

        return tokens

    def encode(self, text: str) -> list[int]:
        """
        text -> byte BPE tokens -> ids。
        """
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids: list[int]) -> str:
        """
        ids -> byte tokens -> UTF-8 text。
        """
        tokens = self.convert_ids_to_tokens(ids)
        return byte_tokens_to_text(tokens)

    def convert_tokens_to_ids(self, tokens: list[ByteToken]) -> list[int]:
        """
        byte token 列表 -> id 列表。

        如果初始化了 256 byte vocab，并且 tokenize 只产生已有 merge token，
        理论上这里不应该频繁遇到 unknown。
        """
        return [
            self._require_token_id(token)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[ByteToken]:
        """
        id 列表 -> byte token 列表。
        """
        return [
            self._require_token(idx)
            for idx in ids
        ]

    def _reset_vocab(self) -> None:
        """
        初始化 256 个 single-byte token。

        id 直接等于 byte value，方便观察:
            b"\\x00" -> 0
            b"A"    -> 65
            b"\\xff" -> 255
        """
        self.token_to_id = {}
        self.id_to_token = {}

        for value in range(256):
            token = bytes([value])
            self.token_to_id[token] = value
            self.id_to_token[value] = token

    def _add_token_to_vocab(self, token: ByteToken) -> None:
        """
        把 BPE merge 产生的多 byte token 加入 vocab。
        """
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def _require_token_id(self, token: ByteToken) -> int:
        """
        byte-level tokenizer 理论上不需要 <unk>。

        如果这里查不到，说明内部 merge/vocab 状态不一致，应该显式报错。
        """
        if token not in self.token_to_id:
            raise KeyError(f"Byte token is not in vocabulary: {token!r}")

        return self.token_to_id[token]

    def _require_token(self, idx: int) -> ByteToken:
        """
        id 不存在时显式报错，避免 decode 悄悄拼出错误文本。
        """
        if idx not in self.id_to_token:
            raise KeyError(f"Token id is not in vocabulary: {idx}")

        return self.id_to_token[idx]

from __future__ import annotations

from dataclasses import dataclass


Pair = tuple[str, str]
TokenSequence = list[str]
Splits = list[TokenSequence]


@dataclass(frozen=True)
class MergeStep:
    """
    记录一次 BPE merge 的观察信息。

    这个结构主要给 run_demo.py 展示训练过程用，不是 BPE 算法必需品。
    """

    step: int
    best_pair: Pair
    frequency: int
    new_token: str
    splits: Splits


def build_initial_splits(corpus: list[str]) -> Splits:
    """
    把训练语料切成 BPE 的初始 token 序列。

    Day02 先做最朴素版本：每个词直接拆成字符。

    Example:
        ["low", "lower"] -> [["l", "o", "w"], ["l", "o", "w", "e", "r"]]
    """
    splits: Splits = []
    for word in corpus:
        token_sequence = list(word)
        splits.append(token_sequence)
    return splits


def compute_pair_freqs(splits: Splits) -> dict[Pair, int]:
    """
    统计所有相邻 token pair 的出现次数。

    输入:
        [
            ["l", "o", "w"],
            ["l", "o", "w", "e", "r"],
        ]

    输出应该包含:
        {
            ("l", "o"): 2,
            ("o", "w"): 2,
            ("w", "e"): 1,
            ("e", "r"): 1,
        }

    实现提示:
        1. 遍历每个 token sequence。
        2. 用 range(len(tokens) - 1) 扫相邻位置。
        3. pair = (tokens[i], tokens[i + 1])
        4. 对 pair 计数。
    """
    pair_freqs: dict[Pair, int] = {}
    for tokens in splits:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in pair_freqs:
                pair_freqs[pair] += 1
            else:
                pair_freqs[pair] = 1
    return pair_freqs


def select_best_pair(pair_freqs: dict[Pair, int]) -> tuple[Pair, int] | None:
    """
    选择本轮要合并的最高频 pair。

    tie-break 规则必须稳定:
        1. frequency 越高越优先。
        2. frequency 相同时，pair 按字典序越小越优先。

    建议实现:
        if not pair_freqs:
            return None

        pair, freq = min(
            pair_freqs.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return pair, freq
    """
    if not pair_freqs:
        return None
    pair, freq = min(
        pair_freqs.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return pair, freq


def merge_pair_in_sequence(tokens: TokenSequence, pair: Pair) -> TokenSequence:
    """
    在单个 token sequence 里合并目标 pair。

    Example:
        tokens = ["l", "o", "w"]
        pair = ("l", "o")
        result = ["lo", "w"]

    最关键的坑:
        tokens = ["a", "a", "a"]
        pair = ("a", "a")
        result 应该是 ["aa", "a"]，不是 ["aa", "aa"]。

    实现提示:
        1. 用 while 循环和 index 手动扫描。
        2. 如果 tokens[i], tokens[i + 1] 命中 pair:
             append(pair[0] + pair[1])
             i += 2
        3. 否则:
             append(tokens[i])
             i += 1
    """
    token_sequence: TokenSequence = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            token_sequence.append(pair[0] + pair[1])
            i += 2
        else:
            token_sequence.append(tokens[i])
            i += 1
    return token_sequence


def merge_pair(splits: Splits, pair: Pair) -> Splits:
    """
    在整个 corpus splits 中合并目标 pair。

    这个函数只是对每个 token sequence 调用 merge_pair_in_sequence。
    它不负责选择 best_pair，也不负责更新 vocab。
    """
    merged_splits: Splits = []
    for tokens in splits:
        merged_tokens = merge_pair_in_sequence(tokens, pair)
        merged_splits.append(merged_tokens)
    return merged_splits


class NaiveBPETokenizer:
    """
    教学版 BPE tokenizer。

    Day02 的目标是理解算法，不追求工业性能。

    训练阶段:
        corpus -> initial splits -> pair freqs -> merge rules -> vocab

    编码阶段:
        new text -> initial char split -> apply learned merges -> ids

    注意:
        tokenize(text) 只能应用已有 self.merges，不能重新学习 merge。
    """

    def __init__(self) -> None:
        self.unk_token = "<unk>"
        self.unk_token_id = 0

        self.token_to_id = {self.unk_token: self.unk_token_id}
        self.id_to_token = {self.unk_token_id: self.unk_token}
        self.merges = []
        self.training_steps = []

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def train(self, corpus: list[str], num_merges: int) -> None:
        """
        从训练语料中学习 BPE merge rules。

        你写完 Step 1 的函数后，再回来实现这里。

        高层流程:
            1. splits = build_initial_splits(corpus)
            2. 把初始字符加入 vocab
            3. 循环 num_merges 次:
                a. pair_freqs = compute_pair_freqs(splits)
                b. best = select_best_pair(pair_freqs)
                c. splits = merge_pair(splits, best_pair)
                d. self.merges.append(best_pair)
                e. 把 new_token 加入 vocab
                f. 记录 MergeStep，供 demo 展示
        """
        self.token_to_id: dict[str, int] = {self.unk_token: self.unk_token_id}
        self.id_to_token: dict[int, str] = {self.unk_token_id: self.unk_token}
        self.merges: list[Pair] = []
        self.training_steps: list[MergeStep] = []

        splits = build_initial_splits(corpus)
        for tokens in splits:
            for token in tokens:
                self._add_token_to_vocab(token)

        for step in range(1, num_merges + 1):
            pair_freqs = compute_pair_freqs(splits)
            best = select_best_pair(pair_freqs)
            if best is None:
                break
            best_pair, frequency = best
            splits = merge_pair(splits, best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self._add_token_to_vocab(new_token)
            self.training_steps.append(MergeStep(step, best_pair, frequency, new_token, splits))

    def tokenize(self, text: str) -> list[str]:
        """
        用已经学到的 self.merges 对新文本做 BPE tokenization。

        注意:
            这里不能重新统计 pair，也不能新增 merge。
            推理阶段只能应用训练阶段得到的规则。
        """
        tokens = list(text)

        for pair in self.merges:
            tokens = merge_pair_in_sequence(tokens, pair)

        return tokens

    def encode(self, text: str) -> list[int]:
        """
        text -> BPE tokens -> ids。
        """
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids: list[int]) -> str:
        """
        ids -> tokens -> text。

        Day02 的 naive decoder 可以先用 ''.join(tokens)。
        真实 tokenizer 的 decoder 后面 Day03/Day04 再讲。
        """
        tokens = self.convert_ids_to_tokens(ids)
        return "".join(tokens)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """
        token 列表 -> id 列表。

        不在 vocab 中的 token 映射到 <unk>。
        """
        return [
            self.token_to_id.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """
        id 列表 -> token 列表。

        不存在的 id 映射到 <unk>。
        """
        return [
            self.id_to_token.get(idx, self.unk_token)
            for idx in ids
        ]
    
    def _add_token_to_vocab(self, token: str) -> None:
        """
        把新 token 加入 vocab。

        如果 token 已经存在，就不做任何事。
        """
        if token not in self.token_to_id:
            new_id = len(self.token_to_id)
            self.token_to_id[token] = new_id
            self.id_to_token[new_id] = token

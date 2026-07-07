from __future__ import annotations

from llm_lab.day03_byte_bpe.byte_bpe import (
    ByteLevelBPETokenizer,
    BytePair,
    ByteToken,
    merge_byte_pair_in_sequence,
    text_to_byte_tokens,
)
from llm_lab.day04_hf_style_tokenizer.core import PreToken


class BPEModel:
    """
    HF-style model component。

    它只负责:
        pre-token bytes -> BPE tokens -> ids

    它不负责 normalizer、pre-tokenizer、post-processor 或 decoder。
    """

    def __init__(
        self,
        token_to_id: dict[ByteToken, int],
        merges: list[BytePair],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens: set[ByteToken] = set()

        for token in special_tokens or []:
            self.add_special_token(token)

    @classmethod
    def from_trained_byte_bpe(
        cls,
        tokenizer: ByteLevelBPETokenizer,
        special_tokens: list[str] | None = None,
    ) -> BPEModel:
        return cls(
            token_to_id=tokenizer.token_to_id,
            merges=tokenizer.merges,
            special_tokens=special_tokens,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def add_special_token(self, token: str) -> int:
        raw = token.encode("utf-8")
        self.special_tokens.add(raw)
        return self._add_token(raw)

    def tokenize_pre_token(self, pre_token: PreToken) -> list[ByteToken]:
        if pre_token.is_special:
            raw = pre_token.text.encode("utf-8")
            self._add_token(raw)
            self.special_tokens.add(raw)
            return [raw]

        return self.tokenize_bytes(pre_token.text.encode("utf-8"))

    def tokenize_bytes(self, raw: bytes) -> list[ByteToken]:
        tokens = [bytes([value]) for value in raw]

        while True:
            best_pair = self._find_best_ranked_pair(tokens)
            if best_pair is None:
                return tokens

            tokens = merge_byte_pair_in_sequence(tokens, best_pair)

    def ids_for_tokens(self, tokens: list[ByteToken]) -> list[int]:
        return [self._require_token_id(token) for token in tokens]

    def tokens_for_ids(self, ids: list[int]) -> list[ByteToken]:
        return [self._require_token(idx) for idx in ids]

    def is_special_token(self, token: ByteToken) -> bool:
        return token in self.special_tokens

    def _find_best_ranked_pair(self, tokens: list[ByteToken]) -> BytePair | None:
        best_pair: BytePair | None = None
        best_rank: int | None = None

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = self.merge_ranks.get(pair)

            if rank is None:
                continue

            if best_rank is None or rank < best_rank:
                best_pair = pair
                best_rank = rank

        return best_pair

    def _add_token(self, token: ByteToken) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]

        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id

    def _require_token_id(self, token: ByteToken) -> int:
        if token not in self.token_to_id:
            # BPE output should only contain single bytes, learned merges, or added special tokens.
            raise KeyError(f"Token is not in vocabulary: {token!r}")
        return self.token_to_id[token]

    def _require_token(self, idx: int) -> ByteToken:
        if idx not in self.id_to_token:
            raise KeyError(f"Token id is not in vocabulary: {idx}")
        return self.id_to_token[idx]

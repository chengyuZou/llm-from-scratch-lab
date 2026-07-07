import pytest

from llm_lab.day03_byte_bpe.byte_bpe import (
    ByteLevelBPETokenizer,
    build_initial_byte_splits,
    byte_tokens_to_text,
    compute_byte_pair_freqs,
    format_byte_token,
    merge_byte_pair_in_sequence,
    text_to_byte_tokens,
)


@pytest.mark.parametrize("text", ["hello", "你好", "😀", "Agent🤖", "𠮷"])
def test_text_to_byte_tokens_roundtrip(text: str) -> None:
    tokens = text_to_byte_tokens(text)

    assert byte_tokens_to_text(tokens) == text


def test_byte_lengths_for_ascii_chinese_and_emoji() -> None:
    assert text_to_byte_tokens("A") == [b"A"]
    assert len(text_to_byte_tokens("你")) == 3
    assert len(text_to_byte_tokens("😀")) == 4


def test_format_byte_token() -> None:
    assert format_byte_token(b"A") == "41"
    assert format_byte_token(b"\xe4") == "e4"
    assert format_byte_token(b"low") == "6c 6f 77"


def test_build_initial_byte_splits() -> None:
    assert build_initial_byte_splits(["A", "你"]) == [
        [b"A"],
        [b"\xe4", b"\xbd", b"\xa0"],
    ]


def test_compute_byte_pair_freqs() -> None:
    splits = build_initial_byte_splits(["low", "lower"])

    pair_freqs = compute_byte_pair_freqs(splits)

    assert pair_freqs[(b"l", b"o")] == 2
    assert pair_freqs[(b"o", b"w")] == 2
    assert pair_freqs[(b"w", b"e")] == 1


def test_merge_byte_pair_in_sequence_does_not_overlap_matches() -> None:
    assert merge_byte_pair_in_sequence([b"a", b"a", b"a"], (b"a", b"a")) == [
        b"aa",
        b"a",
    ]


def test_byte_level_tokenizer_initial_vocab_has_256_single_byte_tokens() -> None:
    tokenizer = ByteLevelBPETokenizer()

    assert tokenizer.vocab_size == 256
    assert tokenizer.token_to_id[b"\x00"] == 0
    assert tokenizer.token_to_id[b"A"] == 65
    assert tokenizer.token_to_id[b"\xff"] == 255


def test_byte_level_tokenizer_roundtrips_unseen_unicode_without_unk() -> None:
    tokenizer = ByteLevelBPETokenizer()
    text = "没见过的文本😀𠮷"

    ids = tokenizer.encode(text)

    assert all(0 <= idx <= 255 for idx in ids)
    assert tokenizer.decode(ids) == text


def test_byte_level_bpe_train_tokenize_encode_decode() -> None:
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(["low", "lower", "lowest"], num_merges=2)

    assert tokenizer.merges == [(b"l", b"o"), (b"lo", b"w")]
    assert tokenizer.tokenize("slow") == [b"s", b"low"]
    assert tokenizer.decode(tokenizer.encode("slow")) == "slow"


def test_invalid_token_or_id_raises() -> None:
    tokenizer = ByteLevelBPETokenizer()

    with pytest.raises(KeyError):
        tokenizer.convert_tokens_to_ids([b"not-in-vocab-yet"])

    with pytest.raises(KeyError):
        tokenizer.convert_ids_to_tokens([999])

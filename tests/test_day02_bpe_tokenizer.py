from llm_lab.day02_bpe_tokenizer.naive_bpe import (
    NaiveBPETokenizer,
    build_initial_splits,
    compute_pair_freqs,
    merge_pair,
    merge_pair_in_sequence,
    select_best_pair,
)


def test_build_initial_splits() -> None:
    assert build_initial_splits(["low", "lower"]) == [
        ["l", "o", "w"],
        ["l", "o", "w", "e", "r"],
    ]


def test_compute_pair_freqs() -> None:
    splits = build_initial_splits(["low", "lower", "lowest"])

    pair_freqs = compute_pair_freqs(splits)

    assert pair_freqs == {
        ("l", "o"): 3,
        ("o", "w"): 3,
        ("w", "e"): 2,
        ("e", "r"): 1,
        ("e", "s"): 1,
        ("s", "t"): 1,
    }


def test_select_best_pair_uses_deterministic_tie_break() -> None:
    pair_freqs = {
        ("o", "w"): 3,
        ("l", "o"): 3,
        ("w", "e"): 2,
    }

    assert select_best_pair(pair_freqs) == (("l", "o"), 3)
    assert select_best_pair({}) is None


def test_merge_pair_in_sequence_does_not_overlap_matches() -> None:
    assert merge_pair_in_sequence(["a", "a", "a"], ("a", "a")) == ["aa", "a"]


def test_merge_pair_merges_every_sequence() -> None:
    splits = build_initial_splits(["low", "lower", "lowest"])

    assert merge_pair(splits, ("l", "o")) == [
        ["lo", "w"],
        ["lo", "w", "e", "r"],
        ["lo", "w", "e", "s", "t"],
    ]


def test_train_records_stable_merges_and_steps() -> None:
    tokenizer = NaiveBPETokenizer()

    tokenizer.train(["low", "lower", "lowest"], num_merges=5)

    assert tokenizer.merges == [
        ("l", "o"),
        ("lo", "w"),
        ("low", "e"),
        ("lowe", "r"),
        ("lowe", "s"),
    ]
    assert [step.new_token for step in tokenizer.training_steps] == [
        "lo",
        "low",
        "lowe",
        "lower",
        "lowes",
    ]


def test_train_resets_previous_state() -> None:
    tokenizer = NaiveBPETokenizer()

    tokenizer.train(["low", "lower", "lowest"], num_merges=5)
    tokenizer.train(["ab"], num_merges=2)

    assert tokenizer.merges == [("a", "b")]
    assert tokenizer.token_to_id == {
        "<unk>": 0,
        "a": 1,
        "b": 2,
        "ab": 3,
    }


def test_tokenize_applies_learned_merges_without_mutating_rules() -> None:
    tokenizer = NaiveBPETokenizer()
    tokenizer.train(["low", "lower", "lowest"], num_merges=2)
    merges_before = list(tokenizer.merges)

    tokens = tokenizer.tokenize("slowest")

    assert tokens == ["s", "low", "e", "s", "t"]
    assert tokenizer.merges == merges_before


def test_encode_decode_and_unknown_fallback() -> None:
    tokenizer = NaiveBPETokenizer()
    tokenizer.train(["low", "lower", "lowest"], num_merges=2)

    ids = tokenizer.encode("slow")

    assert ids == [
        tokenizer.token_to_id["s"],
        tokenizer.token_to_id["low"],
    ]
    assert tokenizer.decode(ids) == "slow"
    assert tokenizer.encode("x") == [tokenizer.unk_token_id]
    assert tokenizer.decode([999]) == "<unk>"

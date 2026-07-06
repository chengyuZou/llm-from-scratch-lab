from llm_lab.day01_tokenizer_basic.char_tokenizer import CharTokenizer
from llm_lab.day01_tokenizer_basic.word_tokenizer import WordTokenizer


def test_char_tokenizer_roundtrip_known_text() -> None:
    tokenizer = CharTokenizer("我正在学习 tokenizer")

    ids = tokenizer.encode("我正在学习")

    assert tokenizer.decode(ids) == "我正在学习"


def test_char_tokenizer_unknown_char_keeps_unk() -> None:
    tokenizer = CharTokenizer("abc")

    ids = tokenizer.encode("abz")

    assert ids == [
        tokenizer.token_to_id["a"],
        tokenizer.token_to_id["b"],
        tokenizer.unk_token_id,
    ]
    assert tokenizer.decode(ids) == "ab<unk>"


def test_encode_adds_bos_and_eos() -> None:
    tokenizer = CharTokenizer("abc")

    ids = tokenizer.encode("abc", add_special_tokens=True)

    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id


def test_decode_skip_special_tokens_does_not_skip_unk() -> None:
    tokenizer = CharTokenizer("abc")
    ids = [
        tokenizer.bos_token_id,
        tokenizer.token_to_id["a"],
        tokenizer.unk_token_id,
        tokenizer.eos_token_id,
    ]

    decoded = tokenizer.decode(ids, skip_special_tokens=True)

    assert decoded == "a<unk>"


def test_word_tokenizer_builds_vocab_from_tokens_not_chars() -> None:
    tokenizer = WordTokenizer(
        "LLM tokenizers convert text into token ids. "
        "def get_user_profile(user_id): return db.query(user_id)"
    )

    tokens = tokenizer.tokenize("def get_user_profile(user_id)")
    ids = tokenizer.encode("def get_user_profile(user_id)")

    assert tokens == ["def", "get_user_profile", "(", "user_id", ")"]
    assert ids == [
        tokenizer.token_to_id["def"],
        tokenizer.token_to_id["get_user_profile"],
        tokenizer.token_to_id["("],
        tokenizer.token_to_id["user_id"],
        tokenizer.token_to_id[")"],
    ]
    assert tokenizer.unk_token_id not in ids

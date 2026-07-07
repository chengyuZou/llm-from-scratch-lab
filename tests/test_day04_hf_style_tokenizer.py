from llm_lab.day04_hf_style_tokenizer.chat_template import (
    IM_END,
    IM_START,
    QwenLikeChatTemplate,
)
from llm_lab.day04_hf_style_tokenizer.core import ChatMessage
from llm_lab.day04_hf_style_tokenizer.factory import build_qwen_like_tokenizer
from llm_lab.day04_hf_style_tokenizer.models import BPEModel
from llm_lab.day04_hf_style_tokenizer.post_processors import TemplatePostProcessor
from llm_lab.day04_hf_style_tokenizer.pre_tokenizers import (
    ByteLevelPreTokenizer,
    SpecialAwarePreTokenizer,
)
from llm_lab.day04_hf_style_tokenizer.tokenizer import HFStyleTokenizer
from llm_lab.day04_hf_style_tokenizer.normalizers import IdentityNormalizer
from llm_lab.day04_hf_style_tokenizer.decoders import ByteLevelDecoder


def test_byte_level_pre_tokenizer_preserves_whitespace() -> None:
    pre_tokenizer = ByteLevelPreTokenizer()

    pieces = pre_tokenizer.pre_tokenize("hello  world")

    assert [piece.text for piece in pieces] == ["hello", "  ", "world"]
    assert [piece.offset for piece in pieces] == [(0, 5), (5, 7), (7, 12)]


def test_special_aware_pre_tokenizer_keeps_special_token_atomic() -> None:
    pre_tokenizer = SpecialAwarePreTokenizer(
        base=ByteLevelPreTokenizer(),
        special_tokens=[IM_START, IM_END],
    )

    pieces = pre_tokenizer.pre_tokenize(f"{IM_START}user\nhi{IM_END}")

    assert pieces[0].text == IM_START
    assert pieces[0].is_special is True
    assert pieces[-1].text == IM_END
    assert pieces[-1].is_special is True


def test_qwen_like_chat_template_renders_generation_prompt() -> None:
    template = QwenLikeChatTemplate()

    prompt = template.render(
        [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="你好"),
        ],
        add_generation_prompt=True,
    )

    assert prompt == (
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_qwen_like_pipeline_roundtrips_plain_text() -> None:
    tokenizer = build_qwen_like_tokenizer()
    text = "hello 你好 Agent🤖"

    encoding = tokenizer.encode(text)

    assert tokenizer.decode(encoding) == text
    assert tokenizer.decode(encoding, skip_special_tokens=True) == text
    assert len(encoding.ids) == len(encoding.tokens)
    assert len(encoding.offsets) == len(encoding.tokens)


def test_qwen_like_pipeline_marks_chat_special_tokens() -> None:
    tokenizer = build_qwen_like_tokenizer()
    template = QwenLikeChatTemplate()
    prompt = template.render([ChatMessage(role="user", content="解释 tokenizer。")])

    encoding = tokenizer.encode(prompt)

    special_tokens = [
        token.decode("utf-8")
        for token, is_special in zip(encoding.tokens, encoding.special_tokens_mask)
        if is_special
    ]

    assert IM_START in special_tokens
    assert IM_END in special_tokens
    assert tokenizer.decode(encoding, skip_special_tokens=False) == prompt
    assert IM_START not in tokenizer.decode(encoding, skip_special_tokens=True)


def test_template_post_processor_adds_special_tokens() -> None:
    model = BPEModel(
        token_to_id={bytes([value]): value for value in range(256)},
        merges=[],
        special_tokens=["<s>", "</s>"],
    )
    tokenizer = HFStyleTokenizer(
        normalizer=IdentityNormalizer(),
        pre_tokenizer=ByteLevelPreTokenizer(),
        model=model,
        decoder=ByteLevelDecoder(),
        post_processor=TemplatePostProcessor(prefix=["<s>"], suffix=["</s>"]),
    )

    encoding = tokenizer.encode("hi", add_special_tokens=True)

    assert encoding.special_tokens_mask[0] == 1
    assert encoding.special_tokens_mask[-1] == 1
    assert tokenizer.decode(encoding, skip_special_tokens=True) == "hi"
    assert tokenizer.decode(encoding, skip_special_tokens=False) == "<s>hi</s>"


def test_bpe_model_does_not_mutate_merges_during_encode() -> None:
    tokenizer = build_qwen_like_tokenizer()
    merges_before = list(tokenizer.model.merges)

    tokenizer.encode("completely unseen text 🤖🤖")

    assert tokenizer.model.merges == merges_before

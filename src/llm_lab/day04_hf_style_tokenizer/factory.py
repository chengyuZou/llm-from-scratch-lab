from __future__ import annotations

from llm_lab.day03_byte_bpe.byte_bpe import ByteLevelBPETokenizer
from llm_lab.day04_hf_style_tokenizer.chat_template import QWEN_LIKE_SPECIAL_TOKENS
from llm_lab.day04_hf_style_tokenizer.decoders import ByteLevelDecoder
from llm_lab.day04_hf_style_tokenizer.models import BPEModel
from llm_lab.day04_hf_style_tokenizer.normalizers import IdentityNormalizer
from llm_lab.day04_hf_style_tokenizer.pre_tokenizers import (
    ByteLevelPreTokenizer,
    SpecialAwarePreTokenizer,
)
from llm_lab.day04_hf_style_tokenizer.tokenizer import HFStyleTokenizer


DEFAULT_TRAINING_CORPUS = [
    "hello world",
    "hello tokenizer",
    "low lower lowest",
    "def get_user_profile(user_id): return db.query(user_id)",
    "你好，世界",
    "Agent🤖 can call tools.",
]


def build_qwen_like_tokenizer() -> HFStyleTokenizer:
    """
    构建一个 Qwen-like 的教学 tokenizer。

    参考真实模型的边界:
        - byte-level BPE model
        - added special tokens: <|im_start|>, <|im_end|>, <|endoftext|>
        - chat template 在单独模块里处理

    注意:
        这里不加载真实 Qwen vocab/merges，只复刻 pipeline 形状。
    """
    trainer = ByteLevelBPETokenizer()
    trainer.train(DEFAULT_TRAINING_CORPUS, num_merges=80)

    model = BPEModel.from_trained_byte_bpe(
        trainer,
        special_tokens=QWEN_LIKE_SPECIAL_TOKENS,
    )

    pre_tokenizer = SpecialAwarePreTokenizer(
        base=ByteLevelPreTokenizer(),
        special_tokens=QWEN_LIKE_SPECIAL_TOKENS,
    )

    return HFStyleTokenizer(
        normalizer=IdentityNormalizer(),
        pre_tokenizer=pre_tokenizer,
        model=model,
        decoder=ByteLevelDecoder(),
    )

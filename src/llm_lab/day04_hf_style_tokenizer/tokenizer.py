from __future__ import annotations

from llm_lab.day04_hf_style_tokenizer.core import Encoding
from llm_lab.day04_hf_style_tokenizer.decoders import ByteLevelDecoder
from llm_lab.day04_hf_style_tokenizer.models import BPEModel
from llm_lab.day04_hf_style_tokenizer.normalizers import Normalizer
from llm_lab.day04_hf_style_tokenizer.post_processors import NoPostProcessor, PostProcessor
from llm_lab.day04_hf_style_tokenizer.pre_tokenizers import PreTokenizer


class HFStyleTokenizer:
    """
    手搓版 Hugging Face tokenizer pipeline。

    normalize -> pre-tokenize -> model -> post-process -> decode
    """

    def __init__(
        self,
        normalizer: Normalizer,
        pre_tokenizer: PreTokenizer,
        model: BPEModel,
        decoder: ByteLevelDecoder,
        post_processor: PostProcessor | None = None,
    ) -> None:
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.model = model
        self.decoder = decoder
        self.post_processor = post_processor or NoPostProcessor()

    def encode(self, text: str, add_special_tokens: bool = True) -> Encoding:
        normalized = self.normalizer.normalize(text)
        pre_tokens = self.pre_tokenizer.pre_tokenize(normalized)

        byte_tokens: list[bytes] = []
        offsets: list[tuple[int, int] | None] = []
        special_mask: list[int] = []

        for pre_token in pre_tokens:
            tokens = self.model.tokenize_pre_token(pre_token)
            byte_tokens.extend(tokens)
            offsets.extend([pre_token.offset] * len(tokens))
            special_mask.extend([1 if pre_token.is_special else 0] * len(tokens))

        encoding = Encoding(
            ids=self.model.ids_for_tokens(byte_tokens),
            tokens=byte_tokens,
            offsets=offsets,
            special_tokens_mask=special_mask,
            normalized_text=normalized,
            pre_tokens=pre_tokens,
        )

        if not add_special_tokens:
            return encoding

        return self.post_processor.process(encoding, self.model)

    def decode(self, encoding: Encoding, skip_special_tokens: bool = False) -> str:
        return self.decoder.decode(encoding, skip_special_tokens=skip_special_tokens)

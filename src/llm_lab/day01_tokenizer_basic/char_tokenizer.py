from __future__ import annotations


class CharTokenizer:
    """
    最朴素的字符级 tokenizer。

    它的逻辑：
    1. 从语料里收集所有字符
    2. 给每个字符分配一个 id
    3. encode: str -> list[int]
    4. decode: list[int] -> str

    真实 LLM 通常不会直接用这种 tokenizer。
    但它非常适合用来理解 tokenizer 的最小闭环。
    """

    def __init__(self, corpus: str):
        # special tokens 放在 vocab 最前面，让它们的 id 稳定，方便测试和观察。
        # <pad>: padding，占位补齐；<unk>: unknown，未知内容；
        # <bos>/<eos>: begin/end of sequence，序列边界。
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

        # CharTokenizer 的 vocab 粒度就是字符，所以这里从 corpus 收集字符。
        # sorted(...) 让 id 分配稳定；否则 set 的遍历顺序会影响输出。
        chars = sorted(set(corpus))

        # vocab 是“token 字符串 -> token id”的唯一来源。
        vocab = special_tokens + chars

        # encode 查 token_to_id；decode 查 id_to_token。
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
    
    def tokenize(self, text: str) -> list[str]:
        """
        将文本拆分为字符列表。
        """
        return list(text)
    
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        将文本编码为 id 列表。如果 add_special_tokens 为 True,则在开头和结尾添加特殊符号。
        """
        tokens = self.tokenize(text)

        ids = [
            self.token_to_id.get(token, self.unk_token_id)
            for token in tokens
        ]

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        return ids


    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """
        将 id 列表解码为文本。如果 skip_special_tokens 为 True,则跳过特殊符号。
        """
        tokens: list[str] = []

        # skip_special_tokens 只跳过结构 token，不跳过 <unk>。
        # <unk> 表示原文有未知内容，删掉它会掩盖 OOV 问题。
        special_set = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        for idx in ids:
            if skip_special_tokens and idx in special_set:
                continue

            tokens.append(self.id_to_token.get(idx, self.unk_token))
        
        return "".join(tokens)
    
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """
        将字符列表转换为 id 列表。
        """
        return [
            self.token_to_id.get(token, self.unk_token_id)
            for token in tokens
        ]
    
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """
        将 id 列表转换为字符列表。
        """
        return [
            self.id_to_token.get(idx, self.unk_token)
            for idx in ids
        ]

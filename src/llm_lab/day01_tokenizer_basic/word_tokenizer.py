from __future__ import annotations

import re


class WordTokenizer:
    """
    非严格 word-level tokenizer。

    它使用正则把文本切成：
    - 英文/数字/下划线连续片段
    - 单个非空白符号

    例如：
    "Hello, world!" -> ["Hello", ",", "world", "!"]
    "我喜欢LLM。" -> ["我喜欢LLM", "。"]

    这仍然不是现代 LLM 主流 tokenizer。
    但它能帮你理解 word-level 的问题：
    词表容易爆炸，遇到新词容易 <unk>。
    """

    # \w+ 会匹配一段连续的“字母/数字/下划线/Unicode word 字符”。
    # 所以 get_user_profile 会被当成一个 token，连续中文也可能被当成一整段。
    # [^\w\s] 用来把标点、括号等非空白符号单独切出来。
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __init__(self, corpus: str):
        # special tokens 放在 vocab 最前面，让它们的 id 稳定，方便测试和观察。
        # <pad>: padding，占位补齐；<unk>: unknown，未知内容；
        # <bos>/<eos>: begin/end of sequence，序列边界。
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

        # WordTokenizer 的 vocab 粒度必须和 tokenize() 的输出一致。
        # 这里不能写 set(corpus)，否则就会退化成“字符词表 + word token 查询”，
        # 导致 def、LLM、get_user_profile 等 token 全部 OOV。
        tokens = self.tokenize(corpus)
        unique_tokens = sorted(set(tokens))

        # vocab 是“token 字符串 -> token id”的唯一来源。
        vocab = special_tokens + unique_tokens

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
        将文本拆分为 token 列表。
        """
        return self.TOKEN_PATTERN.findall(text)
    
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
        
        # 这里故意用最朴素的拼接，暴露 word-level decode 的问题：
        # 原始空格、标点间距、换行和缩进已经在 tokenize 时丢失了。
        return "".join(tokens)
    
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """
        将 token 列表转换为 id 列表。
        """
        return [
            self.token_to_id.get(token, self.unk_token_id)
            for token in tokens
        ]
    
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """
        将 id 列表转换为 token 列表。
        """
        return [
            self.id_to_token.get(idx, self.unk_token)
            for idx in ids
        ]

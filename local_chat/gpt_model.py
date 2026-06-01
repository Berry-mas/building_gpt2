from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    label: str
    chapter: str
    task: str
    checkpoint: str
    config: dict[str, Any]
    description: str


CH05_CONFIG = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

CH06_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

CH07_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

MODEL_SPECS = {
    "ch05": ModelSpec(
        label="ch05 - next-token 생성",
        chapter="ch05",
        task="generate",
        checkpoint="model.pth",
        config=CH05_CONFIG,
        description="the-verdict.txt로 사전학습 원리를 실습한 GPT 모델",
    ),
    "ch06": ModelSpec(
        label="ch06 - SMS 스팸 분류",
        chapter="ch06",
        task="classify",
        checkpoint="review_classifier.pth",
        config=CH06_CONFIG,
        description="GPT-2 small을 SMS 스팸 분류에 파인튜닝한 모델",
    ),
    "ch07": ModelSpec(
        label="ch07 - instruction 응답",
        chapter="ch07",
        task="instruction",
        checkpoint="gpt2-medium355M-sft.pth",
        config=CH07_CONFIG,
        description="GPT-2 medium을 instruction 데이터로 SFT한 모델",
    ),
}


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coefficient = torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
        return 0.5 * x * (1 + torch.tanh(coefficient * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        keys = self.W_key(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = self.W_query(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = self.W_value(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context = (self.dropout(attn_weights) @ values).transpose(1, 2)
        context = context.contiguous().view(batch_size, num_tokens, self.d_out)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        return x + self.drop_shortcut(self.ff(self.norm2(x)))


class GPTModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        return self.out_head(self.final_norm(self.trf_blocks(x)))


def load_model(spec: ModelSpec, checkpoint_dir: Path, device: torch.device) -> GPTModel:
    checkpoint = checkpoint_dir / spec.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"체크포인트가 없습니다: {checkpoint}")

    model = GPTModel(spec.config)
    if spec.task == "classify":
        model.out_head = nn.Linear(spec.config["emb_dim"], 2)

    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate(
    model: GPTModel,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits = torch.where(
                logits < top_logits[:, -1],
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break
        token_ids = torch.cat((token_ids, idx_next), dim=1)

    return token_ids


def format_instruction(instruction: str, input_text: str = "") -> str:
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
    )
    if input_text:
        prompt += f"\n\n### Input:\n{input_text}"
    return prompt + "\n\n### Response:\n"


def classify_sms(
    text: str,
    model: GPTModel,
    tokenizer: Any,
    device: torch.device,
    max_length: int,
    pad_token_id: int = 50256,
) -> str:
    supported_length = model.pos_emb.weight.shape[0]
    max_length = min(max_length, supported_length)
    input_ids = tokenizer.encode(text)[:max_length]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    inputs = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(inputs)[:, -1, :]
    predicted = torch.argmax(logits, dim=-1).item()
    return "스팸" if predicted == 1 else "스팸 아님"

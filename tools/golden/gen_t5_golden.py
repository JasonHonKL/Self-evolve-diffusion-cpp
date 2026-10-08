# Golden generator for the UMT5-XXL encoder (issue #9).
#
# Loads the ORIGINAL Wan2.2 .pth into a verbatim copy of Ovi's T5Encoder class
# (Ovi/ovi/modules/t5.py, trimmed to encoder-only), runs 3 prompts from
# tools/golden/tokenizer_golden.npz (ids produced by the C++ spm tokenizer,
# issue #7) padded to L=16 with pad_id=0 / mask 0, saves final hidden states.
#
# Precision choice (documented per issue): model runs in bf16 — same as the
# .pth weights and same as Wan inference. Model is built on the meta device and
# loaded with load_state_dict(assign=True), so peak RAM is the 11.4 GB state
# dict only (fp32 would peak ~23 GB).
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = "ckpts/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
TG = "tools/golden/tokenizer_golden.npz"
DST = "tools/golden/t5_golden.npz"
CFG = dict(vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240,
           num_heads=64, num_layers=24, num_buckets=32, shared_pos=False)
L = 16
PAD = 0
PROMPTS = [0, 6, 8]  # en / zh / ar, 12/15/10 tokens


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def forward(self, x, mask=None, pos_bias=None):
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(x).view(b, -1, n, c)
        v = self.v(x).view(b, -1, n, c)
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            mask = mask.view(b, 1, 1, -1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias  # no scaling
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)
        return self.o(x.reshape(b, -1, n * c))


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x) * self.gate(x))


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, max_dist=128):
        super().__init__()
        self.num_buckets, self.num_heads = num_buckets, num_heads
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        rel_pos = torch.arange(lk).unsqueeze(0) - torch.arange(lq).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        return self.embedding(rel_pos).permute(2, 0, 1).unsqueeze(0).contiguous()

    def _relative_position_bucket(self, rel_pos):
        num_buckets = self.num_buckets // 2  # bidirectional
        rel_buckets = (rel_pos > 0).long() * num_buckets
        rel_pos = torch.abs(rel_pos)
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5SelfAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets):
        super().__init__()
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads)

    def forward(self, x, mask=None):
        e = self.pos_embedding(x.size(1), x.size(1))
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        x = x + self.ffn(self.norm2(x))
        return x


class T5Encoder(nn.Module):
    def __init__(self, vocab, dim, dim_attn, dim_ffn, num_heads, num_layers,
                 num_buckets, shared_pos):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets)
            for _ in range(num_layers)])
        self.norm = T5LayerNorm(dim)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


with torch.device("meta"):
    model = T5Encoder(**CFG).eval()
sd = torch.load(SRC, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(sd, assign=True)
assert not missing and not unexpected, (missing, unexpected)
del sd
print("loaded", sum(p.numel() for p in model.parameters()) / 1e9, "B params")

g = np.load(TG, allow_pickle=True)
ids_flat, off = g["ids_flat"], g["offsets"]
prompts = json.loads(g["prompts_json"].tobytes())

batch, bmask = [], []
for i in PROMPTS:
    ids = ids_flat[off[i]:off[i + 1]].tolist()
    m = [1] * len(ids) + [0] * (L - len(ids))
    ids = ids + [PAD] * (L - len(ids))
    assert len(ids) == L and 0 not in ids[:m.index(0)]
    batch.append(ids)
    bmask.append(m)
    print(f"prompt {i}: {len(ids)} tok  {prompts[i][:50]!r}")

ids = torch.tensor(batch)
mask = torch.tensor(bmask)
with torch.no_grad():
    hidden = model(ids, mask).float()  # [B, L, 4096] bf16 -> f32

np.savez(DST, ids=ids.numpy(), mask=mask.numpy(), hidden=hidden.numpy())
print("wrote", DST, hidden.shape)

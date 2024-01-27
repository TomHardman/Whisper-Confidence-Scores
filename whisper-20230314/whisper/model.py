import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import loralib

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, dropout: float = 0.0, attn: str = None, lora: int = 0):
        super().__init__()
        self.n_head = n_head
        if lora:
            self.query = loralib.Linear(n_state, n_state, r=lora)
        else:
            self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        if lora:
            self.value = loralib.Linear(n_state, n_state, r=lora)
        else:
            self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.dropout = dropout
        self.attn = attn

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        ilm: str = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        if kv_cache is not None and self.attn not in kv_cache:
            kv_cache[self.attn] = []

        if ilm == 'avgh':
            wv = v.mean(dim=1).unsqueeze(1).repeat(1, x.shape[1], 1).to(q.dtype)
            qk = None
        else:
            wv, qk, w = self.qkv_attention(q, k, v, mask)

        if kv_cache and self.attn:
            # w_p = w[:, :, -1, :].squeeze()
            # kv_cache[self.attn].append(w_p)
            kv_cache[self.attn] = w
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        w = nn.functional.dropout(w, p=self.dropout, training=self.training)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach(), w


class LocalMultiHeadAttention(MultiHeadAttention):
    def __init__(self, n_state: int, n_head: int, dropout: float = 0.0, attn: str = None, lora: int = 0, attention_window: int = -1):
        super().__init__(n_state, n_head, dropout, attn, lora)
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attention_window_size = attention_window // 2

    @staticmethod
    def _skew(x, direction):
        """Convert diagonals into columns (or columns into diagonals depending on `direction`"""
        x_padded = F.pad(x, direction)  # padding value is not important because it will be overwritten
        x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        return x_padded

    @staticmethod
    def _skew2(x):
        """shift every row 1 step to right converting columns into diagonals"""
        # X = B x C x M x L
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1))  # B x C x M x (L+M+1). Padding value is not important because it'll be overwritten
        x = x.view(B, C, -1)  # B x C x ML+MM+M
        x = x[:, :, :-M]  # B x C x ML+MM
        x = x.view(B, C, M, M + L)  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x

    @staticmethod
    def _chunk(x, w):
        """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    def _mask_invalid_locations(self, input_tensor, w) -> torch.Tensor:
        affected_seqlen = w
        beginning_mask_2d = input_tensor.new_ones(w, w + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        seqlen = input_tensor.size(1)
        beginning_input = input_tensor[:, :affected_seqlen, :, : w + 1]
        beginning_mask = beginning_mask[:, :seqlen].expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seqlen:, :, -(w + 1) :]
        ending_mask = ending_mask[:, -seqlen:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8

    def _sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int):
        """Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
        with an overlap of size w"""
        batch_size, seqlen, num_heads, head_dim = q.size()
        assert seqlen % (w * 2) == 0, f"Sequence length should be multiple of {w * 2}. Given {seqlen}"
        assert q.size() == k.size()

        chunks_count = seqlen // w - 1

        # group batch_size and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
        q = q.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)
        k = k.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)

        chunk_q = self._chunk(q, w)
        chunk_k = self._chunk(k, w)

        # matrix multipication
        # bcxd: batch_size * num_heads x chunks x 2w x head_dim
        # bcyd: batch_size * num_heads x chunks x 2w x head_dim
        # bcxy: batch_size * num_heads x chunks x 2w x 2w
        chunk_attn = torch.einsum("bcxd,bcyd->bcxy", (chunk_q, chunk_k))  # multiply

        # convert diagonals into columns
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1))

        # allocate space for the overall attention matrix where the chunks are compined. The last dimension
        # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
        # w previous words). The following column is attention score from each word to itself, then
        # followed by w columns for the upper triangle.

        diagonal_attn = diagonal_chunk_attn.new_empty((batch_size * num_heads, chunks_count + 1, w, w * 2 + 1))

        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
        # - copying the lower triangle
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1) : -1, w + 1 :]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]

        # separate batch_size and num_heads dimensions again
        diagonal_attn = diagonal_attn.view(batch_size, num_heads, seqlen, 2 * w + 1).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attn, w)
        return diagonal_attn

    def _sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        """Same as _sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
        format from _sliding_chunks_matmul_qk"""
        batch_size, seqlen, num_heads, head_dim = v.size()
        assert seqlen % (w * 2) == 0
        assert prob.size()[:3] == v.size()[:3]
        assert prob.size(3) == 2 * w + 1
        chunks_count = seqlen // w - 1
        # group batch_size and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
        chunk_prob = prob.transpose(1, 2).reshape(batch_size * num_heads, seqlen // w, w, 2 * w + 1)

        # group batch_size and num_heads dimensions into one
        v = v.transpose(1, 2).reshape(batch_size * num_heads, seqlen, head_dim)

        # pad seqlen with w at the beginning of the sequence and another w at the end
        padded_v = F.pad(v, (0, 0, w, w), value=-1)

        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunk_v_size = (batch_size * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

        skewed_prob = self._skew2(chunk_prob)
        if chunk_v.dtype == torch.float16:
            skewed_prob = skewed_prob.half()
        context = torch.einsum("bcwd,bcdh->bcwh", (skewed_prob, chunk_v))
        return context.view(batch_size, num_heads, seqlen, head_dim).transpose(1, 2)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1) * scale
        k = k.view(*k.shape[:2], self.n_head, -1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1)

        # qk = q @ k
        # if mask is not None:
        #     qk = qk + mask[:n_ctx, :n_ctx]
        # qk = qk.float()

        # w = F.softmax(qk, dim=-1).to(q.dtype)
        # w = nn.functional.dropout(w, p=self.dropout, training=self.training)
        # return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach(), w

        # attn_weights = (batch_size, seqlen, num_heads, window*2+1)
        attn_weights = self._sliding_chunks_matmul_qk(q, k, self.one_sided_attention_window_size)
        self._mask_invalid_locations(attn_weights, self.one_sided_attention_window_size)

        if mask is not None:
            attn_weights = attn_weights + mask[:n_ctx, :n_ctx]
        attn_weights = attn_weights.float()

        assert list(attn_weights.size()) == [
            n_batch,
            n_ctx,
            self.n_head,
            self.one_sided_attention_window_size * 2 + 1,
        ]

        w = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        w = w.type_as(attn_weights)

        attn_probs = F.dropout(w, p=self.dropout, training=self.training)
        attn = self._sliding_chunks_matmul_pv(attn_probs, v, self.one_sided_attention_window_size)

        return attn.flatten(start_dim=2), attn_weights.detach(), w


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, dropout: float = 0.0, attn: str = None, ilm: str = None, lora: int = 0, attention_window: int = -1):
        super().__init__()

        if attention_window < 0:
            self.attn = MultiHeadAttention(n_state, n_head, dropout=dropout, lora=lora)
        else:
            self.attn = LocalMultiHeadAttention(n_state, n_head, dropout=dropout, lora=lora, attention_window=attention_window)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head, dropout=dropout, attn=attn, lora=lora) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
        self.dropout = dropout
        self.ilm = ilm

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + nn.functional.dropout(self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0], p=self.dropout, training=self.training)
        if self.cross_attn and self.ilm != 'nocross':
            x = x + nn.functional.dropout(self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache, ilm=self.ilm)[0], p=self.dropout, training=self.training)
        x = x + nn.functional.dropout(self.mlp(self.mlp_ln(x)), p=self.dropout, training=self.training) # a_t = mlp_ln(x) - extract here
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout: float = 0.0, lora: int = 0, attention_window: int = -1
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, dropout=dropout, lora=lora, attention_window=attention_window) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        self.dropout = dropout

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout: float = 0.0, ilm: str = None, lora: int = 0
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True, dropout=dropout, attn=str(_), ilm=ilm, lora=lora)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.dropout = dropout

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x) # ---> d_t
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class SoftTextDecoder(TextDecoder):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, n_prompts: int, dropout: float = 0.0):
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer, dropout=dropout)
        self.n_prompts = n_prompts
        self.learned_embedding = nn.parameter.Parameter(torch.FloatTensor(n_prompts, n_state).normal_())
        mask = torch.empty(n_ctx+n_prompts, n_ctx+n_prompts).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        offset = next(iter(kv_cache.values())).shape[1]-self.n_prompts if kv_cache else 0
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        if not kv_cache:
            learned_embedding = self.learned_embedding.repeat(x.size(0), 1, 1)
            x = torch.cat([learned_embedding, x], 1)
        x = x.to(xa.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits


class SoftTextDecoderV2(SoftTextDecoder):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, n_prompts: int, dropout: float = 0.0):
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer, n_prompts)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = self.token_embedding(x)
        if not kv_cache:
            learned_embedding = self.learned_embedding.repeat(x.size(0), 1, 1)
            x = torch.cat([learned_embedding, x], 1)
        x = x + self.positional_embedding[offset : offset + x.shape[1]]
        x = x.to(xa.dtype)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, dropout: float = 0.0, ilm: str = None, lora: int = 0, attention_window: int = -1):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dropout=dropout,
            lora=lora,
            attention_window=attention_window,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            dropout=dropout,
            ilm=ilm,
            lora=lora,
        )
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function


class SoftWhisper(Whisper):
    def __init__(self, dims: ModelDimensions, n_decoder_prompts, dropout: float = 0.0):
        super().__init__(dims, dropout=dropout)
        self.decoder = SoftTextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            n_decoder_prompts,
            dropout=dropout
        )


class SoftWhisperV2(Whisper):
    def __init__(self, dims: ModelDimensions, n_decoder_prompts, dropout: float = 0.0):
        super().__init__(dims, dropout=dropout)
        self.decoder = SoftTextDecoderV2(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            n_decoder_prompts,
            dropout=dropout
        )

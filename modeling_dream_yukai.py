import os
from types import SimpleNamespace
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from configuration_dream import DreamConfig

logger = logging.get_logger(__name__)


class DreamOutput(NamedTuple):
    logits: torch.FloatTensor


class DreamRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_indexed(q, k, pos_cos, pos_sin):
    pos_cos = pos_cos[None, None, :, :]
    pos_sin = pos_sin[None, None, :, :]

    q_, k_ = q.float(), k.float()
    q_out = ((q_ * pos_cos) + (rotate_half(q_) * pos_sin)).to(q.dtype)
    k_out = ((k_ * pos_cos) + (rotate_half(k_) * pos_sin)).to(k.dtype)
    return q_out, k_out


class DreamMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class DreamAttentionWeights(nn.Module):
    def __init__(self, config: DreamConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)


class DreamDecoderLayerYukai(nn.Module):

    def __init__(self, config: DreamConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = DreamAttentionWeights(config)
        self.mlp = DreamMLP(config)
        self.input_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def concat_and_replace(self, matrix_origin, matrix_current, idx_current, shape_target):

        if matrix_origin.shape[-2] < shape_target[-2]:
            length_patch = shape_target[-2] - matrix_origin.shape[-2]

            assert matrix_current.shape[-2] >= length_patch,\
                f'current shape should be >= patch shape, {matrix_current.shape[-2]} >= {length_patch}'
            matrix_patch = matrix_current[:, :, -length_patch:, :]

            matrix_origin = torch.cat([matrix_origin, matrix_patch], dim=-2)

        assert matrix_origin.shape[-2] == shape_target[-2],\
            f'origin shape should equal to target shape after patch, {matrix_origin.shape[-2]} == {shape_target[-2]}'

        matrix_origin[:, :, idx_current, :] = matrix_current
        return matrix_origin

    def get_attn_score_avg(self, Q, K):
        num_q_heads, num_kv_heads = Q.size(1), K.size(1)
        if num_q_heads != num_kv_heads:
            K = K.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

        scale = Q.size(-1) ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights_avg = attn_weights.mean(dim=1)
        return attn_weights_avg

    def _scaled_dot_product_attention(self, q, k, v):
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
            v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

    def attention(
        self,
        q_current: torch.Tensor,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        idx_current: Optional[torch.Tensor] = None,
        shape_target: Optional[Tuple[int, int, int]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        B, T, C = q_current.size()
        n_heads = self.self_attn.num_heads
        n_kv_heads = self.self_attn.num_key_value_heads
        head_dim = self.self_attn.head_dim

        q_current = q_current.view(B, T, n_heads, head_dim).transpose(1, 2)
        k_current = k_current.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        v_current = v_current.view(B, T, n_kv_heads, head_dim).transpose(1, 2)

        pos_cos_full, pos_sin_full = position_embeddings
        pos_cos = pos_cos_full.index_select(0, idx_current)
        pos_sin = pos_sin_full.index_select(0, idx_current)
        q_current_rotated, k_current = apply_rotary_pos_emb_indexed(q_current, k_current, pos_cos, pos_sin)

        k_final, v_final = self.plugin_cache_past_kv.load()
        k_final_rotated = k_final

        self.plugin_cache_attn.save()

        hidden = self._scaled_dot_product_attention(q_current_rotated, k_final_rotated, v_final)
        hidden = hidden.transpose(1, 2).contiguous().view(B, T, C)

        self.plugin_cache_past_kv.save()

        return self.self_attn.o_proj(hidden)

    def forward(
        self,
        x_current: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        idx_current: Optional[torch.Tensor] = None,
        shape_target: Optional[Tuple[int, int, int]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        x_normed_current = self.input_layernorm(x_current)

        v = self.self_attn.v_proj(x_normed_current)
        idx_current, x_current, x_normed_current, v = \
            self.plugin_cache_vo.select_hidden(idx_current, x_current, x_normed_current, v)

        q_current = self.self_attn.q_proj(x_normed_current)
        k = self.self_attn.k_proj(x_normed_current)

        attn_current = self.attention(
            q_current, k, v,
            idx_current=idx_current,
            shape_target=shape_target,
            position_embeddings=position_embeddings,
        )

        x_final = x_current + attn_current

        og_x_final = x_final
        x_final = self.post_attention_layernorm(x_final)
        x_final = self.mlp(x_final)
        x_final = og_x_final + x_final

        x_final = self.plugin_cache_vo.load_merge_and_update_hidden(x_final)
        return x_final


class DreamPreTrainedModelYukai(PreTrainedModel):
    config_class = DreamConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["DreamDecoderLayerYukai"]
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class DreamBaseModelYukai(DreamPreTrainedModelYukai):

    def __init__(self, config: DreamConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DreamDecoderLayerYukai(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DreamRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.rope_scaling is not None:
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            rope_type = "default"
        inv_freq, self.rope_attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, None)
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self._rope_cos = None
        self._rope_sin = None

        self.transformer = SimpleNamespace(blocks=self.layers)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _get_rope_table(self, length, device):
        if (self._rope_cos is None) or (self._rope_cos.shape[0] < length) or (self._rope_cos.device != device):
            t = torch.arange(length, dtype=torch.float32, device=device)
            inv_freq = self.rope_inv_freq.to(device=device, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._rope_cos = emb.cos() * self.rope_attention_scaling
            self._rope_sin = emb.sin() * self.rope_attention_scaling

        return self._rope_cos, self._rope_sin

    def fill_plugin(self, klass_plugin):
        for block in self.layers:
            instance_plugin = klass_plugin()
            setattr(block, instance_plugin.get_plugin_name(), instance_plugin)
        return self

    def collect_plugins_with_index(self, klass_plugin):
        name_plugin = klass_plugin().get_plugin_name()
        plugins = []
        for idx_block, block in enumerate(self.layers):
            plugins.append((idx_block, getattr(block, name_plugin)))
        return plugins

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_bias: Optional[torch.Tensor] = None,
        idx_current: Optional[torch.Tensor] = None,
        shape_target: Optional[Tuple[int, int, int]] = None,
        skip_final_norm: bool = False,
    ) -> torch.Tensor:

        batch_size, seq_len = input_ids.shape

        x = self.embed_tokens(input_ids)

        if idx_current is None:
            idx_current = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        if shape_target is None:
            shape_target = (batch_size, seq_len, -1)

        position_embeddings = self._get_rope_table(shape_target[1], input_ids.device)

        for layer in self.layers:
            x = layer(
                x,
                attention_bias=attention_bias,
                idx_current=idx_current,
                shape_target=shape_target,
                position_embeddings=position_embeddings,
            )

        if skip_final_norm:
            return x

        return self.norm(x)


class DreamModelLM(DreamPreTrainedModelYukai):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DreamConfig):
        super().__init__(config)
        self.model = DreamBaseModelYukai(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        idx_current: Optional[torch.Tensor] = None,
        shape_target: Optional[Tuple[int, int, int]] = None,
        skip_logits: bool = False,
        **kwargs,
    ) -> DreamOutput:

        hidden = self.model(
            input_ids,
            attention_bias=attention_bias,
            idx_current=idx_current,
            shape_target=shape_target,
            skip_final_norm=skip_logits,
        )

        if skip_logits:
            return DreamOutput(logits=None)

        logits = self.lm_head(hidden)
        return DreamOutput(logits=logits)

    def fill_plugin(self, klass_plugin):
        self.model.fill_plugin(klass_plugin)
        return self

    def collect_plugins_with_index(self, klass_plugin):
        return self.model.collect_plugins_with_index(klass_plugin)

from transformers import AutoConfig, PretrainedConfig

from enum import Enum
from os import PathLike
from typing import Union
from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)


__all__ = [
    "ActivationType",
    "ActivationCheckpointingStrategy",
    "BlockType",
    "LayerNormType",
    "InitFnType",
    "ModelConfig",
]

PathOrStr = Union[str, PathLike]


class StrEnum(str, Enum):

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class LayerNormType(StrEnum):
    default = "default"

    low_precision = "low_precision"

    rms = "rms"

    gemma_rms = "gemma_rms"

    amd_compatible = "amd_compatible"


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"
    parallel = "parallel"

    llama = "llama"


class InitFnType(StrEnum):
    mitchell = "mitchell"

    normal = "normal"

    kaiming_normal = "kaiming_normal"

    fan_in = "fan_in"

    full_megatron = "full_megatron"


@dataclass
class ModelConfig():


    d_model: int = 768

    n_heads: int = 12

    n_kv_heads: Optional[int] = None

    n_layers: int = 12

    mlp_ratio: int = 4

    mlp_hidden_size: Optional[int] = None

    activation_type: ActivationType = ActivationType.swiglu

    block_type: BlockType = BlockType.sequential

    block_group_size: int = 1

    alibi: bool = False

    alibi_bias_max: float = 8.0

    rope: bool = False

    rope_full_precision: bool = True

    flash_attention: bool = False

    attention_dropout: float = 0.1

    multi_query_attention: Optional[bool] = None

    attention_layer_norm: bool = False

    residual_dropout: float = 0.1

    embedding_dropout: float = 0.1

    input_emb_norm: bool = False

    layer_norm_type: LayerNormType = LayerNormType.default

    layer_norm_with_affine: bool = True

    rms_norm_eps: float = 1e-05

    attention_layer_norm_with_affine: bool = True

    max_sequence_length: int = 1024

    train_max_sequence_length: int = 1024

    rope_theta: float = 10000.0

    include_qkv_bias: Optional[bool] = False

    include_bias: bool = False

    bias_for_layer_norm: Optional[bool] = None

    scale_logits: bool = False

    vocab_size: int = 50257

    embedding_size: Optional[int] = 50304

    weight_tying: bool = True

    eos_token_id: int = 50256

    pad_token_id: int = 50256

    mask_token_id: Optional[int] = 50256

    init_device: Optional[str] = None

    init_fn: InitFnType = InitFnType.normal

    init_std: float = 0.02

    init_cutoff_factor: Optional[float] = None

    precision: Optional[str] = None

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            if self.multi_query_attention is True:
                return 1
            else:
                return self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            if self.multi_query_attention:
                n_kv_heads_should_be = 1
            else:
                n_kv_heads_should_be = self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise Exception(
                    "You can't set `multi_query_attention` and `n_kv_heads` at the same time."
                )

class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"

    one_in_two = "one_in_two"

    one_in_three = "one_in_three"

    one_in_four = "one_in_four"
    
    two_in_three = "two_in_three"

    three_in_four = "three_in_four"

    four_in_five = "four_in_five"

    nine_in_ten = "nine_in_ten"

    fine_grained = "fine_grained"


class LLaDAConfig(PretrainedConfig):
    model_type = "llada"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, use_cache: bool = False, **kwargs):
        model_config = ModelConfig()
        all_kwargs = model_config.__dict__
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        all_kwargs.update(
            {
                "architectures": all_kwargs.get("architectures", ["LLaDAModelLM"])
            }
        )
        super().__init__(**all_kwargs)

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers

    @property
    def hidden_size(self):
        return self.d_model


AutoConfig.register("llada", LLaDAConfig)

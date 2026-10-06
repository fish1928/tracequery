from dataclasses import dataclass
from typing import Optional

from tools_llada import ConfKSorter, ConfCollectorInterface, BlockDiffusionQuotaHelper
from plugins_llada import InspectorPlugin


@dataclass
class DiffusionConfig:
    id_model: str
    len_prompt: int
    len_target: int
    num_blocks: int
    num_unmask_per_step: int
    id_mask: int
    size_batch: int
    device: str
    klass_sorter: ConfKSorter
    klass_collector: ConfCollectorInterface
    klass_save_kv_previous: InspectorPlugin
    klass_cache_past_kv: InspectorPlugin
    klass_cache_attn: InspectorPlugin
    klass_cache_vo: InspectorPlugin
    
    size_block: Optional[int] = None
    step_per_block: Optional[int] = None

    def __post_init__(self):
        self.size_block= int(self.len_target / self.num_blocks)
        self.step_per_block=int(self.size_block / self.num_unmask_per_step)


@dataclass
class DiffusionConfig_Eval:
    id_model: str
    len_target: int
    num_blocks: int
    num_unmask_per_step: int
    id_mask: int
    size_batch: int
    device: str
    klass_sorter: ConfKSorter
    klass_collector: ConfCollectorInterface

    use_chat_template: Optional[bool] = None
    use_official_gsm8k_prompt: Optional[bool] = None
    stop_at_code_fence: Optional[bool] = None
    truncate_at_eos: Optional[bool] = None

    d2c_k: Optional[int] = None
    d2c_sigma: Optional[float] = None
    d2c_rollout_p: Optional[float] = None
    d2c_conf_mode: Optional[str] = None
    d2c_inflate_w: Optional[int] = None

    dllmc_v_rate: Optional[float] = None


    step_refresh_remainder: Optional[int] = None
    step_refresh_remainder_prompt: Optional[int] = None
    step_refresh_remainder_surfix: Optional[int] = None
    h: Optional[int] = None
    select_only_in_h: Optional[bool] = None
    path_router: Optional[str] = None
    requery_age: Optional[int] = None
    path_report: Optional[str] = None

    klass_save_kv_previous: Optional[InspectorPlugin] = None
    klass_cache_past_kv: Optional[InspectorPlugin] = None
    klass_cache_attn: Optional[InspectorPlugin] = None
    klass_cache_vo: Optional[InspectorPlugin] = None
    
    size_block: Optional[int] = None
    step_per_block: Optional[int] = None

    def __post_init__(self):
        self.size_block= int(self.len_target / self.num_blocks)
        self.step_per_block=int(self.size_block / self.num_unmask_per_step)


@dataclass
class KVAggregateConfig:
    stamp: str
    type_aggregate: str
    len_prompt: str
    len_target: str
    num_blocks: int
    folder_output: Optional[str] = None
    type_fn: Optional[str] = None

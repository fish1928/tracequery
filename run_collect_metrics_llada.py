from run_collect_metrics_llada_base import OracleCollector
from modeling_llada_yukai_06 import LLaDAModelLM
from collect_metrics_common import build_parser, main_collect


if __name__ == '__main__':
    main_collect(
        OracleCollector,
        LLaDAModelLM,
        build_parser(id_model='GSAI-ML/LLaDA-8B-Base', id_mask=126336),
    )

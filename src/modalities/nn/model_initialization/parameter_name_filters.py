from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class WeightInitTypes(Enum):
    PLAIN = "plain"
    SCALED = "scaled"
    SCALED_EMBED = "scaled_embed"


class SupportWeightInitModels(Enum):
    GPT2 = "gpt2"
    COCA = "coca"
    DUALPATH = "dualpath"


class RegexFilter(BaseModel):
    weights: list[str]
    biases: Optional[list[str]] = Field(default_factory=list)


NAMED_PARAMETER_INIT_GROUPS = {
    SupportWeightInitModels.GPT2: {
        # as per https://arxiv.org/abs/2312.16903
        # plain: All linear and embedding layers and embedding layers
        #        are initialized by sampling from a normal distribution.
        #        NOTE: layer norms are ALWAYS initialized beforehand at instantiation time!
        WeightInitTypes.PLAIN: RegexFilter(
            weights=[
                # attention projection weights
                r"transformer\.h\.\d+\.attn\.(q_attn|k_attn|v_attn|c_proj)\.weight",
                # hidden feed forward in attention block
                r"transformer\.h\.\w+\.mlp\.(W|V|W_2)\.weight",  # SwiGLU
                r"transformer\.h\.\w+\.mlp\.(c_fc|c_proj)\.weight",  # gelu
                # embedding weights
                r"transformer\.wte\.weight",
                r"transformer\.wpe\.weight",
                # lm_head weights (although usually tied with transformer.wte.weight)
                r"transformer\.lm_head\.weight",
            ],
            biases=[
                # NOTE: some bias terms might not be present due to user configuration
                r"transformer\.h\.\d+\.attn\.(q_attn|k_attn|v_attn|c_proj)\.bias",
                r"transformer\.h\.\w+\.mlp\.(W|V|W_2)\.bias",  # SwiGLU
                r"transformer\.h\.\w+\.mlp\.(c_fc|c_proj)\.bias",  # gelu
                r"transformer\.lm_head\.bias",  # typically not present
            ],
        ),
        # scaled: In the attention block, we scale the final projection in the FFN (W_2)
        #         and the projection before the FFN (W_o), as defined in
        #         https://arxiv.org/abs/2312.16903
        WeightInitTypes.SCALED: RegexFilter(
            weights=[
                r"transformer\.h\.\d+\.attn\.c_proj\.weight",
                r"transformer\.h\.\w+\.mlp\.W_2.weight",  # SwiGLU
                r"transformer\.h\.\w+\.mlp\.c_proj\.weight",  # gelu
            ]
        ),
        WeightInitTypes.SCALED_EMBED: RegexFilter(
            weights=[
                # embedding weights
                r"transformer\.wte\.weight",
                r"transformer\.wpe\.weight",
                r"transformer\.lm_head\.weight",
            ]
        ),
    },
    # DualPathLLM nests its transformer blocks one level deeper than GPT2:
    # `transformer.h.<N>.block.attn...` / `transformer.h.<N>.wide_block.mlp...`
    # rather than `transformer.h.<N>.attn...`. The GPT2 filters below use
    # re.fullmatch, so they silently match NOTHING on this model and the
    # configured init is skipped entirely -- weights then keep nn.Linear's
    # default Kaiming init, which happens to land near std=0.02 and so looks
    # fine, but loses the 1/sqrt(2*num_layers) residual downscaling.
    #
    # Deliberately NOT matched here (these are initialized on purpose elsewhere,
    # in the modules' own reset_parameters(), which runs before this init):
    #   - `dual_gate.gate_proj.{weight,bias}` and the cross projections: zero-init
    #   - `loop_scales` / `wide_scale`: constant-init to loop_scale_init
    #   - `router.linear.*`: left at nn.Linear default
    #   - all norms and the RoPE `inv_freq` buffer
    SupportWeightInitModels.DUALPATH: {
        WeightInitTypes.PLAIN: RegexFilter(
            weights=[
                # attention projection weights (deep/looped and wide sub-blocks)
                r"transformer\.h\.\d+\.(block|wide_block)\.attn\.(q_attn|k_attn|v_attn|c_proj)\.weight",
                # SwiGLU feed forward
                r"transformer\.h\.\d+\.(block|wide_block)\.mlp\.(W|V|W_2)\.weight",
                # embedding weights
                r"transformer\.wte\.weight",
                r"transformer\.wpe\.weight",
                # lm_head weights (usually tied with transformer.wte.weight)
                r"transformer\.lm_head\.weight",
            ],
            biases=[
                r"transformer\.h\.\d+\.(block|wide_block)\.attn\.(q_attn|k_attn|v_attn|c_proj)\.bias",
                r"transformer\.h\.\d+\.(block|wide_block)\.mlp\.(W|V|W_2)\.bias",
                r"transformer\.lm_head\.bias",
            ],
        ),
        # scaled: the two residual-stream output projections, as for GPT2.
        WeightInitTypes.SCALED: RegexFilter(
            weights=[
                r"transformer\.h\.\d+\.(block|wide_block)\.attn\.c_proj\.weight",
                r"transformer\.h\.\d+\.(block|wide_block)\.mlp\.W_2\.weight",
            ]
        ),
        WeightInitTypes.SCALED_EMBED: RegexFilter(
            weights=[
                r"transformer\.wte\.weight",
                r"transformer\.wpe\.weight",
                r"transformer\.lm_head\.weight",
            ]
        ),
    },
    SupportWeightInitModels.COCA: {
        # we reject all bias and weight parameters belonging to norms
        WeightInitTypes.PLAIN: RegexFilter(
            weights=[r"^(?!.*norm)(?!.*ln_).*\.weight$"], biases=[r"^(?!.*norm)(?!.*ln_).*\.bias$"]
        ),
        WeightInitTypes.SCALED: RegexFilter(weights=[], biases=[]),
        WeightInitTypes.SCALED_EMBED: RegexFilter(weights=[], biases=[]),
    },
}

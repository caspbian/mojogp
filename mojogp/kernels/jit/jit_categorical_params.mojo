"""Categorical raw-parameter transforms for JIT mixed training.

Training optimizes unconstrained categorical parameters, while correlation matrix
builders consume constrained parameters.
"""

from memory import UnsafePointer

from kernels.categorical_state import CategoricalCorrelationState
from kernels.constants import (
    CAT_KERNEL_EHH,
    CAT_KERNEL_FE,
    CAT_KERNEL_HH,
    PI,
)
from kernels.utils import softplus, softplus_derivative, sigmoid, sigmoid_derivative


fn _cat_param_local_info(
    read cat_state: CategoricalCorrelationState,
    param_index: Int,
) -> Tuple[Int, Int, Int]:
    var local_index = param_index
    for var_idx in range(cat_state.num_cat_vars):
        var levels = cat_state.levels[var_idx]
        var n_params = cat_state.get_num_params_for_var(var_idx)
        if local_index < n_params:
            return (cat_state.kernel_types[var_idx], levels, local_index)
        local_index -= n_params
    return (CAT_KERNEL_EHH, 0, 0)


fn constrain_cat_raw_param(
    raw: Float32,
    kernel_type: Int,
    levels: Int,
    local_param_index: Int,
) -> Float32:
    if kernel_type == CAT_KERNEL_EHH or kernel_type == CAT_KERNEL_HH:
        return sigmoid(raw) * Float32(PI)
    elif kernel_type == CAT_KERNEL_FE:
        var num_angles = levels * (levels - 1) // 2
        if local_param_index < num_angles:
            return sigmoid(raw) * Float32(PI)
        return softplus(raw)
    return softplus(raw)


fn cat_chain_derivative(
    raw: Float32,
    kernel_type: Int,
    levels: Int,
    local_param_index: Int,
) -> Float32:
    if kernel_type == CAT_KERNEL_EHH or kernel_type == CAT_KERNEL_HH:
        return sigmoid_derivative(raw) * Float32(PI)
    elif kernel_type == CAT_KERNEL_FE:
        var num_angles = levels * (levels - 1) // 2
        if local_param_index < num_angles:
            return sigmoid_derivative(raw) * Float32(PI)
        return softplus_derivative(raw)
    return softplus_derivative(raw)


fn constrained_cat_param(
    read cat_state: CategoricalCorrelationState,
    read raw_cat: List[Float32],
    param_index: Int,
) -> Float32:
    var info = _cat_param_local_info(cat_state, param_index)
    return constrain_cat_raw_param(raw_cat[param_index], info[0], info[1], info[2])


fn cat_chain_derivative_for_param(
    read cat_state: CategoricalCorrelationState,
    read raw_cat: List[Float32],
    param_index: Int,
) -> Float32:
    var info = _cat_param_local_info(cat_state, param_index)
    return cat_chain_derivative(raw_cat[param_index], info[0], info[1], info[2])


fn write_constrained_cat_params(
    read cat_state: CategoricalCorrelationState,
    read raw_cat: List[Float32],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
):
    for k in range(len(raw_cat)):
        out_ptr[k] = constrained_cat_param(cat_state, raw_cat, k)

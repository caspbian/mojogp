from python import PythonObject


alias DEFAULT_PREDICT_LANCZOS_RANK = 100
alias DEFAULT_ARD_PREDICT_LANCZOS_RANK = 200
alias DEFAULT_MIXED_MATRIX_FREE_PREDICT_LANCZOS_RANK = 200
alias DEFAULT_MIXED_MATERIALIZED_PREDICT_LANCZOS_RANK = 200
alias DEFAULT_MIXED_MATERIALIZED_ARD_PREDICT_LANCZOS_RANK = 300


fn _info_bool(info: PythonObject, key: String) raises -> Bool:
    if Bool(info.__contains__(key)):
        return Bool(info[key].__bool__())
    return False


fn _info_materialization_mode(info: PythonObject) raises -> Int:
    if Bool(info.__contains__("materialization_mode")):
        return Int(info["materialization_mode"].__int__())
    return 0


fn _resolve_exact_predict_lanczos_rank(
    info: PythonObject,
    requested_rank: Int,
    is_mixed: Bool,
    use_materialized: Bool,
) raises -> Int:
    if requested_rank > 0:
        return requested_rank

    var min_rank = DEFAULT_PREDICT_LANCZOS_RANK
    if is_mixed:
        if use_materialized:
            min_rank = DEFAULT_MIXED_MATERIALIZED_PREDICT_LANCZOS_RANK
        else:
            min_rank = DEFAULT_MIXED_MATRIX_FREE_PREDICT_LANCZOS_RANK

    if _info_bool(info, "is_ard"):
        if min_rank < DEFAULT_ARD_PREDICT_LANCZOS_RANK:
            min_rank = DEFAULT_ARD_PREDICT_LANCZOS_RANK
        if is_mixed and use_materialized and min_rank < DEFAULT_MIXED_MATERIALIZED_ARD_PREDICT_LANCZOS_RANK:
            min_rank = DEFAULT_MIXED_MATERIALIZED_ARD_PREDICT_LANCZOS_RANK

    return min_rank


fn _resolve_multi_predict_lanczos_rank(
    requested_rank: Int,
    is_mixed: Bool,
    use_materialized: Bool,
    is_ard: Bool,
) -> Int:
    if requested_rank > 0:
        return requested_rank
    var min_rank = DEFAULT_PREDICT_LANCZOS_RANK
    if is_mixed and not use_materialized:
        min_rank = DEFAULT_MIXED_MATRIX_FREE_PREDICT_LANCZOS_RANK
    if is_ard:
        if min_rank < DEFAULT_ARD_PREDICT_LANCZOS_RANK:
            min_rank = DEFAULT_ARD_PREDICT_LANCZOS_RANK
        if is_mixed and use_materialized and min_rank < DEFAULT_MIXED_MATERIALIZED_ARD_PREDICT_LANCZOS_RANK:
            min_rank = DEFAULT_MIXED_MATERIALIZED_ARD_PREDICT_LANCZOS_RANK
    return min_rank

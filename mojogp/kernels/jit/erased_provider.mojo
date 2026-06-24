"""ErasedJITProvider: fn-ptr based trait adapter for fast JIT.

This struct conforms to JITGradientProvider by forwarding all 17 trait methods
through function pointers loaded from a kernel .so.

Pattern copied from scripts/compile_experiments/exp8_engine.mojo (proven to work).
Key: fn ptrs are stored as actual function types, not Int. Conversion happens
once at construction time.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from memory import UnsafePointer, alloc

from kernels.jit.jit_training import JITGradientProvider


# =============================================================================
# Function pointer type aliases (must match kernel .so exports exactly)
# =============================================================================

alias FwdMatvecFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                       UnsafePointer[Float32, MutAnyOrigin], Int) raises -> None

alias GradMatvecFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                        UnsafePointer[Float32, MutAnyOrigin], Int, Int, Bool) raises -> None

alias FusedGradFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                       UnsafePointer[Float32, MutAnyOrigin], Int) raises -> None

alias FusedLsOsFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                       UnsafePointer[Float32, MutAnyOrigin],
                       UnsafePointer[Float32, MutAnyOrigin], Int) raises -> None

alias Fused3PFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                     UnsafePointer[Float32, MutAnyOrigin],
                     UnsafePointer[Float32, MutAnyOrigin],
                     UnsafePointer[Float32, MutAnyOrigin], Int) raises -> None

alias ExtractDiagFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin]) raises -> None

alias UpdateParamsFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin]) raises -> None

alias UpdateNoiseFn = fn(Int, Float32) -> None

alias GetFloatFn = fn(Int) -> Float32

alias GetIntFn = fn(Int) -> Int

alias GetPtrFn = fn(Int) -> UnsafePointer[Float32, MutAnyOrigin]

alias CrossMatvecFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                         UnsafePointer[Float32, MutAnyOrigin],
                         UnsafePointer[Float32, MutAnyOrigin],
                         Int, Int) raises -> None

alias FillCrossCovFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                          UnsafePointer[Float32, MutAnyOrigin],
                          Int) raises -> None

alias ExtractDiagTestFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                             UnsafePointer[Float32, MutAnyOrigin],
                             Int) raises -> None

alias KroneckerGradFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                           UnsafePointer[Float32, MutAnyOrigin],
                           UnsafePointer[Float32, MutAnyOrigin],
                           Int, Int, Float32) raises -> None

alias KroneckerFwdFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                           UnsafePointer[Float32, MutAnyOrigin],
                           UnsafePointer[Float32, MutAnyOrigin],
                           UnsafePointer[Float32, MutAnyOrigin],
                           Int, Int, Float32) raises -> None

# Mixed kernel fn ptr aliases
alias MixedFwdMatvecFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                             UnsafePointer[Float32, MutAnyOrigin],
                             UnsafePointer[Int32, MutAnyOrigin],
                             UnsafePointer[Float32, MutAnyOrigin],
                             UnsafePointer[Int32, MutAnyOrigin],
                             UnsafePointer[Int32, MutAnyOrigin],
                             Int, Int, Float32) raises -> None

alias MixedFusedGradMatvecFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                                   UnsafePointer[Float32, MutAnyOrigin],
                                   UnsafePointer[Int32, MutAnyOrigin],
                                   UnsafePointer[Float32, MutAnyOrigin],
                                   UnsafePointer[Int32, MutAnyOrigin],
                                   UnsafePointer[Int32, MutAnyOrigin],
                                   Int, Int) raises -> None

alias MixedCrossMatvecFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                               UnsafePointer[Float32, MutAnyOrigin],
                               UnsafePointer[Float32, MutAnyOrigin],
                               UnsafePointer[Int32, MutAnyOrigin],
                               UnsafePointer[Int32, MutAnyOrigin],
                               UnsafePointer[Float32, MutAnyOrigin],
                               UnsafePointer[Int32, MutAnyOrigin],
                               UnsafePointer[Int32, MutAnyOrigin],
                               Int, Int, Int) raises -> None

alias MixedExtractDiagFn = fn(Int, UnsafePointer[Float32, MutAnyOrigin],
                                UnsafePointer[Int32, MutAnyOrigin],
                                UnsafePointer[Float32, MutAnyOrigin],
                                UnsafePointer[Int32, MutAnyOrigin],
                                UnsafePointer[Int32, MutAnyOrigin],
                                Int) raises -> None

alias MixedMaterializeFn = fn(Int,
                               UnsafePointer[Int32, MutAnyOrigin],
                               UnsafePointer[Float32, MutAnyOrigin],
                               UnsafePointer[Int32, MutAnyOrigin],
                               UnsafePointer[Int32, MutAnyOrigin],
                               Int) raises -> None


# =============================================================================
# Int → fn ptr conversion helpers (used ONCE at construction, not per-call)
# =============================================================================

@always_inline
fn _cvt_fwd(addr: Int) -> FwdMatvecFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[FwdMatvecFn]()[]; s.free(); return r

@always_inline
fn _cvt_grad(addr: Int) -> GradMatvecFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[GradMatvecFn]()[]; s.free(); return r

@always_inline
fn _cvt_fused(addr: Int) -> FusedGradFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[FusedGradFn]()[]; s.free(); return r

@always_inline
fn _cvt_lsos(addr: Int) -> FusedLsOsFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[FusedLsOsFn]()[]; s.free(); return r

@always_inline
fn _cvt_3p(addr: Int) -> Fused3PFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[Fused3PFn]()[]; s.free(); return r

@always_inline
fn _cvt_diag(addr: Int) -> ExtractDiagFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[ExtractDiagFn]()[]; s.free(); return r

@always_inline
fn _cvt_upd(addr: Int) -> UpdateParamsFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[UpdateParamsFn]()[]; s.free(); return r

@always_inline
fn _cvt_unoise(addr: Int) -> UpdateNoiseFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[UpdateNoiseFn]()[]; s.free(); return r

@always_inline
fn _cvt_getf(addr: Int) -> GetFloatFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[GetFloatFn]()[]; s.free(); return r

@always_inline
fn _cvt_geti(addr: Int) -> GetIntFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[GetIntFn]()[]; s.free(); return r

@always_inline
fn _cvt_getptr(addr: Int) -> GetPtrFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[GetPtrFn]()[]; s.free(); return r

# No-op fn ptrs for when prediction is not available
fn _noop_cross_matvec(ptr: Int, out_ptr: UnsafePointer[Float32, MutAnyOrigin],
                      xtest: UnsafePointer[Float32, MutAnyOrigin],
                      v: UnsafePointer[Float32, MutAnyOrigin],
                      ntest: Int, ncols: Int) raises -> None:
    pass

fn _noop_extract_diag_test(ptr: Int, diag: UnsafePointer[Float32, MutAnyOrigin],
                           xtest: UnsafePointer[Float32, MutAnyOrigin],
                           ntest: Int) raises -> None:
    pass

fn _noop_fill_cross_covariance(ptr: Int, out_ptr: UnsafePointer[Float32, MutAnyOrigin],
                               xtest: UnsafePointer[Float32, MutAnyOrigin],
                               ntest: Int) raises -> None:
    pass

@always_inline
fn get_noop_cross() -> CrossMatvecFn:
    return _noop_cross_matvec

@always_inline
fn get_noop_diagtest() -> ExtractDiagTestFn:
    return _noop_extract_diag_test

@always_inline
fn get_noop_fill_cross() -> FillCrossCovFn:
    return _noop_fill_cross_covariance

fn _noop_get_noise_mode(ptr: Int) -> Int:
    return 0

fn _noop_get_noise_vector_ptr(ptr: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin]()

@always_inline
fn get_noop_noise_mode() -> GetIntFn:
    return _noop_get_noise_mode

@always_inline
fn get_noop_noise_vector_ptr() -> GetPtrFn:
    return _noop_get_noise_vector_ptr

@always_inline
fn _cvt_cross(addr: Int) -> CrossMatvecFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[CrossMatvecFn]()[]; s.free(); return r

@always_inline
fn _cvt_diagtest(addr: Int) -> ExtractDiagTestFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[ExtractDiagTestFn]()[]; s.free(); return r

@always_inline
fn _cvt_fill_cross(addr: Int) -> FillCrossCovFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[FillCrossCovFn]()[]; s.free(); return r

fn _noop_kron_grad(ptr: Int, out_ptr: UnsafePointer[Float32, MutAnyOrigin],
                   v: UnsafePointer[Float32, MutAnyOrigin],
                   B: UnsafePointer[Float32, MutAnyOrigin],
                   ncols: Int, T: Int, os: Float32) raises -> None:
    pass

@always_inline
fn get_noop_kron_grad() -> KroneckerGradFn:
    return _noop_kron_grad

@always_inline
fn _cvt_kron_grad(addr: Int) -> KroneckerGradFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[KroneckerGradFn]()[]; s.free(); return r

fn _noop_kron_fwd(ptr: Int, out_ptr: UnsafePointer[Float32, MutAnyOrigin],
                  v: UnsafePointer[Float32, MutAnyOrigin],
                  B: UnsafePointer[Float32, MutAnyOrigin],
                  noise: UnsafePointer[Float32, MutAnyOrigin],
                  ncols: Int, T: Int, os: Float32) raises -> None:
    pass

@always_inline
fn get_noop_kron_fwd() -> KroneckerFwdFn:
    return _noop_kron_fwd

@always_inline
fn _cvt_kron_fwd(addr: Int) -> KroneckerFwdFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[KroneckerFwdFn]()[]; s.free(); return r

@always_inline
fn _cvt_xptr(addr: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[UnsafePointer[Float32, MutAnyOrigin]]()[]; s.free(); return r

@always_inline
fn _cvt_i32ptr(addr: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[UnsafePointer[Int32, MutAnyOrigin]]()[]; s.free(); return r

@always_inline
fn _cvt_mixed_fwd(addr: Int) -> MixedFwdMatvecFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[MixedFwdMatvecFn]()[]; s.free(); return r

@always_inline
fn _cvt_mixed_fused_grad(addr: Int) -> MixedFusedGradMatvecFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[MixedFusedGradMatvecFn]()[]; s.free(); return r

@always_inline
fn _cvt_mixed_cross(addr: Int) -> MixedCrossMatvecFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[MixedCrossMatvecFn]()[]; s.free(); return r

@always_inline
fn _cvt_mixed_diag(addr: Int) -> MixedExtractDiagFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[MixedExtractDiagFn]()[]; s.free(); return r

@always_inline
fn _cvt_mixed_mat(addr: Int) -> MixedMaterializeFn:
    var s = alloc[Int](1); s[] = addr; var r = s.bitcast[MixedMaterializeFn]()[]; s.free(); return r

fn _noop_mixed_fwd(ptr: Int, out_ptr: UnsafePointer[Float32, MutAnyOrigin],
                   v: UnsafePointer[Float32, MutAnyOrigin],
                   cat_idx: UnsafePointer[Int32, MutAnyOrigin],
                   corr: UnsafePointer[Float32, MutAnyOrigin],
                   offs: UnsafePointer[Int32, MutAnyOrigin],
                   levs: UnsafePointer[Int32, MutAnyOrigin],
                   ncv: Int, ncols: Int, noise: Float32) raises -> None:
    pass

fn _noop_mixed_fused_grad(ptr: Int, out_ptr: UnsafePointer[Float32, MutAnyOrigin],
                           v: UnsafePointer[Float32, MutAnyOrigin],
                           cat_idx: UnsafePointer[Int32, MutAnyOrigin],
                           corr: UnsafePointer[Float32, MutAnyOrigin],
                           offs: UnsafePointer[Int32, MutAnyOrigin],
                           levs: UnsafePointer[Int32, MutAnyOrigin],
                           ncv: Int, ncols: Int) raises -> None:
    pass

fn _noop_mixed_cross(ptr: Int, out_ptr: UnsafePointer[Float32, MutAnyOrigin],
                     xtest: UnsafePointer[Float32, MutAnyOrigin],
                     v: UnsafePointer[Float32, MutAnyOrigin],
                     cat_test: UnsafePointer[Int32, MutAnyOrigin],
                     cat_train: UnsafePointer[Int32, MutAnyOrigin],
                     corr: UnsafePointer[Float32, MutAnyOrigin],
                     offs: UnsafePointer[Int32, MutAnyOrigin],
                     levs: UnsafePointer[Int32, MutAnyOrigin],
                     ncv: Int, ntest: Int, ncols: Int) raises -> None:
    pass

fn _noop_mixed_diag(ptr: Int, diag: UnsafePointer[Float32, MutAnyOrigin],
                    cat_idx: UnsafePointer[Int32, MutAnyOrigin],
                    corr: UnsafePointer[Float32, MutAnyOrigin],
                    offs: UnsafePointer[Int32, MutAnyOrigin],
                    levs: UnsafePointer[Int32, MutAnyOrigin],
                    ncv: Int) raises -> None:
    pass

fn _noop_mixed_mat(ptr: Int,
                   cat_idx: UnsafePointer[Int32, MutAnyOrigin],
                   corr: UnsafePointer[Float32, MutAnyOrigin],
                   offs: UnsafePointer[Int32, MutAnyOrigin],
                   levs: UnsafePointer[Int32, MutAnyOrigin],
                   ncv: Int) raises -> None:
    pass

@always_inline
fn get_noop_mixed_fwd() -> MixedFwdMatvecFn:
    return _noop_mixed_fwd

@always_inline
fn get_noop_mixed_fused_grad() -> MixedFusedGradMatvecFn:
    return _noop_mixed_fused_grad

@always_inline
fn get_noop_mixed_cross() -> MixedCrossMatvecFn:
    return _noop_mixed_cross

@always_inline
fn get_noop_mixed_diag() -> MixedExtractDiagFn:
    return _noop_mixed_diag

@always_inline
fn get_noop_mixed_mat() -> MixedMaterializeFn:
    return _noop_mixed_mat


# =============================================================================
# ErasedJITProvider: fn-ptr based JITGradientProvider (matches exp8 pattern)
# =============================================================================

struct ErasedJITProvider(JITGradientProvider, Copyable, Movable):
    """JITGradientProvider implementation via function pointers.
    
    All kernel-specific operations are forwarded through fn ptrs to the
    kernel .so. Fn ptrs are stored as actual function types (converted once
    at construction, not on every call).
    """
    var provider_ptr: Int
    
    # Immutable fields
    var _ctx: DeviceContext
    var _n: Int
    var _x_ptr: UnsafePointer[Float32, MutAnyOrigin]
    var _num_gradient_params: Int
    var _supports_fused_gradient: Bool
    var _supports_fused_ls_os: Bool
    var _supports_fused_3param: Bool
    
    # Function pointers (stored as actual fn types, NOT Int)
    var _forward_matvec: FwdMatvecFn
    var _gradient_matvec: GradMatvecFn
    var _fused_gradient_matvec: FusedGradFn
    var _fused_ls_os_gradient_matvec: FusedLsOsFn
    var _fused_3param_gradient_matvec: Fused3PFn
    var _extract_diagonal: ExtractDiagFn
    var _update_params: UpdateParamsFn
    var _update_noise: UpdateNoiseFn
    var _get_noise: GetFloatFn
    var _get_noise_mode: GetIntFn
    var _get_noise_vector_ptr: GetPtrFn
    var _get_diagonal_value: GetFloatFn
    
    # Prediction fn ptrs (optional — set to dummy if not available)
    var _cross_matvec: CrossMatvecFn
    var _extract_diagonal_test: ExtractDiagTestFn
    var _has_prediction: Bool
    var _fill_cross_covariance: FillCrossCovFn
    var _has_fill_cross_covariance: Bool
    
    # Fused Kronecker forward matvec fn ptr (optional)
    var _kronecker_forward_matvec: KroneckerFwdFn
    var _kronecker_gradient_matvec: KroneckerGradFn
    var _has_kronecker: Bool

    # Mixed (continuous × categorical) fn ptrs (optional)
    var _mixed_forward_matvec: MixedFwdMatvecFn
    var _mixed_fused_gradient_matvec: MixedFusedGradMatvecFn
    var _mixed_cross_matvec: MixedCrossMatvecFn
    var _mixed_extract_diagonal: MixedExtractDiagFn
    var _mixed_materialize: MixedMaterializeFn
    var _has_mixed: Bool
    
    fn __init__(
        out self,
        provider_ptr: Int,
        ctx: DeviceContext,
        n: Int,
        x_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_gradient_params: Int,
        supports_fused_gradient: Bool,
        supports_fused_ls_os: Bool,
        supports_fused_3param: Bool,
        forward_matvec: FwdMatvecFn,
        gradient_matvec: GradMatvecFn,
        fused_gradient_matvec: FusedGradFn,
        fused_ls_os_gradient_matvec: FusedLsOsFn,
        fused_3param_gradient_matvec: Fused3PFn,
        extract_diagonal: ExtractDiagFn,
        update_params: UpdateParamsFn,
        update_noise: UpdateNoiseFn,
        get_noise: GetFloatFn,
        get_diagonal_value: GetFloatFn,
        cross_matvec: CrossMatvecFn,
        extract_diagonal_test: ExtractDiagTestFn,
        has_prediction: Bool,
        kronecker_forward_matvec: KroneckerFwdFn,
        kronecker_gradient_matvec: KroneckerGradFn,
        has_kronecker: Bool,
        fill_cross_covariance: FillCrossCovFn = get_noop_fill_cross(),
        has_fill_cross_covariance: Bool = False,
        mixed_forward_matvec: MixedFwdMatvecFn = get_noop_mixed_fwd(),
        mixed_fused_gradient_matvec: MixedFusedGradMatvecFn = get_noop_mixed_fused_grad(),
        mixed_cross_matvec: MixedCrossMatvecFn = get_noop_mixed_cross(),
        mixed_extract_diagonal: MixedExtractDiagFn = get_noop_mixed_diag(),
        mixed_materialize: MixedMaterializeFn = get_noop_mixed_mat(),
        has_mixed: Bool = False,
        get_noise_mode: GetIntFn = get_noop_noise_mode(),
        get_noise_vector_ptr: GetPtrFn = get_noop_noise_vector_ptr(),
    ):
        self.provider_ptr = provider_ptr
        self._ctx = ctx
        self._n = n
        self._x_ptr = x_ptr
        self._num_gradient_params = num_gradient_params
        self._supports_fused_gradient = supports_fused_gradient
        self._supports_fused_ls_os = supports_fused_ls_os
        self._supports_fused_3param = supports_fused_3param
        self._forward_matvec = forward_matvec
        self._gradient_matvec = gradient_matvec
        self._fused_gradient_matvec = fused_gradient_matvec
        self._fused_ls_os_gradient_matvec = fused_ls_os_gradient_matvec
        self._fused_3param_gradient_matvec = fused_3param_gradient_matvec
        self._extract_diagonal = extract_diagonal
        self._update_params = update_params
        self._update_noise = update_noise
        self._get_noise = get_noise
        self._get_noise_mode = get_noise_mode
        self._get_noise_vector_ptr = get_noise_vector_ptr
        self._get_diagonal_value = get_diagonal_value
        self._cross_matvec = cross_matvec
        self._extract_diagonal_test = extract_diagonal_test
        self._has_prediction = has_prediction
        self._fill_cross_covariance = fill_cross_covariance
        self._has_fill_cross_covariance = has_fill_cross_covariance
        self._kronecker_forward_matvec = kronecker_forward_matvec
        self._kronecker_gradient_matvec = kronecker_gradient_matvec
        self._has_kronecker = has_kronecker
        self._mixed_forward_matvec = mixed_forward_matvec
        self._mixed_fused_gradient_matvec = mixed_fused_gradient_matvec
        self._mixed_cross_matvec = mixed_cross_matvec
        self._mixed_extract_diagonal = mixed_extract_diagonal
        self._mixed_materialize = mixed_materialize
        self._has_mixed = has_mixed
    
    fn __moveinit__(out self, deinit other: Self):
        self.provider_ptr = other.provider_ptr
        self._ctx = other._ctx
        self._n = other._n
        self._x_ptr = other._x_ptr
        self._num_gradient_params = other._num_gradient_params
        self._supports_fused_gradient = other._supports_fused_gradient
        self._supports_fused_ls_os = other._supports_fused_ls_os
        self._supports_fused_3param = other._supports_fused_3param
        self._forward_matvec = other._forward_matvec
        self._gradient_matvec = other._gradient_matvec
        self._fused_gradient_matvec = other._fused_gradient_matvec
        self._fused_ls_os_gradient_matvec = other._fused_ls_os_gradient_matvec
        self._fused_3param_gradient_matvec = other._fused_3param_gradient_matvec
        self._extract_diagonal = other._extract_diagonal
        self._update_params = other._update_params
        self._update_noise = other._update_noise
        self._get_noise = other._get_noise
        self._get_noise_mode = other._get_noise_mode
        self._get_noise_vector_ptr = other._get_noise_vector_ptr
        self._get_diagonal_value = other._get_diagonal_value
        self._cross_matvec = other._cross_matvec
        self._extract_diagonal_test = other._extract_diagonal_test
        self._has_prediction = other._has_prediction
        self._fill_cross_covariance = other._fill_cross_covariance
        self._has_fill_cross_covariance = other._has_fill_cross_covariance
        self._kronecker_forward_matvec = other._kronecker_forward_matvec
        self._kronecker_gradient_matvec = other._kronecker_gradient_matvec
        self._has_kronecker = other._has_kronecker
        self._mixed_forward_matvec = other._mixed_forward_matvec
        self._mixed_fused_gradient_matvec = other._mixed_fused_gradient_matvec
        self._mixed_cross_matvec = other._mixed_cross_matvec
        self._mixed_extract_diagonal = other._mixed_extract_diagonal
        self._mixed_materialize = other._mixed_materialize
        self._has_mixed = other._has_mixed
    
    fn __copyinit__(out self, other: Self):
        self.provider_ptr = other.provider_ptr
        self._ctx = other._ctx
        self._n = other._n
        self._x_ptr = other._x_ptr
        self._num_gradient_params = other._num_gradient_params
        self._supports_fused_gradient = other._supports_fused_gradient
        self._supports_fused_ls_os = other._supports_fused_ls_os
        self._supports_fused_3param = other._supports_fused_3param
        self._forward_matvec = other._forward_matvec
        self._gradient_matvec = other._gradient_matvec
        self._fused_gradient_matvec = other._fused_gradient_matvec
        self._fused_ls_os_gradient_matvec = other._fused_ls_os_gradient_matvec
        self._fused_3param_gradient_matvec = other._fused_3param_gradient_matvec
        self._extract_diagonal = other._extract_diagonal
        self._update_params = other._update_params
        self._update_noise = other._update_noise
        self._get_noise = other._get_noise
        self._get_noise_mode = other._get_noise_mode
        self._get_noise_vector_ptr = other._get_noise_vector_ptr
        self._get_diagonal_value = other._get_diagonal_value
        self._cross_matvec = other._cross_matvec
        self._extract_diagonal_test = other._extract_diagonal_test
        self._has_prediction = other._has_prediction
        self._fill_cross_covariance = other._fill_cross_covariance
        self._has_fill_cross_covariance = other._has_fill_cross_covariance
        self._kronecker_forward_matvec = other._kronecker_forward_matvec
        self._kronecker_gradient_matvec = other._kronecker_gradient_matvec
        self._has_kronecker = other._has_kronecker
        self._mixed_forward_matvec = other._mixed_forward_matvec
        self._mixed_fused_gradient_matvec = other._mixed_fused_gradient_matvec
        self._mixed_cross_matvec = other._mixed_cross_matvec
        self._mixed_extract_diagonal = other._mixed_extract_diagonal
        self._mixed_materialize = other._mixed_materialize
        self._has_mixed = other._has_mixed
    
    fn clone(self) -> Self:
        """Create an explicit copy of this provider."""
        return Self(
            provider_ptr=self.provider_ptr,
            ctx=self._ctx, n=self._n, x_ptr=self._x_ptr,
            num_gradient_params=self._num_gradient_params,
            supports_fused_gradient=self._supports_fused_gradient,
            supports_fused_ls_os=self._supports_fused_ls_os,
            supports_fused_3param=self._supports_fused_3param,
            forward_matvec=self._forward_matvec,
            gradient_matvec=self._gradient_matvec,
            fused_gradient_matvec=self._fused_gradient_matvec,
            fused_ls_os_gradient_matvec=self._fused_ls_os_gradient_matvec,
            fused_3param_gradient_matvec=self._fused_3param_gradient_matvec,
            extract_diagonal=self._extract_diagonal,
            update_params=self._update_params,
            update_noise=self._update_noise,
            get_noise=self._get_noise,
            get_noise_mode=self._get_noise_mode,
            get_noise_vector_ptr=self._get_noise_vector_ptr,
            get_diagonal_value=self._get_diagonal_value,
            cross_matvec=self._cross_matvec,
            extract_diagonal_test=self._extract_diagonal_test,
            has_prediction=self._has_prediction,
            fill_cross_covariance=self._fill_cross_covariance,
            has_fill_cross_covariance=self._has_fill_cross_covariance,
            kronecker_forward_matvec=self._kronecker_forward_matvec,
            kronecker_gradient_matvec=self._kronecker_gradient_matvec,
            has_kronecker=self._has_kronecker,
            mixed_forward_matvec=self._mixed_forward_matvec,
            mixed_fused_gradient_matvec=self._mixed_fused_gradient_matvec,
            mixed_cross_matvec=self._mixed_cross_matvec,
            mixed_extract_diagonal=self._mixed_extract_diagonal,
            mixed_materialize=self._mixed_materialize,
            has_mixed=self._has_mixed,
        )
    
    # --- ForwardProvider methods ---
    
    fn forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        # The engine allocates CG buffers with self._ctx while the generated
        # provider launches kernels with its own internal DeviceContext. Sync
        # the engine context before handing its buffers to the provider so the
        # provider cannot read stale RHS data from a different stream/context.
        self._ctx.synchronize()
        self._forward_matvec(self.provider_ptr, out_ptr, v_ptr, num_cols)
    
    fn get_n(self) -> Int:
        return self._n
    
    fn get_ctx(self) -> DeviceContext:
        return self._ctx
    
    fn get_noise(self) -> Float32:
        return self._get_noise(self.provider_ptr)

    fn get_noise_mode(self) -> Int:
        return self._get_noise_mode(self.provider_ptr)

    fn get_noise_vector_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self._get_noise_vector_ptr(self.provider_ptr)
    
    fn get_diagonal_value(self) -> Float32:
        return self._get_diagonal_value(self.provider_ptr)
    
    fn extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        self._ctx.synchronize()
        self._extract_diagonal(self.provider_ptr, diag_ptr)
    
    fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]:
        return self._x_ptr
    
    # --- GradientProvider methods ---
    
    fn gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        param_index: Int,
        sync: Bool = True,
    ) raises:
        self._ctx.synchronize()
        self._gradient_matvec(self.provider_ptr, out_ptr, v_ptr, num_cols, param_index, sync)
    
    fn num_gradient_params(self) -> Int:
        return self._num_gradient_params
    
    fn supports_fused_gradient(self) -> Bool:
        return self._supports_fused_gradient
    
    fn fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        self._ctx.synchronize()
        self._fused_gradient_matvec(self.provider_ptr, out_ptr, v_ptr, num_cols)
    
    fn supports_fused_ls_os(self) -> Bool:
        return self._supports_fused_ls_os
    
    fn fused_ls_os_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        self._ctx.synchronize()
        self._fused_ls_os_gradient_matvec(self.provider_ptr, ls_out_ptr, os_out_ptr, v_ptr, num_cols)
    
    fn supports_fused_3param(self) -> Bool:
        return self._supports_fused_3param
    
    fn fused_3param_gradient_matvec(
        self,
        ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        p1_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
    ) raises:
        self._ctx.synchronize()
        self._fused_3param_gradient_matvec(self.provider_ptr, ls_out_ptr, p1_out_ptr, os_out_ptr, v_ptr, num_cols)
    
    # --- JITGradientProvider methods ---
    
    fn update_params(
        mut self,
        params_host_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        self._update_params(self.provider_ptr, params_host_ptr)
    
    fn update_noise(mut self, noise: Float32):
        self._update_noise(self.provider_ptr, noise)
    
    # --- Prediction methods (optional, check _has_prediction first) ---
    
    fn has_prediction(self) -> Bool:
        return self._has_prediction
    
    fn cross_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n_test: Int,
        num_cols: Int,
    ) raises:
        """Compute K(X_test, X_train) @ v."""
        self._ctx.synchronize()
        self._cross_matvec(self.provider_ptr, out_ptr, x_test_ptr, v_ptr, n_test, num_cols)
    
    fn extract_diagonal_test(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n_test: Int,
    ) raises:
        """Compute k(x*_i, x*_i) for test points."""
        self._ctx.synchronize()
        self._extract_diagonal_test(self.provider_ptr, diag_ptr, x_test_ptr, n_test)

    fn has_fill_cross_covariance(self) -> Bool:
        return self._has_fill_cross_covariance

    fn fill_cross_covariance(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n_test: Int,
    ) raises:
        self._ctx.synchronize()
        self._fill_cross_covariance(self.provider_ptr, out_ptr, x_test_ptr, n_test)
    
    # --- Kronecker methods (optional, check _has_kronecker first) ---
    
    fn has_kronecker(self) -> Bool:
        return self._has_kronecker
    
    fn kronecker_forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        B_ptr: UnsafePointer[Float32, MutAnyOrigin],
        noise_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        T: Int,
        outputscale: Float32,
    ) raises:
        """Fused Kronecker forward matvec: no reshuffle, no separate B combine."""
        self._ctx.synchronize()
        self._kronecker_forward_matvec(self.provider_ptr, out_ptr, v_ptr, B_ptr, noise_ptr, num_cols, T, outputscale)
    
    fn kronecker_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        B_ptr: UnsafePointer[Float32, MutAnyOrigin],
        num_cols: Int,
        T: Int,
        outputscale: Float32,
    ) raises:
        """Fused Kronecker gradient matvec: all params × B combine in one kernel."""
        self._ctx.synchronize()
        self._kronecker_gradient_matvec(self.provider_ptr, out_ptr, v_ptr, B_ptr, num_cols, T, outputscale)

    # --- Mixed (continuous × categorical) methods ---

    fn has_mixed(self) -> Bool:
        return self._has_mixed

    fn mixed_forward_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        cat_indices: UnsafePointer[Int32, MutAnyOrigin],
        corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
        offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
        levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
        num_cat_vars: Int,
        num_cols: Int,
        noise: Float32,
    ) raises:
        """Mixed forward: K_cont * prod_cv R_cv @ v + noise*v."""
        self._ctx.synchronize()
        self._mixed_forward_matvec(
            self.provider_ptr, out_ptr, v_ptr,
            cat_indices, corr_flat_ptr, offsets_ptr, levels_ptr,
            num_cat_vars, num_cols, noise,
        )

    fn mixed_fused_gradient_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        cat_indices: UnsafePointer[Int32, MutAnyOrigin],
        corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
        offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
        levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
        num_cat_vars: Int,
        num_cols: Int,
    ) raises:
        """Mixed fused all-gradients: (dK/dtheta * R) @ v for all cont params."""
        self._ctx.synchronize()
        self._mixed_fused_gradient_matvec(
            self.provider_ptr, out_ptr, v_ptr,
            cat_indices, corr_flat_ptr, offsets_ptr, levels_ptr,
            num_cat_vars, num_cols,
        )

    fn mixed_cross_matvec(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        x_test_ptr: UnsafePointer[Float32, MutAnyOrigin],
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        cat_test: UnsafePointer[Int32, MutAnyOrigin],
        cat_train: UnsafePointer[Int32, MutAnyOrigin],
        corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
        offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
        levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
        num_cat_vars: Int,
        n_test: Int,
        num_cols: Int,
    ) raises:
        """Mixed cross-covariance: K_cont(X_test, X_train) * R @ v."""
        self._ctx.synchronize()
        self._mixed_cross_matvec(
            self.provider_ptr, out_ptr, x_test_ptr, v_ptr,
            cat_test, cat_train, corr_flat_ptr, offsets_ptr, levels_ptr,
            num_cat_vars, n_test, num_cols,
        )

    fn mixed_extract_diagonal(
        self,
        diag_ptr: UnsafePointer[Float32, MutAnyOrigin],
        cat_indices: UnsafePointer[Int32, MutAnyOrigin],
        corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
        offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
        levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
        num_cat_vars: Int,
    ) raises:
        """Mixed diagonal: diag[i] = K(x_i,x_i) * prod_cv R_cv(c_i,c_i)."""
        self._ctx.synchronize()
        self._mixed_extract_diagonal(
            self.provider_ptr, diag_ptr,
            cat_indices, corr_flat_ptr, offsets_ptr, levels_ptr,
            num_cat_vars,
        )

    fn mixed_materialize(
        self,
        cat_indices: UnsafePointer[Int32, MutAnyOrigin],
        corr_flat_ptr: UnsafePointer[Float32, MutAnyOrigin],
        offsets_ptr: UnsafePointer[Int32, MutAnyOrigin],
        levels_ptr: UnsafePointer[Int32, MutAnyOrigin],
        num_cat_vars: Int,
    ) raises:
        """Materialize the full mixed kernel matrix K_mixed on GPU."""
        self._ctx.synchronize()
        self._mixed_materialize(
            self.provider_ptr,
            cat_indices,
            corr_flat_ptr,
            offsets_ptr,
            levels_ptr,
            num_cat_vars,
        )

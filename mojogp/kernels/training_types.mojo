"""Types and structs for GP training.

Provides result structs and Adam optimizer state structs used by training functions.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

alias float_dtype = DType.float32


# =============================================================================
# Result Structs
# =============================================================================

@fieldwise_init
struct NLLResult(Copyable):
    """Result from NLL computation.
    
    Fields:
        nll: Negative log-likelihood (per-sample)
        inv_quad: Inverse quadratic form y^T @ K^{-1} @ y
        log_det: Log determinant of K
        cg_iterations: Number of CG iterations used
        alpha: Solution to K @ alpha = y (on host)
    """
    var nll: Float32
    var inv_quad: Float32
    var log_det: Float32
    var cg_iterations: Int
    var alpha: HostBuffer[float_dtype]


@fieldwise_init
struct GradientResult(Copyable):
    """Result from gradient computation.
    
    Fields:
        grad_lengthscale: Gradient w.r.t. lengthscale
        grad_noise: Gradient w.r.t. noise
        grad_outputscale: Gradient w.r.t. output scale
        grad_param1: Gradient w.r.t. kernel param1 (period/alpha) - 0 if not applicable
    """
    var grad_lengthscale: Float32
    var grad_noise: Float32
    var grad_outputscale: Float32
    var grad_param1: Float32


struct TrainingResult:
    """Result from training.
    
    Fields:
        lengthscale: Optimized lengthscale
        noise: Optimized noise
        outputscale: Optimized output scale
        kernel_param1: Secondary kernel parameter (period/alpha/variance/degree)
        kernel_param2: Tertiary kernel parameter (polynomial offset)
        mean: Learned constant mean function value
        final_nll: Final NLL value
        nll_history: List of NLL values at each iteration
        iterations: Number of training iterations
        converged: Whether training converged (early stopping)
        lanczos_root: Cached Lanczos root S [n × r] for LOVE variance (row-major)
        lanczos_rank: Rank r (number of Lanczos iterations)
        n: Number of training points
        has_alpha: Whether alpha contains valid cached data
        alpha: Cached alpha = K^{-1} @ (y - mean) [n] for fast prediction (row-major).
               Only valid when has_alpha=True. When False, prediction computes alpha on demand.
    """
    var lengthscale: Float32
    var noise: Float32
    var outputscale: Float32
    var kernel_param1: Float32
    var kernel_param2: Float32
    var mean: Float32
    var final_nll: Float32
    var nll_history: List[Float32]
    var iterations: Int
    var converged: Bool
    var lanczos_root: HostBuffer[float_dtype]
    var lanczos_rank: Int
    var n: Int
    var has_alpha: Bool
    var alpha: HostBuffer[float_dtype]
    
    fn __init__(
        out self,
        lengthscale: Float32,
        noise: Float32,
        outputscale: Float32,
        kernel_param1: Float32,
        kernel_param2: Float32,
        mean: Float32,
        final_nll: Float32,
        var nll_history: List[Float32],
        iterations: Int,
        converged: Bool,
        var lanczos_root: HostBuffer[float_dtype],
        lanczos_rank: Int,
        n: Int,
        var alpha: HostBuffer[float_dtype],
        has_alpha: Bool = True,
    ):
        self.lengthscale = lengthscale
        self.noise = noise
        self.outputscale = outputscale
        self.kernel_param1 = kernel_param1
        self.kernel_param2 = kernel_param2
        self.mean = mean
        self.final_nll = final_nll
        self.nll_history = nll_history^
        self.iterations = iterations
        self.converged = converged
        self.lanczos_root = lanczos_root^
        self.lanczos_rank = lanczos_rank
        self.n = n
        self.has_alpha = has_alpha
        self.alpha = alpha^


struct TrainingResultARD:
    """Result from ARD training.
    
    Fields:
        lengthscales: Optimized per-dimension lengthscales [dim]
        noise: Optimized noise
        outputscale: Optimized output scale
        kernel_param1: Secondary kernel parameter (period/alpha/variance/degree)
        kernel_param2: Tertiary kernel parameter (polynomial offset)
        mean: Learned constant mean function value
        final_nll: Final NLL value
        nll_history: List of NLL values at each iteration
        iterations: Number of training iterations
        converged: Whether training converged (early stopping)
        lanczos_root: Cached Lanczos root S [n × r] for LOVE variance (row-major)
        lanczos_rank: Rank r (number of Lanczos iterations)
        n: Number of training points
        dim: Input dimension
        has_alpha: Whether alpha contains valid cached data
        alpha: Cached alpha = K^{-1} @ (y - mean) [n] for fast prediction (row-major).
               Only valid when has_alpha=True. When False, prediction computes alpha on demand.
    """
    var lengthscales: HostBuffer[float_dtype]  # [dim] per-dimension lengthscales
    var noise: Float32
    var outputscale: Float32
    var kernel_param1: Float32
    var kernel_param2: Float32
    var mean: Float32
    var final_nll: Float32
    var nll_history: List[Float32]
    var iterations: Int
    var converged: Bool
    var lanczos_root: HostBuffer[float_dtype]
    var lanczos_rank: Int
    var n: Int
    var dim: Int
    var has_alpha: Bool
    var alpha: HostBuffer[float_dtype]
    
    fn __init__(
        out self,
        var lengthscales: HostBuffer[float_dtype],
        noise: Float32,
        outputscale: Float32,
        kernel_param1: Float32,
        kernel_param2: Float32,
        mean: Float32,
        final_nll: Float32,
        var nll_history: List[Float32],
        iterations: Int,
        converged: Bool,
        var lanczos_root: HostBuffer[float_dtype],
        lanczos_rank: Int,
        n: Int,
        dim: Int,
        var alpha: HostBuffer[float_dtype],
        has_alpha: Bool = True,
    ):
        self.lengthscales = lengthscales^
        self.noise = noise
        self.outputscale = outputscale
        self.kernel_param1 = kernel_param1
        self.kernel_param2 = kernel_param2
        self.mean = mean
        self.final_nll = final_nll
        self.nll_history = nll_history^
        self.iterations = iterations
        self.converged = converged
        self.lanczos_root = lanczos_root^
        self.lanczos_rank = lanczos_rank
        self.n = n
        self.dim = dim
        self.has_alpha = has_alpha
        self.alpha = alpha^


# =============================================================================
# Adam Optimizer State
# =============================================================================

struct AdamState(ImplicitlyCopyable):
    """Adam optimizer state for 3-6 parameters.
    
    Fields:
        m_ls, v_ls: First/second moment for lengthscale
        m_noise, v_noise: First/second moment for noise
        m_os, v_os: First/second moment for output scale
        m_param1, v_param1: First/second moment for param1 (period/alpha)
        m_param2, v_param2: First/second moment for param2 (offset for Polynomial)
        m_mean, v_mean: First/second moment for constant mean function
        t: Time step (iteration count)
    """
    var m_ls: Float32
    var v_ls: Float32
    var m_noise: Float32
    var v_noise: Float32
    var m_os: Float32
    var v_os: Float32
    var m_param1: Float32
    var v_param1: Float32
    var m_param2: Float32
    var v_param2: Float32
    var m_mean: Float32
    var v_mean: Float32
    var t: Int
    
    fn __init__(out self):
        """Initialize all moments to zero."""
        self.m_ls = Float32(0.0)
        self.v_ls = Float32(0.0)
        self.m_noise = Float32(0.0)
        self.v_noise = Float32(0.0)
        self.m_os = Float32(0.0)
        self.v_os = Float32(0.0)
        self.m_param1 = Float32(0.0)
        self.v_param1 = Float32(0.0)
        self.m_param2 = Float32(0.0)
        self.v_param2 = Float32(0.0)
        self.m_mean = Float32(0.0)
        self.v_mean = Float32(0.0)
        self.t = 0


struct AdamStateARD(Movable):
    """Adam optimizer state for ARD (per-dimension lengthscales).
    
    Fields:
        m_ls: First moment for each lengthscale [dim]
        v_ls: Second moment for each lengthscale [dim]
        m_noise, v_noise: First/second moment for noise
        m_os, v_os: First/second moment for output scale
        m_param1, v_param1: First/second moment for param1 (period/alpha)
        m_param2, v_param2: First/second moment for param2 (offset for Polynomial)
        m_mean, v_mean: First/second moment for constant mean function
        t: Time step (iteration count)
        dim: Input dimension
    """
    var m_ls: List[Float32]
    var v_ls: List[Float32]
    var m_noise: Float32
    var v_noise: Float32
    var m_os: Float32
    var v_os: Float32
    var m_param1: Float32
    var v_param1: Float32
    var m_param2: Float32
    var v_param2: Float32
    var m_mean: Float32
    var v_mean: Float32
    var t: Int
    var dim: Int
    
    fn __init__(out self, dim: Int):
        """Initialize all moments to zero."""
        self.dim = dim
        self.m_ls = List[Float32]()
        self.v_ls = List[Float32]()
        for _ in range(dim):
            self.m_ls.append(Float32(0.0))
            self.v_ls.append(Float32(0.0))
        self.m_noise = Float32(0.0)
        self.v_noise = Float32(0.0)
        self.m_os = Float32(0.0)
        self.v_os = Float32(0.0)
        self.m_param1 = Float32(0.0)
        self.v_param1 = Float32(0.0)
        self.m_param2 = Float32(0.0)
        self.v_param2 = Float32(0.0)
        self.m_mean = Float32(0.0)
        self.v_mean = Float32(0.0)
        self.t = 0
    
    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.m_ls = other.m_ls^
        self.v_ls = other.v_ls^
        self.m_noise = other.m_noise
        self.v_noise = other.v_noise
        self.m_os = other.m_os
        self.v_os = other.v_os
        self.m_param1 = other.m_param1
        self.v_param1 = other.v_param1
        self.m_param2 = other.m_param2
        self.v_param2 = other.v_param2
        self.m_mean = other.m_mean
        self.v_mean = other.v_mean
        self.t = other.t
        self.dim = other.dim


# =============================================================================
# Generic Adam State for Composite Kernels (N parameters)
# =============================================================================

struct AdamStateGeneric(Movable):
    """Adam optimizer state for N kernel parameters + noise + mean.
    
    This is the generic version of AdamState that supports composite kernels
    with arbitrary numbers of parameters. Instead of hardcoded m_ls/v_ls fields,
    it uses Lists for all kernel parameter moments.
    
    Used by: train_gp_composite() for composite kernel training.
    
    Fields:
        m: First moment estimates for all N+1 parameters [N kernel params + noise]
        v: Second moment estimates for all N+1 parameters
        m_mean, v_mean: First/second moment for constant mean function
        t: Time step (iteration count)
        num_params: Total number of parameters (N+1)
    """
    var m: List[Float32]  # First moments for N+1 params
    var v: List[Float32]  # Second moments for N+1 params
    var m_mean: Float32
    var v_mean: Float32
    var t: Int
    var num_params: Int  # N+1 (N kernel params + noise)
    
    fn __init__(out self, num_kernel_params: Int):
        """Initialize all moments to zero.
        
        Args:
            num_kernel_params: Number of kernel parameters (N).
                               Total params = N + 1 (including noise).
        """
        self.num_params = num_kernel_params + 1  # +1 for noise
        self.m = List[Float32]()
        self.v = List[Float32]()
        for _ in range(self.num_params):
            self.m.append(Float32(0.0))
            self.v.append(Float32(0.0))
        self.m_mean = Float32(0.0)
        self.v_mean = Float32(0.0)
        self.t = 0
    
    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.m = other.m^
        self.v = other.v^
        self.m_mean = other.m_mean
        self.v_mean = other.v_mean
        self.t = other.t
        self.num_params = other.num_params


struct AdamUpdateResultGeneric(Movable):
    """Result from generic Adam update step.
    
    Fields:
        state: Updated Adam state
        raw_params: Updated raw (unconstrained) parameters [N+1]
    """
    var state: AdamStateGeneric
    var raw_params: List[Float32]  # [N+1] raw parameters
    
    fn __init__(out self, var state: AdamStateGeneric, var raw_params: List[Float32]):
        self.state = state^
        self.raw_params = raw_params^
    
    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.state = other.state^
        self.raw_params = other.raw_params^


# =============================================================================
# Generic Training Result for Composite Kernels
# =============================================================================

struct TrainingResultGeneric(Movable):
    """Result from composite kernel training.
    
    Fields:
        final_params: Optimized kernel parameters [N]
        noise: Optimized noise variance
        mean: Learned constant mean function value
        final_nll: Final NLL value
        iterations: Number of training iterations
        converged: Whether training converged (early stopping)
        lanczos_root: Cached Lanczos root S [n × r] for LOVE variance (row-major)
        lanczos_rank: Rank r (number of Lanczos iterations)
        n: Number of training points
        num_kernel_params: Number of kernel parameters (N)
        alpha: Cached best-seen alpha = K^{-1} @ (y - mean) [n]
        noise_function_params: Learned input-dependent raw noise-function params, if any
    """
    var final_params: List[Float32]  # [N] kernel parameters
    var noise: Float32
    var mean: Float32
    var final_nll: Float32
    var iterations: Int
    var converged: Bool
    var lanczos_root: HostBuffer[float_dtype]
    var lanczos_rank: Int
    var n: Int
    var num_kernel_params: Int
    var alpha: HostBuffer[float_dtype]
    var has_alpha: Bool
    var iter_times_ns: List[Int]  # Per-iteration wall-clock times in nanoseconds
    var nll_history: List[Float32]  # NLL value at each iteration
    var cg_iterations_history: List[Int]  # Realized CG iterations per optimizer step
    var precond_build_count: Int  # Number of preconditioner builds including initial build
    var precond_build_total_ns: Int  # Total wall time spent building preconditioners
    var precond_rank_history: List[Int]  # Actual preconditioner rank used at each optimizer step
    var precond_rebuild_steps: List[Int]  # Optimizer-step indices where a rebuild occurred
    var noise_function_params: List[Float32]  # [bias, weights...] raw learned noise-function params

    fn __init__(out self, var final_params: List[Float32], noise: Float32,
                mean: Float32, final_nll: Float32, iterations: Int, converged: Bool,
                var lanczos_root: HostBuffer[float_dtype], lanczos_rank: Int,
        n: Int, num_kernel_params: Int,
        var alpha: HostBuffer[float_dtype],
        has_alpha: Bool = True,
        var iter_times_ns: List[Int] = List[Int](),
        var nll_history: List[Float32] = List[Float32](),
        var cg_iterations_history: List[Int] = List[Int](),
        precond_build_count: Int = 0,
        precond_build_total_ns: Int = 0,
        var precond_rank_history: List[Int] = List[Int](),
        var precond_rebuild_steps: List[Int] = List[Int](),
        var noise_function_params: List[Float32] = List[Float32]()):
        self.final_params = final_params^
        self.noise = noise
        self.mean = mean
        self.final_nll = final_nll
        self.iterations = iterations
        self.converged = converged
        self.lanczos_root = lanczos_root^
        self.lanczos_rank = lanczos_rank
        self.n = n
        self.num_kernel_params = num_kernel_params
        self.alpha = alpha^
        self.has_alpha = has_alpha
        self.iter_times_ns = iter_times_ns^
        self.nll_history = nll_history^
        self.cg_iterations_history = cg_iterations_history^
        self.precond_build_count = precond_build_count
        self.precond_build_total_ns = precond_build_total_ns
        self.precond_rank_history = precond_rank_history^
        self.precond_rebuild_steps = precond_rebuild_steps^
        self.noise_function_params = noise_function_params^

    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.final_params = other.final_params^
        self.noise = other.noise
        self.mean = other.mean
        self.final_nll = other.final_nll
        self.iterations = other.iterations
        self.converged = other.converged
        self.lanczos_root = other.lanczos_root^
        self.lanczos_rank = other.lanczos_rank
        self.n = other.n
        self.num_kernel_params = other.num_kernel_params
        self.alpha = other.alpha^
        self.has_alpha = other.has_alpha
        self.iter_times_ns = other.iter_times_ns^
        self.nll_history = other.nll_history^
        self.cg_iterations_history = other.cg_iterations_history^
        self.precond_build_count = other.precond_build_count
        self.precond_build_total_ns = other.precond_build_total_ns
        self.precond_rank_history = other.precond_rank_history^
        self.precond_rebuild_steps = other.precond_rebuild_steps^
        self.noise_function_params = other.noise_function_params^



# =============================================================================
# Adam Optimizer
# =============================================================================
# NOTE: Param1 (period/alpha) remains fixed until the training loop includes it
# in the Adam state and applies the required positive transform.

struct AdamUpdateResult(Copyable):
    """Result from Adam update step.
    
    Fields:
        state: Updated Adam state
        raw_ls: Updated raw lengthscale
        raw_noise: Updated raw noise
        raw_os: Updated raw output scale
        raw_param1: Updated raw param1 (period/alpha), only used for periodic/RQ/linear
        raw_param2: Updated raw param2 (offset), only used for polynomial
    """
    var state: AdamState
    var raw_ls: Float32
    var raw_noise: Float32
    var raw_os: Float32
    var raw_param1: Float32
    var raw_param2: Float32
    
    fn __init__(out self, state: AdamState, raw_ls: Float32, raw_noise: Float32, raw_os: Float32, raw_param1: Float32 = Float32(0.0), raw_param2: Float32 = Float32(0.0)):
        self.state = state
        self.raw_ls = raw_ls
        self.raw_noise = raw_noise
        self.raw_os = raw_os
        self.raw_param1 = raw_param1
        self.raw_param2 = raw_param2


struct AdamUpdateResultARD(Movable):
    """Result from ARD Adam update step."""
    var state: AdamStateARD
    var raw_lengthscales: List[Float32]  # [dim]
    var raw_noise: Float32
    var raw_os: Float32
    
    fn __init__(out self, var state: AdamStateARD, var raw_lengthscales: List[Float32], raw_noise: Float32, raw_os: Float32):
        self.state = state^
        self.raw_lengthscales = raw_lengthscales^
        self.raw_noise = raw_noise
        self.raw_os = raw_os
    
    fn __moveinit__(out self, deinit other: Self):
        """Move constructor."""
        self.state = other.state^
        self.raw_lengthscales = other.raw_lengthscales^
        self.raw_noise = other.raw_noise
        self.raw_os = other.raw_os

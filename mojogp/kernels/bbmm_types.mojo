"""BBMM Types and Structs.

This module contains the type definitions and data structures used by the BBMM
(Black-Box Matrix-Matrix) CG solver and related functions. Extracted from
combined_inv_quad_logdet.mojo as a pure refactoring — no logic changes.

Types:
- CGBufferPool: Reusable GPU buffer pool for CG solver iterations
- BBMMPrecondType: Preconditioner type selector (PIVOTED_CHOLESKY)
- CGResultWithTridiag: CG result with tridiagonal matrices for log-det
- UnifiedBBMMResult: Full BBMM result with NLL, gradients, and CG solution
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer

alias float_dtype = DType.float32


# =============================================================================
# CG Buffer Pool for Memory Reuse
# =============================================================================

struct CGBufferPool:
    """Reusable buffer pool for CG solver and BBMM to eliminate per-iteration allocations.
    
    This struct caches GPU buffers and reuses them across training iterations, avoiding
    the overhead of allocating ~40+ buffers per bbmm_compute_nll_and_gradients() call.
    
    Buffer categories:
    1. CG-level buffers: x, r, z, p, Ap, etc. (used in batched_cg_with_pivoted_cholesky)
    2. BBMM-level buffers: rhs, probes, gradient outputs, etc. (used in bbmm_compute_nll_and_gradients)
    3. Tridiagonal tracking buffers: all_alphas, all_betas (for log-det computation)
    
    Usage:
        var pool = CGBufferPool()
        pool.ensure_capacity(ctx, n, num_cols, max_tridiag_iter)
        # Pass pool to bbmm_compute_nll_and_gradients() and batched_cg_with_pivoted_cholesky()
    """
    # -------------------------------------------------------------------------
    # CG-level buffers (used in batched_cg_with_pivoted_cholesky)
    # -------------------------------------------------------------------------
    var x: DeviceBuffer[float_dtype]
    var r: DeviceBuffer[float_dtype]
    var z: DeviceBuffer[float_dtype]
    var p: DeviceBuffer[float_dtype]
    var Ap: DeviceBuffer[float_dtype]
    var rz_old: DeviceBuffer[float_dtype]
    var rz_new: DeviceBuffer[float_dtype]
    var pAp: DeviceBuffer[float_dtype]
    var cg_alpha: DeviceBuffer[float_dtype]  # Renamed to avoid confusion with solution alpha
    var cg_beta: DeviceBuffer[float_dtype]
    var residual_norms_sq: DeviceBuffer[float_dtype]
    var max_residual: DeviceBuffer[float_dtype]
    var max_residual_host: HostBuffer[float_dtype]
    
    # Tridiagonal tracking buffers
    var all_alphas: DeviceBuffer[float_dtype]
    var all_betas: DeviceBuffer[float_dtype]
    var all_alphas_host: HostBuffer[float_dtype]
    var all_betas_host: HostBuffer[float_dtype]
    
    # -------------------------------------------------------------------------
    # BBMM-level buffers (used in bbmm_compute_nll_and_gradients)
    # -------------------------------------------------------------------------
    # RHS and probe buffers
    var rhs_device: DeviceBuffer[float_dtype]
    var rhs_host: HostBuffer[float_dtype]
    var y_host: HostBuffer[float_dtype]
    var y_device_buf: DeviceBuffer[float_dtype]
    var probes_host: HostBuffer[float_dtype]
    var grad_probes_host: HostBuffer[float_dtype]
    var grad_probes_device: DeviceBuffer[float_dtype]
    var probes_device: DeviceBuffer[float_dtype]
    var probe_norms: DeviceBuffer[float_dtype]
    var probes_host_buf: HostBuffer[float_dtype]
    var probe_norms_host: HostBuffer[float_dtype]
    var rhs_cg: DeviceBuffer[float_dtype]
    var rhs_norms: DeviceBuffer[float_dtype]  # For CG RHS normalization optimization
    
    # inv_quad computation buffers
    var inv_quad_device: DeviceBuffer[float_dtype]
    var inv_quad_host: HostBuffer[float_dtype]
    
    # Gradient computation buffers
    var alpha_device: DeviceBuffer[float_dtype]  # K^{-1} @ y
    var probe_solutions_device: DeviceBuffer[float_dtype]  # K^{-1} @ Z_rad
    var dK_alpha_device: DeviceBuffer[float_dtype]
    var dK_Z_device: DeviceBuffer[float_dtype]
    var K_unscaled_alpha_device: DeviceBuffer[float_dtype]
    var K_unscaled_Z_device: DeviceBuffer[float_dtype]
    var alpha_dK_alpha_device: DeviceBuffer[float_dtype]
    var trace_ls_device: DeviceBuffer[float_dtype]
    var alpha_norm_sq_device: DeviceBuffer[float_dtype]
    var trace_noise_device: DeviceBuffer[float_dtype]
    var alpha_K_unscaled_alpha_device: DeviceBuffer[float_dtype]
    var trace_os_device: DeviceBuffer[float_dtype]
    var all_scalars_device: DeviceBuffer[float_dtype]
    var all_scalars_host: HostBuffer[float_dtype]
    
    # -------------------------------------------------------------------------
    # sample_from_preconditioner_gpu buffers (Optimization #5b)
    # -------------------------------------------------------------------------
    var z_rank_host: HostBuffer[float_dtype]       # [rank × num_probes] Gaussian samples
    var z_rank_device: DeviceBuffer[float_dtype]   # [rank × num_probes] on GPU
    var z_noise_host: HostBuffer[float_dtype]      # [n × num_probes] noise samples
    var z_noise_device: DeviceBuffer[float_dtype]  # [n × num_probes] on GPU
    var sampled_probes_device: DeviceBuffer[float_dtype]  # [n × num_probes] output
    
    # -------------------------------------------------------------------------
    # ARD gradient computation buffers
    # -------------------------------------------------------------------------
    # For ARD, we compute d gradients (one per dimension) instead of 1 isotropic gradient.
    # These buffers store per-dimension gradient results.
    var trace_ls_ard_device: DeviceBuffer[float_dtype]  # [d × num_probes] per-dim trace values
    var alpha_dK_alpha_ard_device: DeviceBuffer[float_dtype]  # [d] per-dim data terms
    var all_scalars_ard_device: DeviceBuffer[float_dtype]  # [d + d*num_probes] combined ARD scalars
    var all_scalars_ard_host: HostBuffer[float_dtype]  # Host copy
    
    # -------------------------------------------------------------------------
    # Generic gradient computation buffers (for composite kernels with N params)
    # -------------------------------------------------------------------------
    # These buffers are reused for each parameter in sequence (loop-and-reuse strategy).
    # This trades parallelism for simplicity and memory efficiency.
    var dK_param_alpha_device: DeviceBuffer[float_dtype]  # [n] for dK/dθ @ alpha
    var dK_param_Z_device: DeviceBuffer[float_dtype]      # [n * num_probes] for dK/dθ @ Z
    var os_alpha_device: DeviceBuffer[float_dtype]         # [n] for fused ls+os alpha gradient
    var os_probes_device: DeviceBuffer[float_dtype]        # [n * num_probes] for fused ls+os probe gradient
    var p1_alpha_device: DeviceBuffer[float_dtype]         # [n] for fused 3-param alpha gradient (param1)
    var p1_probes_device: DeviceBuffer[float_dtype]        # [n * num_probes] for fused 3-param probe gradient (param1)
    var param_data_term_device: DeviceBuffer[float_dtype] # [1] scalar per param
    var param_trace_device: DeviceBuffer[float_dtype]     # [num_probes] traces per param
    var param_trace_host: HostBuffer[float_dtype]         # Host copy for reduction
    var param_data_term_host: HostBuffer[float_dtype]     # Host copy for reduction
    
    # -------------------------------------------------------------------------
    # Fused gradient computation buffers (for ARD optimization)
    # -------------------------------------------------------------------------
    # These buffers enable computing forward + all gradients in a single kernel.
    # Used when num_params > 3 (typically ARD with d >= 3).
    var v_combined_device: DeviceBuffer[float_dtype]       # [n * (1 + num_probes)] alpha + probes
    var forward_out_device: DeviceBuffer[float_dtype]      # [n * (1 + num_probes)] forward matvec output
    var gradient_out_device: DeviceBuffer[float_dtype]     # [n * num_params * (1 + num_probes)] all gradients
    var gradient_dots_device: DeviceBuffer[float_dtype]    # [num_params * (1 + num_probes)] batched dots
    var gradient_dots_host: HostBuffer[float_dtype]        # Host copy
    var capacity_num_params_fused: Int  # Capacity for fused gradient buffers
    
    # -------------------------------------------------------------------------
    # Capacity tracking
    # -------------------------------------------------------------------------
    var capacity_n: Int
    var capacity_num_cols: Int
    var capacity_num_probes: Int
    var capacity_max_tridiag_iter: Int
    var capacity_rank: Int  # Preconditioner rank for sample_from_preconditioner buffers
    var capacity_dim: Int  # Input dimension for ARD buffers
    var capacity_num_kernel_params: Int  # Number of kernel parameters for generic buffers
    
    fn __init__(out self, ctx: DeviceContext, n: Int = 1, num_cols: Int = 1, num_probes: Int = 1, max_tridiag_iter: Int = 1) raises:
        """Create a buffer pool with initial capacity.
        
        Args:
            ctx: GPU device context
            n: Initial number of data points (default: 1)
            num_cols: Initial number of RHS columns (default: 1)
            num_probes: Initial number of probe vectors (default: 1)
            max_tridiag_iter: Initial max tridiagonal iterations (default: 1)
        """
        # Initialize capacity tracking
        self.capacity_n = n
        self.capacity_num_cols = num_cols
        self.capacity_num_probes = num_probes
        self.capacity_max_tridiag_iter = max_tridiag_iter
        
        # Initialize all CG-level buffers
        self.x = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
        self.r = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
        self.z = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
        self.p = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
        self.Ap = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
        self.rz_old = ctx.enqueue_create_buffer[float_dtype](num_cols)
        self.rz_new = ctx.enqueue_create_buffer[float_dtype](num_cols)
        self.pAp = ctx.enqueue_create_buffer[float_dtype](num_cols)
        self.cg_alpha = ctx.enqueue_create_buffer[float_dtype](num_cols)
        self.cg_beta = ctx.enqueue_create_buffer[float_dtype](num_cols)
        self.residual_norms_sq = ctx.enqueue_create_buffer[float_dtype](num_cols)
        self.max_residual = ctx.enqueue_create_buffer[float_dtype](1)
        self.max_residual_host = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        # Tridiagonal tracking buffers
        self.all_alphas = ctx.enqueue_create_buffer[float_dtype](max_tridiag_iter * num_cols)
        self.all_betas = ctx.enqueue_create_buffer[float_dtype](max_tridiag_iter * num_cols)
        self.all_alphas_host = ctx.enqueue_create_host_buffer[float_dtype](max_tridiag_iter * num_cols)
        self.all_betas_host = ctx.enqueue_create_host_buffer[float_dtype](max_tridiag_iter * num_cols)
        
        # Initialize all BBMM-level buffers
        self.rhs_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
        self.rhs_host = ctx.enqueue_create_host_buffer[float_dtype](n * num_cols)
        self.y_host = ctx.enqueue_create_host_buffer[float_dtype](n)
        self.y_device_buf = ctx.enqueue_create_buffer[float_dtype](n)
        self.probes_host = ctx.enqueue_create_host_buffer[float_dtype](n * num_probes)
        self.grad_probes_host = ctx.enqueue_create_host_buffer[float_dtype](n * num_probes)
        self.grad_probes_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        self.probes_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        self.probe_norms = ctx.enqueue_create_buffer[float_dtype](num_probes)
        self.probes_host_buf = ctx.enqueue_create_host_buffer[float_dtype](n * num_probes)
        self.probe_norms_host = ctx.enqueue_create_host_buffer[float_dtype](num_probes)
        self.rhs_cg = ctx.enqueue_create_buffer[float_dtype](n * num_cols)
        self.rhs_norms = ctx.enqueue_create_buffer[float_dtype](num_cols)  # For CG RHS normalization
        
        # inv_quad computation buffers
        self.inv_quad_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.inv_quad_host = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        # Gradient computation buffers
        self.alpha_device = ctx.enqueue_create_buffer[float_dtype](n)
        self.probe_solutions_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        self.dK_alpha_device = ctx.enqueue_create_buffer[float_dtype](n)
        self.dK_Z_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        self.K_unscaled_alpha_device = ctx.enqueue_create_buffer[float_dtype](n)
        self.K_unscaled_Z_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        self.alpha_dK_alpha_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.trace_ls_device = ctx.enqueue_create_buffer[float_dtype](num_probes)
        self.alpha_norm_sq_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.trace_noise_device = ctx.enqueue_create_buffer[float_dtype](num_probes)
        self.alpha_K_unscaled_alpha_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.trace_os_device = ctx.enqueue_create_buffer[float_dtype](num_probes)
        
        # Combined scalar output buffer
        var num_scalars = 3 + 3 * num_probes
        self.all_scalars_device = ctx.enqueue_create_buffer[float_dtype](num_scalars)
        self.all_scalars_host = ctx.enqueue_create_host_buffer[float_dtype](num_scalars)
        
        # sample_from_preconditioner_gpu buffers (Optimization #5b)
        # Default rank=10 for initial allocation
        var default_rank = 10
        self.capacity_rank = default_rank
        self.z_rank_host = ctx.enqueue_create_host_buffer[float_dtype](default_rank * num_probes)
        self.z_rank_device = ctx.enqueue_create_buffer[float_dtype](default_rank * num_probes)
        self.z_noise_host = ctx.enqueue_create_host_buffer[float_dtype](n * num_probes)
        self.z_noise_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        self.sampled_probes_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        
        # ARD gradient computation buffers (default dim=1)
        var default_dim = 1
        self.capacity_dim = default_dim
        self.trace_ls_ard_device = ctx.enqueue_create_buffer[float_dtype](default_dim * num_probes)
        self.alpha_dK_alpha_ard_device = ctx.enqueue_create_buffer[float_dtype](default_dim)
        var num_ard_scalars = default_dim + default_dim * num_probes
        self.all_scalars_ard_device = ctx.enqueue_create_buffer[float_dtype](num_ard_scalars)
        self.all_scalars_ard_host = ctx.enqueue_create_host_buffer[float_dtype](num_ard_scalars)
        
        # Generic gradient computation buffers (for composite kernels)
        # These are reused for each parameter in sequence
        self.capacity_num_kernel_params = 1  # Default, will be resized as needed
        self.dK_param_alpha_device = ctx.enqueue_create_buffer[float_dtype](n)
        self.dK_param_Z_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        self.os_alpha_device = ctx.enqueue_create_buffer[float_dtype](n)
        self.os_probes_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        self.p1_alpha_device = ctx.enqueue_create_buffer[float_dtype](n)
        self.p1_probes_device = ctx.enqueue_create_buffer[float_dtype](n * num_probes)
        self.param_data_term_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.param_trace_device = ctx.enqueue_create_buffer[float_dtype](num_probes)
        self.param_trace_host = ctx.enqueue_create_host_buffer[float_dtype](num_probes)
        self.param_data_term_host = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        # Fused gradient computation buffers (for ARD optimization)
        # Default: num_params = 1, will be resized for ARD
        self.capacity_num_params_fused = 1
        var num_cols_fused = 1 + num_probes  # alpha + probes
        self.v_combined_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols_fused)
        self.forward_out_device = ctx.enqueue_create_buffer[float_dtype](n * num_cols_fused)
        self.gradient_out_device = ctx.enqueue_create_buffer[float_dtype](n * 1 * num_cols_fused)
        self.gradient_dots_device = ctx.enqueue_create_buffer[float_dtype](1 * num_cols_fused)
        self.gradient_dots_host = ctx.enqueue_create_host_buffer[float_dtype](1 * num_cols_fused)
    
    fn ensure_capacity(mut self, ctx: DeviceContext, n: Int, num_cols: Int, num_probes: Int, max_tridiag_iter: Int, rank: Int = 10, dim: Int = 1, num_kernel_params: Int = 1) raises:
        """Ensure all buffers have sufficient capacity, reallocating if needed.
        
        Adds 20% headroom to avoid frequent reallocations when problem size
        increases slightly.
        
        Args:
            ctx: GPU device context
            n: Number of data points
            num_cols: Total number of RHS columns (1 + num_probes + num_grad_probes)
            num_probes: Number of probe vectors for log-det and gradients
            max_tridiag_iter: Maximum tridiagonal iterations to track
            rank: Preconditioner rank
            dim: Input dimension (for ARD buffers)
            num_kernel_params: Number of kernel parameters (for generic composite buffers)
        """
        # Check if current capacity is sufficient (including rank, dim, and num_kernel_params)
        if n <= self.capacity_n and num_cols <= self.capacity_num_cols and num_probes <= self.capacity_num_probes and max_tridiag_iter <= self.capacity_max_tridiag_iter and rank <= self.capacity_rank and dim <= self.capacity_dim and num_kernel_params <= self.capacity_num_kernel_params:
            return  # Reuse existing buffers
        
        # Allocate with headroom to avoid frequent reallocations
        var new_n = n * 12 // 10  # 20% headroom
        var new_cols = num_cols * 12 // 10
        var new_probes = num_probes * 12 // 10
        var new_tridiag = max_tridiag_iter * 12 // 10
        if new_n < n:
            new_n = n  # Handle overflow
        if new_cols < num_cols:
            new_cols = num_cols
        if new_probes < num_probes:
            new_probes = num_probes
        if new_tridiag < max_tridiag_iter:
            new_tridiag = max_tridiag_iter
        
        # =====================================================================
        # CG-level buffers
        # =====================================================================
        self.x = ctx.enqueue_create_buffer[float_dtype](new_n * new_cols)
        self.r = ctx.enqueue_create_buffer[float_dtype](new_n * new_cols)
        self.z = ctx.enqueue_create_buffer[float_dtype](new_n * new_cols)
        self.p = ctx.enqueue_create_buffer[float_dtype](new_n * new_cols)
        self.Ap = ctx.enqueue_create_buffer[float_dtype](new_n * new_cols)
        self.rz_old = ctx.enqueue_create_buffer[float_dtype](new_cols)
        self.rz_new = ctx.enqueue_create_buffer[float_dtype](new_cols)
        self.pAp = ctx.enqueue_create_buffer[float_dtype](new_cols)
        self.cg_alpha = ctx.enqueue_create_buffer[float_dtype](new_cols)
        self.cg_beta = ctx.enqueue_create_buffer[float_dtype](new_cols)
        self.residual_norms_sq = ctx.enqueue_create_buffer[float_dtype](new_cols)
        self.max_residual = ctx.enqueue_create_buffer[float_dtype](1)
        self.max_residual_host = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        # Tridiagonal tracking buffers
        self.all_alphas = ctx.enqueue_create_buffer[float_dtype](new_tridiag * new_cols)
        self.all_betas = ctx.enqueue_create_buffer[float_dtype](new_tridiag * new_cols)
        self.all_alphas_host = ctx.enqueue_create_host_buffer[float_dtype](new_tridiag * new_cols)
        self.all_betas_host = ctx.enqueue_create_host_buffer[float_dtype](new_tridiag * new_cols)
        
        # =====================================================================
        # BBMM-level buffers
        # =====================================================================
        # RHS and probe buffers
        self.rhs_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_cols)
        self.rhs_host = ctx.enqueue_create_host_buffer[float_dtype](new_n * new_cols)
        self.y_host = ctx.enqueue_create_host_buffer[float_dtype](new_n)
        self.y_device_buf = ctx.enqueue_create_buffer[float_dtype](new_n)
        self.probes_host = ctx.enqueue_create_host_buffer[float_dtype](new_n * new_probes)
        self.grad_probes_host = ctx.enqueue_create_host_buffer[float_dtype](new_n * new_probes)
        self.grad_probes_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        self.probes_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        self.probe_norms = ctx.enqueue_create_buffer[float_dtype](new_probes)
        self.probes_host_buf = ctx.enqueue_create_host_buffer[float_dtype](new_n * new_probes)
        self.probe_norms_host = ctx.enqueue_create_host_buffer[float_dtype](new_probes)
        self.rhs_cg = ctx.enqueue_create_buffer[float_dtype](new_n * new_cols)
        self.rhs_norms = ctx.enqueue_create_buffer[float_dtype](new_cols)  # For CG RHS normalization
        
        # inv_quad computation buffers
        self.inv_quad_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.inv_quad_host = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        # Gradient computation buffers
        self.alpha_device = ctx.enqueue_create_buffer[float_dtype](new_n)
        self.probe_solutions_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        self.dK_alpha_device = ctx.enqueue_create_buffer[float_dtype](new_n)
        self.dK_Z_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        self.K_unscaled_alpha_device = ctx.enqueue_create_buffer[float_dtype](new_n)
        self.K_unscaled_Z_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        self.alpha_dK_alpha_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.trace_ls_device = ctx.enqueue_create_buffer[float_dtype](new_probes)
        self.alpha_norm_sq_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.trace_noise_device = ctx.enqueue_create_buffer[float_dtype](new_probes)
        self.alpha_K_unscaled_alpha_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.trace_os_device = ctx.enqueue_create_buffer[float_dtype](new_probes)
        
        # Combined scalar output buffer: 3 scalars + 3*num_probes trace values
        var num_scalars = 3 + 3 * new_probes
        self.all_scalars_device = ctx.enqueue_create_buffer[float_dtype](num_scalars)
        self.all_scalars_host = ctx.enqueue_create_host_buffer[float_dtype](num_scalars)
        
        # =====================================================================
        # sample_from_preconditioner_gpu buffers (Optimization #5b)
        # =====================================================================
        var new_rank = rank * 12 // 10  # 20% headroom
        if new_rank < rank:
            new_rank = rank  # Handle overflow
        self.z_rank_host = ctx.enqueue_create_host_buffer[float_dtype](new_rank * new_probes)
        self.z_rank_device = ctx.enqueue_create_buffer[float_dtype](new_rank * new_probes)
        self.z_noise_host = ctx.enqueue_create_host_buffer[float_dtype](new_n * new_probes)
        self.z_noise_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        self.sampled_probes_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        
        # =====================================================================
        # ARD gradient computation buffers
        # =====================================================================
        var new_dim = dim * 12 // 10  # 20% headroom
        if new_dim < dim:
            new_dim = dim  # Handle overflow
        self.trace_ls_ard_device = ctx.enqueue_create_buffer[float_dtype](new_dim * new_probes)
        self.alpha_dK_alpha_ard_device = ctx.enqueue_create_buffer[float_dtype](new_dim)
        var num_ard_scalars = new_dim + new_dim * new_probes
        self.all_scalars_ard_device = ctx.enqueue_create_buffer[float_dtype](num_ard_scalars)
        self.all_scalars_ard_host = ctx.enqueue_create_host_buffer[float_dtype](num_ard_scalars)
        
        # =====================================================================
        # Generic gradient computation buffers (for composite kernels)
        # =====================================================================
        # These are reused for each parameter in sequence (loop-and-reuse strategy)
        self.dK_param_alpha_device = ctx.enqueue_create_buffer[float_dtype](new_n)
        self.os_alpha_device = ctx.enqueue_create_buffer[float_dtype](new_n)
        self.p1_alpha_device = ctx.enqueue_create_buffer[float_dtype](new_n)
        self.dK_param_Z_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        self.os_probes_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        self.p1_probes_device = ctx.enqueue_create_buffer[float_dtype](new_n * new_probes)
        self.param_data_term_device = ctx.enqueue_create_buffer[float_dtype](1)
        self.param_trace_device = ctx.enqueue_create_buffer[float_dtype](new_probes)
        self.param_trace_host = ctx.enqueue_create_host_buffer[float_dtype](new_probes)
        self.param_data_term_host = ctx.enqueue_create_host_buffer[float_dtype](1)
        
        # =====================================================================
        # Fused gradient computation buffers (for ARD optimization)
        # =====================================================================
        var num_cols_fused = 1 + new_probes  # alpha + probes
        self.v_combined_device = ctx.enqueue_create_buffer[float_dtype](new_n * num_cols_fused)
        self.forward_out_device = ctx.enqueue_create_buffer[float_dtype](new_n * num_cols_fused)
        self.gradient_out_device = ctx.enqueue_create_buffer[float_dtype](new_n * num_kernel_params * num_cols_fused)
        self.gradient_dots_device = ctx.enqueue_create_buffer[float_dtype](num_kernel_params * num_cols_fused)
        self.gradient_dots_host = ctx.enqueue_create_host_buffer[float_dtype](num_kernel_params * num_cols_fused)
        self.capacity_num_params_fused = num_kernel_params
        
        # Update capacity tracking
        self.capacity_n = new_n
        self.capacity_num_cols = new_cols
        self.capacity_num_probes = new_probes
        self.capacity_max_tridiag_iter = new_tridiag
        self.capacity_rank = new_rank
        self.capacity_dim = new_dim
        self.capacity_num_kernel_params = num_kernel_params


# =============================================================================
# Preconditioner Type
# =============================================================================

@fieldwise_init
struct BBMMPrecondType(ImplicitlyCopyable):
    """Preconditioner type for BBMM solver.
    
    Options:
    - PIVOTED_CHOLESKY: Low-rank Pivoted Cholesky (the only production preconditioner)
    """
    var _value: Int
    
    comptime PIVOTED_CHOLESKY = BBMMPrecondType(0)
    
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value
    
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# =============================================================================
# Batched CG with Tridiagonal Tracking
# =============================================================================

struct CGResultWithTridiag(Copyable):
    """CG result with tridiagonal matrices."""
    var solution: DeviceBuffer[float_dtype]
    var num_iterations: Int
    var converged: Bool
    var tridiag_diag: List[List[Float32]]  # Diagonal elements for each column
    var tridiag_offdiag: List[List[Float32]]  # Off-diagonal elements for each column
    var tridiag_size: Int  # Actual size of tridiagonals
    
    fn __init__(out self, var solution: DeviceBuffer[float_dtype], num_iterations: Int, converged: Bool,
                var tridiag_diag: List[List[Float32]], var tridiag_offdiag: List[List[Float32]], tridiag_size: Int):
        self.solution = solution^
        self.num_iterations = num_iterations
        self.converged = converged
        self.tridiag_diag = tridiag_diag^
        self.tridiag_offdiag = tridiag_offdiag^
        self.tridiag_size = tridiag_size


# =============================================================================
# Unified BBMM Result Struct
# =============================================================================

struct UnifiedBBMMResult(Copyable, Movable):
    """Result from unified BBMM inference.
    
    Contains NLL, all gradients (kernel params + noise), and CG solution.
    The gradients list has length num_gradient_params + 1, where the last
    element is always the noise gradient.
    
    right_factors = P^{-1} @ z (unnormalized probes through preconditioner).
    Needed for computing custom gradients (e.g., B and per-task noise gradients
    in Kronecker multi-output training).
    """
    var inv_quad: Float32
    var log_det: Float32
    var nll: Float32
    var solution: DeviceBuffer[float_dtype]         # K^{-1} @ y  [n]
    var probe_solutions: DeviceBuffer[float_dtype]  # K^{-1} @ Z  [n * num_probes]
    var right_factors: DeviceBuffer[float_dtype]    # P^{-1} @ Z  [n * num_probes]
    var gradients: List[Float32]                    # [grad_param_0, ..., grad_param_{N-1}, grad_noise]
    var num_iterations: Int
    var converged: Bool
    var log_det_std: Float32
    var num_gradient_params: Int                    # N (excluding noise)
    
    fn __init__(
        out self,
        inv_quad: Float32,
        log_det: Float32,
        nll: Float32,
        var solution: DeviceBuffer[float_dtype],
        var probe_solutions: DeviceBuffer[float_dtype],
        var right_factors: DeviceBuffer[float_dtype],
        var gradients: List[Float32],
        num_iterations: Int,
        converged: Bool,
        log_det_std: Float32,
        num_gradient_params: Int,
    ):
        self.inv_quad = inv_quad
        self.log_det = log_det
        self.nll = nll
        self.solution = solution^
        self.probe_solutions = probe_solutions^
        self.right_factors = right_factors^
        self.gradients = gradients^
        self.num_iterations = num_iterations
        self.converged = converged
        self.log_det_std = log_det_std
        self.num_gradient_params = num_gradient_params

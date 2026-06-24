"""GPU kernels for Kronecker multi-output operations.

Extracted from kronecker_direct_provider.mojo for reuse by
FusedKroneckerProvider and other multi-output paths.
"""

from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_dim, block_idx, thread_idx
from memory import UnsafePointer


fn kernel_reshuffle_to_flat(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],   # [n * T * num_cols]
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],     # [nT * num_cols] Kronecker layout
    n: Int, T: Int, num_cols: Int,
) -> None:
    """Reshuffle from Kronecker layout [col*nT + task*n + i] to flat [batch_col*n + i].
    
    batch_col = task * num_cols + col
    One thread per output element.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var total = n * T * num_cols
    if idx >= total:
        return
    # Decode: idx = batch_col * n + i, batch_col = task * num_cols + col
    var i = idx % n
    var batch_col = idx // n
    var task = batch_col // num_cols
    var col = batch_col - task * num_cols
    # Read from Kronecker layout
    out_ptr[idx] = v_ptr[col * n * T + task * n + i]


fn kernel_kronecker_combine_batched(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_v_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n * T * num_cols] flat layout
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],      # [nT * num_cols] Kronecker layout
    B_ptr: UnsafePointer[Float32, MutAnyOrigin],      # [T * T]
    noise_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [T]
    n: Int, T: Int, num_cols: Int,
    outputscale: Float32,
) -> None:
    """Combine K_X @ v results with B matrix and noise for ALL CG columns.
    
    For each task s, column c, data point i:
        out[c*nT + s*n + i] = os * sum_t(B[s,t] * kx_v[(t*num_cols+c)*n + i]) + noise[s] * v[c*nT + s*n + i]
    
    kx_v_ptr is in flat layout: [(task*num_cols+col)*n + i]
    v_ptr is in Kronecker layout: [col*nT + task*n + i]
    out_ptr is in Kronecker layout: [col*nT + task*n + i]
    
    Grid: one thread per (s, i) pair, loops over columns.
    """
    var si = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nT = n * T
    
    if si >= nT:
        return
    
    var i = si % n       # data point index
    var s = si // n      # task index
    
    var noise_s = noise_ptr[s]
    
    for c in range(num_cols):
        # Compute outputscale * sum_t(B[s,t] * kx_v[(t*num_cols+c)*n + i])
        var val = Float32(0.0)
        for t in range(T):
            val += B_ptr[s * T + t] * kx_v_ptr[(t * num_cols + c) * n + i]
        val *= outputscale
        
        # Add noise: noise[s] * v[c*nT + s*n + i]
        var v_idx = c * nT + s * n + i
        val += noise_s * v_ptr[v_idx]
        
        out_ptr[v_idx] = val


fn kernel_kronecker_combine_batched_vector_noise(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    kx_v_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [n * T * num_cols]
    v_ptr: UnsafePointer[Float32, MutAnyOrigin],      # [nT * num_cols]
    B_ptr: UnsafePointer[Float32, MutAnyOrigin],      # [T * T]
    noise_vec_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [T * n]
    n: Int, T: Int, num_cols: Int,
    outputscale: Float32,
) -> None:
    """Combine Kronecker matvec with fixed per-sample-task diagonal noise.

    noise_vec_ptr uses task-blocked layout: noise_vec[task * n + i].
    """
    var si = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nT = n * T

    if si >= nT:
        return

    var i = si % n
    var s = si // n
    var noise_si = noise_vec_ptr[s * n + i]

    for c in range(num_cols):
        var val = Float32(0.0)
        for t in range(T):
            val += B_ptr[s * T + t] * kx_v_ptr[(t * num_cols + c) * n + i]
        val *= outputscale

        var v_idx = c * nT + s * n + i
        val += noise_si * v_ptr[v_idx]

        out_ptr[v_idx] = val

"""Preconditioner Trait for Generic CG Solvers and BBMM.

This module defines the Preconditioner trait that allows both batched_cg_unified
and bbmm_with_precond to work with different preconditioner types:
- PivotedCholeskyPrecond: Standard single-output preconditioner (P = L L^T + noise I)
- KroneckerPreconditioner: Multi-output Kronecker preconditioner (P = os * (B ⊗ L L^T) + D)

Methods:
- apply_precond: P^{-1} @ v (used in CG loop)
- sample_probes: Sample from N(0, P) (used in BBMM for SLQ probes)
- log_det: Compute log|P| (used in BBMM for log-det correction)
"""

from gpu.host import DeviceContext, DeviceBuffer
from memory import UnsafePointer

alias float_dtype = DType.float32


trait Preconditioner:
    """Trait for preconditioners used in BBMM and batched CG solvers.
    
    A preconditioner approximates K and provides:
    1. P^{-1} @ v — applied per CG iteration
    2. Samples from N(0, P) — used as SLQ probe vectors
    3. log|P| — correction for preconditioned log-det estimation
    """
    
    fn apply_precond(
        self,
        ctx: DeviceContext,
        v_ptr: UnsafePointer[Float32, MutAnyOrigin],
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        n: Int,
        num_cols: Int,
        sync: Bool,
    ) raises:
        """Apply preconditioner inverse: out = P^{-1} @ v.
        
        Args:
            ctx: GPU device context.
            v_ptr: Input vectors [n x num_cols] column-major ON DEVICE.
            out_ptr: Output vectors [n x num_cols] column-major ON DEVICE.
            n: Number of rows (problem dimension).
            num_cols: Number of columns (batch size).
            sync: Whether to synchronize after kernel launches.
        """
        ...
    
    fn sample_probes(
        self,
        ctx: DeviceContext,
        out_device: DeviceBuffer[float_dtype],
        num_probes: Int,
        seed_val: UInt64,
    ) raises:
        """Sample probe vectors from N(0, P).
        
        Used for SLQ log-det estimation. Each probe column is a sample from
        the multivariate normal N(0, P) where P is the preconditioner.
        
        Args:
            ctx: GPU device context.
            out_device: Output buffer [n x num_probes] column-major ON DEVICE.
            num_probes: Number of probe vectors to generate.
            seed_val: Random seed for reproducibility.
        """
        ...
    
    fn log_det(self, ctx: DeviceContext) raises -> Float32:
        """Compute log|P| for log-det correction.
        
        When using preconditioned CG, the tridiagonals give eigenvalues of
        P^{-1}K, not K. The correction is: log|K| = log|P^{-1}K| + log|P|.
        
        Args:
            ctx: GPU device context.
            
        Returns:
            log|P| (the log-determinant of the preconditioner).
        """
        ...

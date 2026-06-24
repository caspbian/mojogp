from mojogp.kernels.native_numerics import (
    cholesky_decompose,
    lu_decompose,
    matrix_inv_native,
    compute_slogdet_native,
    tridiagonal_eigh_native
)
from memory import UnsafePointer
from math import abs

fn test_cholesky() raises:
    var n = 3
    var A = UnsafePointer[Float32].alloc(n * n)
    # SPD matrix
    # [4, 12, -16]
    # [12, 37, -43]
    # [-16, -43, 98]
    A[0] = 4.0; A[1] = 12.0; A[2] = -16.0
    A[3] = 12.0; A[4] = 37.0; A[5] = -43.0
    A[6] = -16.0; A[7] = -43.0; A[8] = 98.0

    var L = UnsafePointer[Float32].alloc(n * n)
    var is_spd = cholesky_decompose(A, n, L)

    if not is_spd:
        raise Error("Cholesky failed on SPD matrix")

    # Expected L:
    # [2, 0, 0]
    # [6, 1, 0]
    # [-8, 5, 3]
    if abs(L[0] - 2.0) > 1e-5 or abs(L[3] - 6.0) > 1e-5 or abs(L[4] - 1.0) > 1e-5 or abs(L[6] - -8.0) > 1e-5 or abs(L[7] - 5.0) > 1e-5 or abs(L[8] - 3.0) > 1e-5:
        raise Error("Cholesky output incorrect")

    print("Cholesky test passed")
    A.free()
    L.free()

fn test_lu() raises:
    var n = 3
    var A = UnsafePointer[Float32].alloc(n * n)
    # [1, 2, 3]
    # [2, -4, 6]
    # [3, -9, -3]
    A[0] = 1.0; A[1] = 2.0; A[2] = 3.0
    A[3] = 2.0; A[4] = -4.0; A[5] = 6.0
    A[6] = 3.0; A[7] = -9.0; A[8] = -3.0

    var LU = UnsafePointer[Float32].alloc(n * n)
    var P = UnsafePointer[Int].alloc(n)
    var swaps = lu_decompose(A, n, LU, P)

    if swaps == -1:
        raise Error("LU failed on invertible matrix")

    print("LU test passed")
    A.free()
    LU.free()
    P.free()

fn test_inv() raises:
    var n = 2
    var A = UnsafePointer[Float32].alloc(n * n)
    # [4, 7]
    # [2, 6]
    A[0] = 4.0; A[1] = 7.0
    A[2] = 2.0; A[3] = 6.0

    var invA = UnsafePointer[Float32].alloc(n * n)
    matrix_inv_native(A, n, invA)

    # Expected invA:
    # [0.6, -0.7]
    # [-0.2, 0.4]
    if abs(invA[0] - 0.6) > 1e-5 or abs(invA[1] - -0.7) > 1e-5 or abs(invA[2] - -0.2) > 1e-5 or abs(invA[3] - 0.4) > 1e-5:
        raise Error("Matrix inversion incorrect")

    print("Matrix inversion test passed")
    A.free()
    invA.free()

fn test_eigh() raises:
    var m = 3
    var diag = UnsafePointer[Float32].alloc(m)
    var offdiag = UnsafePointer[Float32].alloc(m - 1)

    # Tridiagonal matrix:
    # [2, -1, 0]
    # [-1, 2, -1]
    # [0, -1, 2]
    diag[0] = 2.0; diag[1] = 2.0; diag[2] = 2.0
    offdiag[0] = -1.0; offdiag[1] = -1.0

    var evals = UnsafePointer[Float32].alloc(m)
    var evecs = UnsafePointer[Float32].alloc(m * m)

    tridiagonal_eigh_native(diag, offdiag, m, evals, evecs)

    # Expected eigenvalues: 2 - sqrt(2), 2, 2 + sqrt(2)
    # approx: 0.585786, 2.0, 3.414213
    if abs(evals[0] - 0.585786) > 1e-4 or abs(evals[1] - 2.0) > 1e-4 or abs(evals[2] - 3.414213) > 1e-4:
        raise Error("Eigendecomposition incorrect")

    print("Eigendecomposition test passed")
    diag.free()
    offdiag.free()
    evals.free()
    evecs.free()

fn main() raises:
    test_cholesky()
    test_lu()
    test_inv()
    test_eigh()

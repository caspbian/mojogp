"""
Shared pytest fixtures for MojoGP unit tests.
"""

import gc
from pathlib import Path

import pytest
import numpy as np
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _cleanup_gpu_memory():
    """Auto-cleanup GPU memory between tests to prevent cumulative OOM.

    The Mojo DeviceContext and PyTorch CUDA allocators cache freed GPU memory
    for performance. Over hundreds of tests, this cached memory accumulates
    and eventually exhausts the GPU (especially on 6GB cards). This fixture
    forces Python garbage collection and clears the PyTorch CUDA cache after
    each test to reclaim GPU memory.
    """
    yield
    # After each test: force GC to trigger Mojo object destructors, then
    # release PyTorch's cached CUDA memory back to the driver.
    gc.collect()
    if torch.cuda.is_available():
        # After a CUDA OOM, one cleanup call can fail while later ones would
        # still succeed. Keep cleanup best-effort so one poisoned call does not
        # prevent the rest of the teardown from running.
        for cuda_cleanup in (
            torch.cuda.synchronize,
            torch.cuda.empty_cache,
            torch.cuda.ipc_collect,
        ):
            try:
                cuda_cleanup()
            except Exception:
                pass


@pytest.fixture
def random_seed():
    """Fixed seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture
def gpytorch_available():
    """Check if GPyTorch is available."""
    try:
        import gpytorch

        return True
    except ImportError:
        return False


@pytest.fixture
def linear_operator_available():
    """Check if linear_operator is available."""
    try:
        import linear_operator

        return True
    except ImportError:
        return False


@pytest.fixture
def small_dataset(random_seed):
    """Generate a small dataset for testing."""
    n = 100
    d = 5
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    return X, y


@pytest.fixture
def medium_dataset(random_seed):
    """Generate a medium dataset for testing."""
    n = 500
    d = 5
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    return X, y


@pytest.fixture
def rbf_kernel_matrix(small_dataset):
    """Generate RBF kernel matrix for testing."""
    from scipy.spatial.distance import cdist

    X, _ = small_dataset
    lengthscale = 1.0
    outputscale = 1.0
    noise = 0.01

    dist_sq = cdist(X, X, metric="sqeuclidean")
    K = outputscale * np.exp(-dist_sq / (2 * lengthscale**2))
    K_noisy = K + noise * np.eye(len(X))

    return K, K_noisy, lengthscale, outputscale, noise


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "slow: marks tests as slow (>5s)")
    config.addinivalue_line("markers", "gpytorch: marks tests requiring GPyTorch")
    config.addinivalue_line(
        "markers",
        "reference: marks correctness tests that compare against an oracle or trusted external reference",
    )


def pytest_addoption(parser):
    """Add shared test command-line options."""
    try:
        parser.addoption(
            "--n-override",
            type=int,
            default=None,
            help=(
                "Override n (dataset size) for test configs that accept an "
                "n_override fixture."
            ),
        )
    except ValueError:
        # Some benchmark conftests also register this option when run directly.
        pass


@pytest.fixture(scope="session")
def n_override(request):
    """Optional dataset-size override for system and benchmark tests."""
    return request.config.getoption("--n-override", default=None)


def pytest_collection_modifyitems(config, items):
    """Run the most GPU-hungry integration files earlier in the suite.

    Some large integration files are reliable in isolation but can OOM late in a
    long run if they execute after many other GPU-heavy tests. Running them
    early keeps the suite exercising the same coverage while reducing
    order-dependent memory starvation.
    """

    priority_files = {
        str(Path("tests/integration/test_precond_pivot_methods.py")): 0,
        str(
            Path("tests/integration/test_posterior_sampling_and_heterogeneous_lmc.py")
        ): 1,
    }

    items.sort(
        key=lambda item: (
            priority_files.get(str(Path(item.location[0])), 100),
            str(Path(item.location[0])),
            item.location[1],
            item.name,
        )
    )

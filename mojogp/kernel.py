"""Python Kernel Builder API for MojoGP.

This module provides a Pythonic API for building composite kernels that will be
JIT-compiled to Mojo code. Users can compose kernels using operator overloading:

    kernel = RBF() + Matern52()          # Sum kernel
    kernel = RBF() * Linear()            # Product kernel
    kernel = 2.0 * RBF()                 # Scaled kernel
    kernel = RBF(ard=True)               # ARD kernel
    kernel = Periodic(period=2.0)        # Named initial values

The kernel tree is then converted to Mojo type syntax for JIT compilation.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any, Sequence
from enum import Enum
import numpy as np


class KernelType(Enum):
    """Base kernel types supported by MojoGP."""

    # Continuous kernels (GPU inline evaluation)
    RBF = "RBFComposable"
    MATERN12 = "Matern12Composable"
    MATERN32 = "Matern32Composable"
    MATERN52 = "Matern52Composable"
    PERIODIC = "PeriodicComposable"
    LINEAR = "LinearComposable"
    RQ = "RQComposable"
    POLYNOMIAL = "PolynomialComposable"

    # Categorical kernels (CPU precompute L x L correlation matrix, GPU lookup)
    GD = "GDKernel"
    CR = "CRKernel"
    EHH = "EHHKernel"
    HH = "HHKernel"
    FE = "FEKernel"


# Sets for type classification
_CONTINUOUS_TYPES = {
    KernelType.RBF,
    KernelType.MATERN12,
    KernelType.MATERN32,
    KernelType.MATERN52,
    KernelType.PERIODIC,
    KernelType.LINEAR,
    KernelType.RQ,
    KernelType.POLYNOMIAL,
}

_CATEGORICAL_TYPES = {
    KernelType.GD,
    KernelType.CR,
    KernelType.EHH,
    KernelType.HH,
    KernelType.FE,
}

# Parameter count functions for categorical kernels (L = number of levels)
_CAT_PARAM_COUNTS = {
    KernelType.GD: lambda L: 1,
    KernelType.CR: lambda L: L,
    KernelType.EHH: lambda L: L * (L - 1) // 2,
    KernelType.HH: lambda L: L * (L - 1) // 2,
    KernelType.FE: lambda L: L * (L + 1) // 2,
}


def _inv_softplus(x: float) -> float:
    """Inverse of softplus: log(exp(x) - 1)."""
    if x > 20.0:
        return x
    return float(np.log(np.exp(x) - 1.0))


def _inv_sigmoid(x: float) -> float:
    """Inverse of sigmoid: log(x / (1 - x))."""
    return float(np.log(x / (1.0 - x)))


def _cat_default_raw_param(kernel_type: KernelType, param_index: int, L: int) -> float:
    """Get the default raw (unconstrained) parameter initialization for a categorical kernel.

    Matches the initialization in mixed_composite_training.mojo.
    """
    if kernel_type == KernelType.GD:
        return _inv_softplus(0.5)
    elif kernel_type == KernelType.CR:
        return _inv_softplus(0.3)
    elif kernel_type in (KernelType.EHH, KernelType.HH):
        return _inv_sigmoid(0.25)
    elif kernel_type == KernelType.FE:
        # FE has L*(L-1)/2 angle params (sigmoid*pi) then L diagonal params (softplus)
        n_angles = L * (L - 1) // 2
        if param_index < n_angles:
            return _inv_sigmoid(0.25)
        else:
            return _inv_softplus(0.3)
    return 0.0


_CAT_KERNEL_STR_TO_TYPE = {
    "gd": KernelType.GD,
    "cr": KernelType.CR,
    "ehh": KernelType.EHH,
    "hh": KernelType.HH,
    "fe": KernelType.FE,
}


def build_default_categorical_raw_params(cat_specs: List[Dict[str, Any]]) -> np.ndarray:
    """Build default raw categorical parameter initialization from engine-style specs."""
    raw_params: List[float] = []
    for spec in cat_specs:
        kernel_type = spec["kernel_type"]
        if isinstance(kernel_type, KernelType):
            kernel_enum = kernel_type
        else:
            kernel_enum = _CAT_KERNEL_STR_TO_TYPE[str(kernel_type)]
        levels = int(spec["levels"])
        num_params = _CAT_PARAM_COUNTS[kernel_enum](levels)
        raw_params.extend(
            _cat_default_raw_param(kernel_enum, param_index, levels)
            for param_index in range(num_params)
        )
    return np.asarray(raw_params, dtype=np.float32)


def categorical_prediction_params(
    cat_specs: List[Dict[str, Any]], raw_params: np.ndarray
) -> np.ndarray:
    """Convert stored categorical params to the layout expected by prediction paths.

    The mixed prediction bindings expect constrained categorical parameters,
    while single-output and ICM mixed training store raw unconstrained values.
    """
    raw_params = np.asarray(raw_params, dtype=np.float32)
    out = raw_params.copy()
    offset = 0
    for spec in cat_specs:
        kernel_type = spec["kernel_type"]
        if isinstance(kernel_type, KernelType):
            kernel_name = kernel_type.name.lower()
            kernel_enum = kernel_type
        else:
            kernel_name = str(kernel_type).lower()
            kernel_enum = _CAT_KERNEL_STR_TO_TYPE[kernel_name]
        levels = int(spec["levels"])
        n_params = _CAT_PARAM_COUNTS[kernel_enum](levels)
        block = out[offset : offset + n_params].astype(np.float64)
        if kernel_name in {"gd", "cr"}:
            out[offset : offset + n_params] = np.log1p(
                np.exp(-np.abs(block))
            ) + np.maximum(block, 0.0)
        elif kernel_name in {"ehh", "hh"}:
            out[offset : offset + n_params] = (
                1.0 / (1.0 + np.exp(-block))
            ) * np.pi
        elif kernel_name == "fe":
            n_angles = levels * (levels - 1) // 2
            angles = block[:n_angles]
            diag = block[n_angles:]
            out[offset : offset + n_angles] = (
                1.0 / (1.0 + np.exp(-angles))
            ) * np.pi
            out[offset + n_angles : offset + n_params] = np.log1p(
                np.exp(-np.abs(diag))
            ) + np.maximum(diag, 0.0)
        offset += n_params
    return out.astype(np.float32)


@dataclass
class KernelNode:
    """A node in the kernel composition tree.

    This represents either a base kernel or a composition of kernels.

    When ard_dim is set (by SingleOutputGP with ard=True), base kernels generate
    ARD Mojo types (e.g., RBFComposableARD[5] instead of RBFComposable).
    """

    kernel_type: Optional[KernelType] = None  # For base kernels
    operator: Optional[str] = None  # "sum", "product", or "scale"
    left: Optional["KernelNode"] = None  # Left child for binary ops
    right: Optional["KernelNode"] = None  # Right child for binary ops
    scale_factor: Optional[float] = None  # For scale operator
    ard_dim: Optional[int] = None  # When set, use ARD variant with this many dims
    initial_values: Optional[Dict[str, float]] = None  # Named initial hyperparams
    ard: bool = False  # Whether this kernel requests ARD
    active_dims: Optional[tuple] = None  # Input dimensions this kernel operates on
    levels: Optional[int] = (
        None  # Number of categorical levels (for categorical kernels)
    )
    # Internal: set by compute_dim_permutation() after column reordering
    _dim_start: Optional[int] = field(default=None, repr=False)
    _dim_end: Optional[int] = field(default=None, repr=False)

    def __add__(self, other: "KernelNode") -> "KernelNode":
        """Create a sum kernel: K1 + K2."""
        return KernelNode(operator="sum", left=self, right=other)

    def __radd__(self, other: "KernelNode") -> "KernelNode":
        """Handle right-side addition."""
        if isinstance(other, (int, float)) and other == 0:
            return self  # Handle sum() starting with 0
        return KernelNode(operator="sum", left=other, right=self)

    def __mul__(self, other: Union["KernelNode", float, int]) -> "KernelNode":
        """Create a product kernel: K1 * K2, or a scaled kernel: c * K."""
        if isinstance(other, (float, int)):
            return KernelNode(operator="scale", left=self, scale_factor=float(other))
        return KernelNode(operator="product", left=self, right=other)

    def __rmul__(self, other: Union["KernelNode", float, int]) -> "KernelNode":
        """Handle right-side multiplication (for c * K)."""
        if isinstance(other, (float, int)):
            return KernelNode(operator="scale", left=self, scale_factor=float(other))
        return KernelNode(operator="product", left=other, right=self)

    def with_active_dims(self, dims: Sequence[int]) -> "KernelNode":
        """Restrict this kernel (possibly composite) to specified input dimensions.

        This wraps the kernel so it only operates on the given dimensions of the
        input, enabling dimension routing in compositions:

            kernel = (RBF() * Matern52()).with_active_dims([0, 1]) + Periodic(active_dims=[2])

        Args:
            dims: List or tuple of input dimension indices.

        Returns:
            A new KernelNode with active_dims set.
        """
        import copy

        new_node = copy.deepcopy(self)
        new_node.active_dims = tuple(sorted(dims))
        return new_node

    def has_active_dims(self) -> bool:
        """Check if any kernel in the tree has active_dims set."""
        if self.active_dims is not None:
            return True
        if self.left and self.left.has_active_dims():
            return True
        if self.right and self.right.has_active_dims():
            return True
        return False

    def is_categorical(self) -> bool:
        """Check if this leaf node is a categorical kernel."""
        return self.kernel_type is not None and self.kernel_type in _CATEGORICAL_TYPES

    def is_continuous(self) -> bool:
        """Check if this leaf node is a continuous kernel."""
        return self.kernel_type is not None and self.kernel_type in _CONTINUOUS_TYPES

    def has_categorical(self) -> bool:
        """Check if any kernel in the tree is categorical."""
        if self.is_categorical():
            return True
        if self.left and self.left.has_categorical():
            return True
        if self.right and self.right.has_categorical():
            return True
        return False

    # Map from KernelType to ARD Mojo struct name template
    _ARD_TYPE_MAP = {
        KernelType.RBF: "RBFComposableARD",
        KernelType.MATERN12: "Matern12ComposableARD",
        KernelType.MATERN32: "Matern32ComposableARD",
        KernelType.MATERN52: "Matern52ComposableARD",
        KernelType.PERIODIC: "PeriodicComposableARD",
        KernelType.RQ: "RQComposableARD",
        KernelType.LINEAR: "LinearComposableARD",
        KernelType.POLYNOMIAL: "PolynomialComposableARD",
    }

    def to_mojo_type(self) -> str:
        """Convert this kernel tree to Mojo type syntax.

        Returns:
            A string like "SumKernel[RBFComposable, Matern52Composable]"
            or "RBFComposableARD[5]" for ARD kernels.
            If active_dims is resolved (_dim_start/_dim_end set), wraps in
            DimSliceKernel[..., START, END].
        """
        inner_type = self._inner_mojo_type()

        # Wrap in DimSliceKernel if dimension routing is resolved
        if self._dim_start is not None and self._dim_end is not None:
            return f"DimSliceKernel[{inner_type}, {self._dim_start}, {self._dim_end}]"
        return inner_type

    def _inner_mojo_type(self) -> str:
        """Generate the Mojo type string without DimSlice wrapping."""
        if self.kernel_type is not None:
            # Base kernel — check for ARD variant
            if self.ard_dim is not None and self.kernel_type in self._ARD_TYPE_MAP:
                return f"{self._ARD_TYPE_MAP[self.kernel_type]}[{self.ard_dim}]"
            return self.kernel_type.value

        if self.operator == "sum":
            left_type = self.left.to_mojo_type()
            right_type = self.right.to_mojo_type()
            return f"SumKernel[{left_type}, {right_type}]"

        if self.operator == "product":
            left_type = self.left.to_mojo_type()
            right_type = self.right.to_mojo_type()
            return f"ProductKernel[{left_type}, {right_type}]"

        if self.operator == "scale":
            inner_type = self.left.to_mojo_type()
            return f"ScaleKernel[{inner_type}]"

        raise ValueError(f"Unknown operator: {self.operator}")

    def num_params(self) -> int:
        """Count the total number of parameters in this kernel.

        Returns:
            Total number of kernel parameters (not including noise).
        """
        if self.kernel_type is not None:
            # Categorical kernels: param count depends on number of levels
            if self.kernel_type in _CATEGORICAL_TYPES:
                if self.levels is None:
                    raise ValueError(
                        f"Categorical kernel {self.kernel_type.name} requires levels to be set"
                    )
                return _CAT_PARAM_COUNTS[self.kernel_type](self.levels)

            if self.ard_dim is not None:
                # ARD kernels: DIM per-dim params + outputscale [+ extra params]
                if self.kernel_type in (KernelType.PERIODIC, KernelType.RQ):
                    return self.ard_dim + 2  # DIM ls + period/alpha + outputscale
                if self.kernel_type == KernelType.POLYNOMIAL:
                    return self.ard_dim + 3  # DIM ls + degree + offset + outputscale
                # RBF, Matern, Linear: DIM per-dim params + outputscale
                return self.ard_dim + 1
            # Isotropic: 2 params (lengthscale + outputscale)
            # Periodic/RQ: 3 params (ls + extra + outputscale)
            if self.kernel_type == KernelType.POLYNOMIAL:
                return 3  # degree + offset + outputscale
            if self.kernel_type in (KernelType.PERIODIC, KernelType.RQ):
                return 3
            return 2

        if self.operator == "sum":
            return self.left.num_params() + self.right.num_params()

        if self.operator == "product":
            return self.left.num_params() + self.right.num_params()

        if self.operator == "scale":
            # ScaleKernel adds 1 param (the scale factor)
            return self.left.num_params() + 1

        raise ValueError(f"Unknown operator: {self.operator}")

    def get_param_names(self, prefix: str = "") -> List[str]:
        """Get names for all parameters in this kernel.

        Args:
            prefix: Prefix for parameter names (for nested kernels)

        Returns:
            List of parameter names like ["rbf_lengthscale", "rbf_outputscale", ...]
            For ARD: ["rbf_ls_0", "rbf_ls_1", ..., "rbf_outputscale"]
            For categorical: ["ehh_col2_theta_0", "ehh_col2_theta_1", ...]
        """
        if self.kernel_type is not None:
            base_name = self.kernel_type.name.lower()
            if prefix:
                base_name = f"{prefix}_{base_name}"

            # Categorical kernel parameter names
            if self.kernel_type in _CATEGORICAL_TYPES:
                col_suffix = ""
                if self.active_dims is not None and len(self.active_dims) == 1:
                    col_suffix = f"_col{self.active_dims[0]}"
                n_params = self.num_params()
                if n_params == 1:
                    return [f"{base_name}{col_suffix}_theta"]
                return [f"{base_name}{col_suffix}_theta_{i}" for i in range(n_params)]

            if self.ard_dim is not None:
                # ARD parameter names
                names = [f"{base_name}_ls_{d}" for d in range(self.ard_dim)]
                if self.kernel_type == KernelType.PERIODIC:
                    names.append(f"{base_name}_period")
                elif self.kernel_type == KernelType.RQ:
                    names.append(f"{base_name}_alpha")
                elif self.kernel_type == KernelType.POLYNOMIAL:
                    names.append(f"{base_name}_degree")
                    names.append(f"{base_name}_offset")
                names.append(f"{base_name}_outputscale")
                return names

            if self.kernel_type == KernelType.POLYNOMIAL:
                return [
                    f"{base_name}_degree",
                    f"{base_name}_offset",
                    f"{base_name}_outputscale",
                ]
            if self.kernel_type == KernelType.LINEAR:
                return [
                    f"{base_name}_variance",
                    f"{base_name}_outputscale",
                ]
            if self.kernel_type in (KernelType.PERIODIC, KernelType.RQ):
                extra = "period" if self.kernel_type == KernelType.PERIODIC else "alpha"
                return [
                    f"{base_name}_lengthscale",
                    f"{base_name}_{extra}",
                    f"{base_name}_outputscale",
                ]
            return [f"{base_name}_lengthscale", f"{base_name}_outputscale"]

        if self.operator == "sum":
            left_names = self.left.get_param_names(
                f"{prefix}left" if prefix else "left"
            )
            right_names = self.right.get_param_names(
                f"{prefix}right" if prefix else "right"
            )
            return left_names + right_names

        if self.operator == "product":
            left_names = self.left.get_param_names(
                f"{prefix}left" if prefix else "left"
            )
            right_names = self.right.get_param_names(
                f"{prefix}right" if prefix else "right"
            )
            return left_names + right_names

        if self.operator == "scale":
            inner_names = self.left.get_param_names(prefix)
            scale_name = f"{prefix}_scale" if prefix else "scale"
            return inner_names + [scale_name]

        raise ValueError(f"Unknown operator: {self.operator}")

    def get_initial_params(self) -> np.ndarray:
        """Get initial parameter values as a flat array.

        Returns values from `initial_values` dict where available, defaulting to
        1.0 for any unset parameters. Order matches `get_param_names()`.
        """
        names = self.get_param_names()
        values = np.ones(len(names), dtype=np.float32)
        self._fill_initial_params(values, offset=0)
        return values

    def engine_num_params(self) -> int:
        """Count parameters expected by the live JIT engine.

        The public Python polynomial surface exposes degree, offset, and
        outputscale, while the current generated engine layout still carries an
        internal fixed variance slot. Wrapper-managed categorical leaves are
        excluded here
        because their parameters are passed through separate categorical buffers,
        not the codegen-managed continuous parameter array.
        """
        if self.kernel_type in _CATEGORICAL_TYPES:
            return 0
        if self.kernel_type == KernelType.POLYNOMIAL and self.ard_dim is None:
            return 4
        if self.operator in ("sum", "product"):
            return self.left.engine_num_params() + self.right.engine_num_params()
        if self.operator == "scale":
            return self.left.engine_num_params() + 1
        return self.num_params()

    def engine_trainable_mask(self) -> np.ndarray:
        """Return which engine-layout params may be optimized.

        GPyTorch's PolynomialKernel keeps `power` fixed. The current Mojo
        generated polynomial engine still carries a hidden isotropic variance
        slot and a degree slot, so wrappers freeze those while training offset
        and outputscale.
        """
        mask = np.ones(self.engine_num_params(), dtype=np.bool_)
        self._fill_engine_trainable_mask(mask, 0)
        return mask

    def to_engine_params(self, public_params: np.ndarray) -> np.ndarray:
        """Convert public parameter layout to the current engine layout.

        For mixed kernels this consumes the full public parameter vector but only
        emits the codegen-managed continuous block; categorical params are handled
        separately by the mixed wrapper paths.
        """
        public_params = np.asarray(public_params, dtype=np.float32)
        if public_params.shape != (self.num_params(),):
            raise ValueError(
                f"public_params must have shape ({self.num_params()},), got {public_params.shape}"
            )

        out = np.empty(self.engine_num_params(), dtype=np.float32)
        self._fill_engine_params(public_params, out, 0, 0)
        return out

    def from_engine_params(self, engine_params: np.ndarray) -> np.ndarray:
        """Convert engine parameter layout back to the public layout.

        For mixed kernels, categorical public slots remain at their default
        initialization because the engine parameter vector only carries the
        codegen-managed continuous block.
        """
        engine_params = np.asarray(engine_params, dtype=np.float32)
        if engine_params.shape != (self.engine_num_params(),):
            raise ValueError(
                f"engine_params must have shape ({self.engine_num_params()},), got {engine_params.shape}"
            )

        out = self.get_initial_params()
        self._strip_engine_params(engine_params, out, 0, 0)
        return out

    def _fill_engine_params(
        self,
        public_params: np.ndarray,
        engine_params: np.ndarray,
        public_offset: int,
        engine_offset: int,
    ) -> tuple[int, int]:
        if self.kernel_type is not None:
            n_public = self.num_params()
            if self.kernel_type in _CATEGORICAL_TYPES:
                return public_offset + n_public, engine_offset
            if self.kernel_type == KernelType.POLYNOMIAL and self.ard_dim is None:
                degree, offset, outputscale = public_params[
                    public_offset : public_offset + 3
                ]
                engine_params[engine_offset : engine_offset + 4] = np.array(
                    [1.0, degree, offset, outputscale], dtype=np.float32
                )
                return public_offset + 3, engine_offset + 4

            engine_params[engine_offset : engine_offset + n_public] = public_params[
                public_offset : public_offset + n_public
            ]
            return public_offset + n_public, engine_offset + n_public

        if self.operator in ("sum", "product"):
            public_mid, engine_mid = self.left._fill_engine_params(
                public_params, engine_params, public_offset, engine_offset
            )
            return self.right._fill_engine_params(
                public_params, engine_params, public_mid, engine_mid
            )

        if self.operator == "scale":
            public_mid, engine_mid = self.left._fill_engine_params(
                public_params, engine_params, public_offset, engine_offset
            )
            engine_params[engine_mid] = public_params[public_mid]
            return public_mid + 1, engine_mid + 1

        return public_offset, engine_offset

    def _strip_engine_params(
        self,
        engine_params: np.ndarray,
        public_params: np.ndarray,
        engine_offset: int,
        public_offset: int,
    ) -> tuple[int, int]:
        if self.kernel_type is not None:
            n_public = self.num_params()
            if self.kernel_type in _CATEGORICAL_TYPES:
                return engine_offset, public_offset + n_public
            if self.kernel_type == KernelType.POLYNOMIAL and self.ard_dim is None:
                structural_degree = float(self.initial_values.get("degree", 2.0))
                public_params[public_offset] = structural_degree
                public_params[public_offset + 1 : public_offset + 3] = engine_params[
                    engine_offset + 2 : engine_offset + 4
                ]
                return engine_offset + 4, public_offset + 3

            public_params[public_offset : public_offset + n_public] = engine_params[
                engine_offset : engine_offset + n_public
            ]
            return engine_offset + n_public, public_offset + n_public

        if self.operator in ("sum", "product"):
            engine_mid, public_mid = self.left._strip_engine_params(
                engine_params, public_params, engine_offset, public_offset
            )
            return self.right._strip_engine_params(
                engine_params, public_params, engine_mid, public_mid
            )

        if self.operator == "scale":
            engine_mid, public_mid = self.left._strip_engine_params(
                engine_params, public_params, engine_offset, public_offset
            )
            public_params[public_mid] = engine_params[engine_mid]
            return engine_mid + 1, public_mid + 1

        return engine_offset, public_offset

    def _fill_engine_trainable_mask(self, mask: np.ndarray, engine_offset: int) -> int:
        if self.kernel_type is not None:
            if self.kernel_type in _CATEGORICAL_TYPES:
                return engine_offset
            if self.kernel_type == KernelType.POLYNOMIAL and self.ard_dim is None:
                mask[engine_offset] = False  # hidden variance
                mask[engine_offset + 1] = False  # fixed degree/power
                return engine_offset + 4
            return engine_offset + self.engine_num_params()

        if self.operator in ("sum", "product"):
            mid = self.left._fill_engine_trainable_mask(mask, engine_offset)
            return self.right._fill_engine_trainable_mask(mask, mid)

        if self.operator == "scale":
            mid = self.left._fill_engine_trainable_mask(mask, engine_offset)
            return mid + 1

        return engine_offset

    def _fill_initial_params(self, values: np.ndarray, offset: int) -> int:
        """Recursively fill initial params from initial_values dicts."""
        if self.kernel_type is not None:
            n = self.num_params()

            # Categorical kernels: use default raw parameter initialization
            if self.kernel_type in _CATEGORICAL_TYPES:
                L = self.levels or 2
                for i in range(n):
                    values[offset + i] = _cat_default_raw_param(self.kernel_type, i, L)
                return offset + n

            if self.initial_values:
                names = self.get_param_names()
                for i, name in enumerate(names):
                    # Match by short name (e.g. "lengthscale" matches "rbf_lengthscale")
                    for key, val in self.initial_values.items():
                        if name.endswith(f"_{key}") or name == key:
                            values[offset + i] = val
                            break
            return offset + n

        if self.operator in ("sum", "product"):
            mid = self.left._fill_initial_params(values, offset)
            end = self.right._fill_initial_params(values, mid)
            return end

        if self.operator == "scale":
            mid = self.left._fill_initial_params(values, offset)
            if self.initial_values and "scale" in self.initial_values:
                values[mid] = self.initial_values["scale"]
            elif self.scale_factor is not None:
                values[mid] = self.scale_factor
            return mid + 1

        return offset

    def has_ard(self) -> bool:
        """Check if any kernel in the tree requests ARD."""
        if self.ard:
            return True
        if self.left and self.left.has_ard():
            return True
        if self.right and self.right.has_ard():
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize kernel tree to a JSON-compatible dict."""
        d: Dict[str, Any] = {}
        if self.kernel_type is not None:
            d["kernel_type"] = self.kernel_type.name
        if self.operator is not None:
            d["operator"] = self.operator
        if self.left is not None:
            d["left"] = self.left.to_dict()
        if self.right is not None:
            d["right"] = self.right.to_dict()
        if self.scale_factor is not None:
            d["scale_factor"] = self.scale_factor
        if self.ard_dim is not None:
            d["ard_dim"] = self.ard_dim
        if self.initial_values is not None:
            d["initial_values"] = self.initial_values
        if self.ard:
            d["ard"] = True
        if self.active_dims is not None:
            d["active_dims"] = list(self.active_dims)
        if self.levels is not None:
            d["levels"] = self.levels
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "KernelNode":
        """Deserialize kernel tree from a dict."""
        active_dims = None
        if "active_dims" in d:
            active_dims = tuple(d["active_dims"])
        return KernelNode(
            kernel_type=KernelType[d["kernel_type"]] if "kernel_type" in d else None,
            operator=d.get("operator"),
            left=KernelNode.from_dict(d["left"]) if "left" in d else None,
            right=KernelNode.from_dict(d["right"]) if "right" in d else None,
            scale_factor=d.get("scale_factor"),
            ard_dim=d.get("ard_dim"),
            initial_values=d.get("initial_values"),
            ard=d.get("ard", False),
            active_dims=active_dims,
            levels=d.get("levels"),
        )

    def evaluate(
        self,
        X: np.ndarray,
        X2: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the kernel matrix K(X, X2) in pure Python/NumPy.

        This is a convenience method for inspecting kernel matrices without
        needing the Mojo backend. It supports all base kernels and compositions.

        Args:
            X: Input data [n, d]
            X2: Second input data [m, d]. If None, computes K(X, X).
            params: Kernel parameters as a flat array. If None, uses all 1.0.
                    Order matches get_param_names().

        Returns:
            Kernel matrix [n, m] (or [n, n] if X2 is None)
        """
        X = np.asarray(X, dtype=np.float64)
        if X2 is not None:
            X2 = np.asarray(X2, dtype=np.float64)
        else:
            X2 = X

        if params is None:
            params = np.ones(self.num_params(), dtype=np.float64)
        else:
            params = np.asarray(params, dtype=np.float64)

        K, _ = self._evaluate_recursive(X, X2, params, offset=0)
        return K.astype(np.float32)

    def _evaluate_recursive(
        self,
        X: np.ndarray,
        X2: np.ndarray,
        params: np.ndarray,
        offset: int,
    ) -> tuple:
        """Recursively evaluate kernel tree.

        Returns:
            (K, next_offset) where K is the kernel matrix and next_offset
            is the parameter index after consuming this node's params.
        """
        # If this node has active_dims, slice input columns before evaluating
        if self.active_dims is not None:
            X_sliced = X[:, list(self.active_dims)]
            X2_sliced = X2[:, list(self.active_dims)]
            # Evaluate the kernel on sliced inputs (without active_dims to avoid recursion)
            import copy

            node_without_dims = copy.deepcopy(self)
            node_without_dims.active_dims = None
            node_without_dims._dim_start = None
            node_without_dims._dim_end = None
            return node_without_dims._evaluate_recursive(
                X_sliced, X2_sliced, params, offset
            )

        if self.kernel_type is not None:
            return self._evaluate_base(X, X2, params, offset)

        if self.operator == "sum":
            K_left, mid = self.left._evaluate_recursive(X, X2, params, offset)
            K_right, end = self.right._evaluate_recursive(X, X2, params, mid)
            return K_left + K_right, end

        if self.operator == "product":
            K_left, mid = self.left._evaluate_recursive(X, X2, params, offset)
            K_right, end = self.right._evaluate_recursive(X, X2, params, mid)
            return K_left * K_right, end

        if self.operator == "scale":
            K_inner, mid = self.left._evaluate_recursive(X, X2, params, offset)
            scale = params[mid]
            return scale * K_inner, mid + 1

        raise ValueError(f"Unknown operator: {self.operator}")

    def _evaluate_base(
        self,
        X: np.ndarray,
        X2: np.ndarray,
        params: np.ndarray,
        offset: int,
    ) -> tuple:
        """Evaluate a base kernel.

        Returns:
            (K, next_offset)
        """
        n_params = self.num_params()
        p = params[offset : offset + n_params]

        if self.kernel_type == KernelType.LINEAR:
            # params: [variance, outputscale]
            variance, outputscale = p[0], p[1]
            K = outputscale * variance * (X @ X2.T)
            return K, offset + n_params

        if self.kernel_type == KernelType.POLYNOMIAL:
            if self.ard_dim is not None:
                raise NotImplementedError(
                    "ARD polynomial kernel is not supported in Python-side evaluation. "
                    "Use ard=False for polynomial kernels, or use a different kernel type."
                )
            # params: [degree, offset, outputscale]
            degree, poly_offset, outputscale = p[0], p[1], p[2]
            degree_int = int(round(float(degree)))
            if abs(float(degree) - float(degree_int)) > 1e-5 or degree_int < 1:
                raise ValueError(
                    "Polynomial kernel degree must be a fixed positive integer for "
                    f"exact covariance evaluation, got {float(degree):.6g}."
                )
            K = outputscale * np.power(X @ X2.T + poly_offset, degree_int)
            return K, offset + n_params

        # Distance-based kernels
        if self.ard_dim is not None and self.kernel_type not in (
            KernelType.LINEAR,
            KernelType.POLYNOMIAL,
        ):
            # ARD: first ard_dim params are per-dim lengthscales
            lengthscales = p[: self.ard_dim]
            extra_idx = self.ard_dim
        else:
            # Isotropic: first param is lengthscale
            lengthscales = np.array([p[0]])
            extra_idx = 1

        # Compute scaled distance
        if len(lengthscales) > 1:
            X_scaled = X / lengthscales[np.newaxis, :]
            X2_scaled = X2 / lengthscales[np.newaxis, :]
        else:
            X_scaled = X / lengthscales[0]
            X2_scaled = X2 / lengthscales[0]

        # Squared distance
        sq_dist = (
            np.sum(X_scaled**2, axis=1, keepdims=True)
            + np.sum(X2_scaled**2, axis=1, keepdims=True).T
            - 2 * X_scaled @ X2_scaled.T
        )
        sq_dist = np.maximum(sq_dist, 0.0)
        dist = np.sqrt(sq_dist + 1e-20)

        if self.kernel_type == KernelType.RBF:
            outputscale = p[extra_idx]
            K = outputscale * np.exp(-0.5 * sq_dist)
            return K, offset + n_params

        if self.kernel_type == KernelType.MATERN12:
            outputscale = p[extra_idx]
            K = outputscale * np.exp(-dist)
            return K, offset + n_params

        if self.kernel_type == KernelType.MATERN32:
            outputscale = p[extra_idx]
            sqrt3_r = np.sqrt(3.0) * dist
            K = outputscale * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)
            return K, offset + n_params

        if self.kernel_type == KernelType.MATERN52:
            outputscale = p[extra_idx]
            sqrt5_r = np.sqrt(5.0) * dist
            K = outputscale * (1.0 + sqrt5_r + 5.0 / 3.0 * sq_dist) * np.exp(-sqrt5_r)
            return K, offset + n_params

        if self.kernel_type == KernelType.PERIODIC:
            period = p[extra_idx]
            outputscale = p[extra_idx + 1]
            # Periodic kernel: outputscale * exp(-2 * sin^2(pi*|x-x'|/period) / l^2)
            # For multi-dim, sum over dimensions
            if len(lengthscales) > 1:
                X_raw = X
                X2_raw = X2
            else:
                X_raw = X * lengthscales[0]  # undo scaling
                X2_raw = X2 * lengthscales[0]
            # Compute per-dimension distances
            diff = X_raw[:, np.newaxis, :] - X2_raw[np.newaxis, :, :]  # [n, m, d]
            sin_term = np.sin(np.pi * diff / period) ** 2
            if len(lengthscales) > 1:
                sin_sum = np.sum(
                    sin_term / (lengthscales[np.newaxis, np.newaxis, :] ** 2), axis=2
                )
            else:
                sin_sum = np.sum(sin_term, axis=2) / (lengthscales[0] ** 2)
            K = outputscale * np.exp(-2.0 * sin_sum)
            return K, offset + n_params

        if self.kernel_type == KernelType.RQ:
            alpha_param = p[extra_idx]
            outputscale = p[extra_idx + 1]
            K = outputscale * (1.0 + sq_dist / (2.0 * alpha_param)) ** (-alpha_param)
            return K, offset + n_params

        raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.kernel_type is not None:
            name = self.kernel_type.name.upper()
            parts = []
            if self.levels is not None:
                parts.append(f"levels={self.levels}")
            if self.ard:
                parts.append("ard=True")
            if self.active_dims is not None:
                parts.append(f"active_dims={list(self.active_dims)}")
            if self.initial_values:
                for k, v in self.initial_values.items():
                    if k == "lengthscale" and v == 1.0:
                        continue
                    if k == "outputscale" and v == 1.0:
                        continue
                    if k == "variance" and v == 1.0:
                        continue
                    parts.append(f"{k}={v}")
            args = ", ".join(parts)
            return f"{name}({args})" if parts else f"{name}()"

        if self.operator == "sum":
            return f"({self.left!r} + {self.right!r})"

        if self.operator == "product":
            return f"({self.left!r} * {self.right!r})"

        if self.operator == "scale":
            return f"({self.scale_factor} * {self.left!r})"

        # Composite with active_dims (via with_active_dims())
        base_repr = "KernelNode(?)"
        if self.active_dims is not None:
            return f"{base_repr}.with_active_dims({list(self.active_dims)})"
        return base_repr


def _collect_active_dims_groups(
    kernel: KernelNode, groups: List[tuple], node_ids: List[int], current_id: List[int]
) -> None:
    """Walk the kernel tree and collect (node_id, active_dims) pairs."""
    my_id = current_id[0]
    current_id[0] += 1

    if kernel.active_dims is not None and not kernel.is_categorical():
        groups.append(kernel.active_dims)
        node_ids.append(my_id)

    if kernel.left is not None:
        _collect_active_dims_groups(kernel.left, groups, node_ids, current_id)
    if kernel.right is not None:
        _collect_active_dims_groups(kernel.right, groups, node_ids, current_id)


def _set_dim_ranges(
    kernel: KernelNode, dim_map: Dict[int, tuple], current_id: List[int]
) -> None:
    """Walk the tree and set _dim_start/_dim_end on nodes with active_dims."""
    my_id = current_id[0]
    current_id[0] += 1

    if my_id in dim_map:
        kernel._dim_start, kernel._dim_end = dim_map[my_id]

    if kernel.left is not None:
        _set_dim_ranges(kernel.left, dim_map, current_id)
    if kernel.right is not None:
        _set_dim_ranges(kernel.right, dim_map, current_id)


def compute_dim_permutation(
    kernel: KernelNode,
    total_dim: int,
    cat_col_indices: Optional[List[int]] = None,
) -> tuple:
    """Compute column reordering that makes all active_dims contiguous for DimSlice.

    This handles non-contiguous and overlapping active_dims by reordering the input
    columns so each kernel's dimensions form a contiguous block.

    Args:
        kernel: Kernel tree (may contain active_dims on any node).
        total_dim: Total number of input dimensions (continuous only).
        cat_col_indices: Indices of categorical columns (for validation).

    Returns:
        (permutation, effective_dim): Column permutation for X and the effective
        input dimension after reordering. permutation is None if no reordering needed.
    """
    import copy

    # Collect all active_dims groups from the tree
    groups: List[tuple] = []
    node_ids: List[int] = []
    _collect_active_dims_groups(kernel, groups, node_ids, [0])

    if not groups:
        return None, total_dim

    # Validate: no overlap with cat_dims
    if cat_col_indices:
        cat_set = set(cat_col_indices)
        for g in groups:
            overlap = set(g) & cat_set
            if overlap:
                raise ValueError(
                    f"active_dims {list(g)} overlap with cat_dims {list(overlap)}"
                )

    # Validate: all indices in range
    for g in groups:
        for d in g:
            if d < 0 or d >= total_dim:
                raise ValueError(
                    f"active_dims index {d} out of range for {total_dim}-dimensional input"
                )

    # Build the permutation: place each group's dims contiguously
    # Track which dims are used and their position in the reordered layout
    used_dims = []  # ordered list of dims as they appear in reordered X
    dim_map = {}  # node_id -> (start, end) in reordered layout

    for i, g in enumerate(groups):
        start = len(used_dims)
        for d in g:
            used_dims.append(d)
        end = len(used_dims)
        dim_map[node_ids[i]] = (start, end)

    # Add any remaining dims not covered by any active_dims group
    all_active = set()
    for g in groups:
        all_active.update(g)
    remaining = [d for d in range(total_dim) if d not in all_active]
    used_dims.extend(remaining)

    effective_dim = len(used_dims)

    # Check if permutation is identity (already contiguous, no reordering needed)
    is_identity = used_dims == list(range(total_dim)) and effective_dim == total_dim
    permutation = None if is_identity else used_dims

    # Set _dim_start/_dim_end on the kernel tree nodes
    _set_dim_ranges(kernel, dim_map, [0])

    return permutation, effective_dim


# ============================================================================
# Kernel tree analysis: decompose into continuous + categorical components
# ============================================================================


@dataclass
class CategoricalSpec:
    """Specification for one categorical variable in a mixed kernel."""

    kernel_type: KernelType  # EHH, GD, CR, HH, FE
    levels: int  # Number of categorical levels
    col_index: int  # Original column index (from active_dims)
    param_names: List[str]  # Parameter names for this variable


@dataclass
class KernelTreeAnalysis:
    """Result of analyzing a kernel tree for mixed continuous + categorical composition.

    Fields:
        structured_kernel: The original kernel tree, preserved exactly for JIT/codegen
            paths that support arbitrary mixed compositions.
        continuous_kernel: The continuous sub-tree (with categorical nodes removed), or
            None if the kernel is purely categorical. This is a continuous-only
            projection and does not preserve arbitrary mixed structure.
        categorical_specs: One CategoricalSpec per categorical variable, in tree order.
        compose_op: Top-level compose operator summary.
        continuous_dims: Column indices for the continuous kernel's input.
        categorical_dims: Column indices for categorical variables.
        is_pure_continuous: True if no categorical nodes exist.
        is_pure_categorical: True if no continuous nodes exist.
    """

    structured_kernel: KernelNode
    continuous_kernel: Optional[KernelNode]
    categorical_specs: List[CategoricalSpec]
    compose_op: str  # "product" or "sum"
    continuous_dims: List[int]
    categorical_dims: List[int]
    is_pure_continuous: bool
    is_pure_categorical: bool


def analyze_kernel_tree(kernel: KernelNode, total_dim: int) -> KernelTreeAnalysis:
    """Analyze a kernel tree while preserving arbitrary mixed structure.

    Supported patterns:
        - Pure continuous: RBF() * Matern52()
        - Product: RBF(active_dims=[0,1]) * EHH(active_dims=[2], levels=5)
        - Sum: RBF(active_dims=[0,1]) + EHH(active_dims=[2], levels=5)
        - Multi-categorical: RBF(...) * EHH(...) * GD(...)
        - Pure categorical: EHH(...) * GD(...)
        - Arbitrary mixed trees: (RBF() + Matern32()) * (GD() + CR())

    Args:
        kernel: Root of the kernel composition tree.
        total_dim: Total number of input dimensions.

    Returns:
        KernelTreeAnalysis with the decomposition.
    """
    if kernel.operator == "scale":
        if kernel.left and kernel.left.has_categorical():
            raise ValueError(
                "Cannot apply scale to a kernel tree containing categorical nodes. "
                "Categorical kernels produce correlation matrices (bounded [0,1] or [-1,1]) "
                "and should not be scaled."
            )
    cat_specs: List[CategoricalSpec] = []
    _collect_categorical_specs(kernel, cat_specs)
    cat_dims_set = {spec.col_index for spec in cat_specs}
    cat_dims = sorted(cat_dims_set)
    cont_dims = [d for d in range(total_dim) if d not in cat_dims_set]
    compose_op = kernel.operator if kernel.operator in ("sum", "product") else "product"
    is_pure_continuous = not cat_specs
    continuous_kernel = (
        kernel if is_pure_continuous else _strip_categorical_subtree(kernel)
    )

    return KernelTreeAnalysis(
        structured_kernel=kernel,
        continuous_kernel=continuous_kernel,
        categorical_specs=cat_specs,
        compose_op=compose_op,
        continuous_dims=cont_dims,
        categorical_dims=cat_dims,
        is_pure_continuous=is_pure_continuous,
        is_pure_categorical=continuous_kernel is None,
    )


def _collect_categorical_specs(
    node: KernelNode, cat_specs: List[CategoricalSpec]
) -> None:
    """Collect categorical leaves in left-to-right tree order."""
    if node.kernel_type is not None:
        if node.is_categorical():
            cat_specs.append(_make_cat_spec(node))
        return

    if node.left:
        _collect_categorical_specs(node.left, cat_specs)
    if node.right:
        _collect_categorical_specs(node.right, cat_specs)


def _strip_categorical_subtree(node: KernelNode) -> Optional[KernelNode]:
    """Return the continuous-only projection of a kernel tree.

    This returns a continuous-only projection for callers that need separated
    continuous inputs; arbitrary mixed-tree structure is preserved separately in
    ``KernelTreeAnalysis.structured_kernel``.
    """
    if node.kernel_type is not None:
        return None if node.is_categorical() else node

    if node.operator == "scale":
        inner = _strip_categorical_subtree(node.left)
        if inner is None:
            return None
        return KernelNode(
            operator="scale",
            left=inner,
            scale_factor=node.scale_factor,
        )

    if node.operator in ("sum", "product"):
        left = _strip_categorical_subtree(node.left) if node.left else None
        right = _strip_categorical_subtree(node.right) if node.right else None
        if left is None:
            return right
        if right is None:
            return left
        return KernelNode(operator=node.operator, left=left, right=right)

    if node.operator is None:
        return

    raise ValueError(f"Unknown kernel operator: {node.operator}")


def continuous_kernel_tree(kernel: KernelNode) -> Optional[KernelNode]:
    """Return the continuous-only projection of the current kernel tree.

    This differs from ``KernelTreeAnalysis.continuous_kernel`` when later fit-time
    transformations have already updated the live kernel tree (for example ARD
    resolution or active-dims remapping).
    """
    if kernel.has_categorical():
        return _strip_categorical_subtree(kernel)
    return kernel


def _make_cat_spec(node: KernelNode) -> CategoricalSpec:
    """Create a CategoricalSpec from a categorical KernelNode."""
    if not node.is_categorical():
        raise ValueError(f"Expected categorical kernel, got {node.kernel_type}")
    if node.levels is None:
        raise ValueError(f"Categorical kernel {node.kernel_type.name} missing levels")
    if node.active_dims is None or len(node.active_dims) != 1:
        raise ValueError(
            f"Categorical kernel {node.kernel_type.name} must have exactly 1 active_dim, "
            f"got {node.active_dims}"
        )
    return CategoricalSpec(
        kernel_type=node.kernel_type,
        levels=node.levels,
        col_index=node.active_dims[0],
        param_names=node.get_param_names(),
    )


class Kernel:
    """Factory class for creating kernel nodes.

    Usage:
        kernel = Kernel.rbf() + Kernel.matern52()
        kernel = Kernel.rbf(ard=True) * Kernel.linear()
        kernel = 2.0 * Kernel.rbf()
        kernel = Kernel.periodic(period=2.0)
        kernel = Kernel.rbf(active_dims=[0, 1]) * Kernel.periodic(active_dims=[2])
    """

    @staticmethod
    def rbf(
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        ard: bool = False,
        active_dims: Optional[List[int]] = None,
    ) -> KernelNode:
        """Create an RBF (Squared Exponential) kernel.

        Args:
            active_dims: Input dimensions this kernel operates on. If None, all dims.
        """
        return KernelNode(
            kernel_type=KernelType.RBF,
            ard=ard,
            initial_values={"lengthscale": lengthscale, "outputscale": outputscale},
            active_dims=tuple(sorted(active_dims)) if active_dims is not None else None,
        )

    @staticmethod
    def matern12(
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        ard: bool = False,
        active_dims: Optional[List[int]] = None,
    ) -> KernelNode:
        """Create a Matern-1/2 (Exponential) kernel.

        Args:
            active_dims: Input dimensions this kernel operates on. If None, all dims.
        """
        return KernelNode(
            kernel_type=KernelType.MATERN12,
            ard=ard,
            initial_values={"lengthscale": lengthscale, "outputscale": outputscale},
            active_dims=tuple(sorted(active_dims)) if active_dims is not None else None,
        )

    @staticmethod
    def matern32(
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        ard: bool = False,
        active_dims: Optional[List[int]] = None,
    ) -> KernelNode:
        """Create a Matern-3/2 kernel.

        Args:
            active_dims: Input dimensions this kernel operates on. If None, all dims.
        """
        return KernelNode(
            kernel_type=KernelType.MATERN32,
            ard=ard,
            initial_values={"lengthscale": lengthscale, "outputscale": outputscale},
            active_dims=tuple(sorted(active_dims)) if active_dims is not None else None,
        )

    @staticmethod
    def matern52(
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        ard: bool = False,
        active_dims: Optional[List[int]] = None,
    ) -> KernelNode:
        """Create a Matern-5/2 kernel.

        Args:
            active_dims: Input dimensions this kernel operates on. If None, all dims.
        """
        return KernelNode(
            kernel_type=KernelType.MATERN52,
            ard=ard,
            initial_values={"lengthscale": lengthscale, "outputscale": outputscale},
            active_dims=tuple(sorted(active_dims)) if active_dims is not None else None,
        )

    @staticmethod
    def periodic(
        lengthscale: float = 1.0,
        period: float = 1.0,
        outputscale: float = 1.0,
        ard: bool = False,
        active_dims: Optional[List[int]] = None,
    ) -> KernelNode:
        """Create a Periodic kernel.

        Args:
            active_dims: Input dimensions this kernel operates on. If None, all dims.
        """
        return KernelNode(
            kernel_type=KernelType.PERIODIC,
            ard=ard,
            initial_values={
                "lengthscale": lengthscale,
                "period": period,
                "outputscale": outputscale,
            },
            active_dims=tuple(sorted(active_dims)) if active_dims is not None else None,
        )

    @staticmethod
    def linear(
        variance: float = 1.0,
        outputscale: float = 1.0,
        ard: bool = False,
        active_dims: Optional[List[int]] = None,
    ) -> KernelNode:
        """Create a Linear kernel.

        Args:
            active_dims: Input dimensions this kernel operates on. If None, all dims.
        """
        return KernelNode(
            kernel_type=KernelType.LINEAR,
            ard=ard,
            initial_values={"variance": variance, "outputscale": outputscale},
            active_dims=tuple(sorted(active_dims)) if active_dims is not None else None,
        )

    @staticmethod
    def rq(
        lengthscale: float = 1.0,
        alpha: float = 1.0,
        outputscale: float = 1.0,
        ard: bool = False,
        active_dims: Optional[List[int]] = None,
    ) -> KernelNode:
        """Create a Rational Quadratic kernel.

        Args:
            active_dims: Input dimensions this kernel operates on. If None, all dims.
        """
        return KernelNode(
            kernel_type=KernelType.RQ,
            ard=ard,
            initial_values={
                "lengthscale": lengthscale,
                "alpha": alpha,
                "outputscale": outputscale,
            },
            active_dims=tuple(sorted(active_dims)) if active_dims is not None else None,
        )

    @staticmethod
    def polynomial(
        degree: float = 3.0,
        offset: float = 1.0,
        outputscale: float = 1.0,
        ard: bool = False,
        active_dims: Optional[List[int]] = None,
    ) -> KernelNode:
        """Create a Polynomial kernel: outputscale * (X @ X2.T + offset)^degree.

        Args:
            degree: Fixed positive integer polynomial degree (typically 2 or 3).
            offset: Offset added before exponentiation.
            outputscale: Output scale multiplier.
            ard: If True, use per-dimension lengthscales (ARD).
                 Note: ARD polynomial is not yet supported in Python-side evaluation.
            active_dims: Input dimensions this kernel operates on. If None, all dims.
        """
        return KernelNode(
            kernel_type=KernelType.POLYNOMIAL,
            ard=ard,
            initial_values={
                "degree": degree,
                "offset": offset,
                "outputscale": outputscale,
            },
            active_dims=tuple(sorted(active_dims)) if active_dims is not None else None,
        )

    # --- Categorical kernel factory methods ---

    @staticmethod
    def gd(levels: int, active_dims: Optional[List[int]] = None) -> KernelNode:
        """Create a Gower Distance (GD) categorical kernel.

        The simplest categorical kernel: one parameter controls the correlation
        between all pairs of different levels. k(c_i, c_j) = 1 if c_i == c_j,
        else rho (with rho in [0,1]).

        Args:
            levels: Number of categorical levels (must be >= 2).
            active_dims: Must be exactly 1 dimension (the categorical column).
        """
        return _make_categorical_node(KernelType.GD, levels, active_dims)

    @staticmethod
    def cr(levels: int, active_dims: Optional[List[int]] = None) -> KernelNode:
        """Create a Compound Radial (CR) categorical kernel.

        Each level has its own radial parameter. More flexible than GD but
        fewer parameters than EHH.

        Args:
            levels: Number of categorical levels (must be >= 2).
            active_dims: Must be exactly 1 dimension (the categorical column).
        """
        return _make_categorical_node(KernelType.CR, levels, active_dims)

    @staticmethod
    def ehh(levels: int, active_dims: Optional[List[int]] = None) -> KernelNode:
        """Create an Exponential Homogeneous Hypersphere (EHH) categorical kernel.

        Full pairwise correlation via hypersphere decomposition. L*(L-1)/2
        parameters for L levels.

        Args:
            levels: Number of categorical levels (must be >= 2).
            active_dims: Must be exactly 1 dimension (the categorical column).
        """
        return _make_categorical_node(KernelType.EHH, levels, active_dims)

    @staticmethod
    def hh(levels: int, active_dims: Optional[List[int]] = None) -> KernelNode:
        """Create a Homogeneous Hypersphere (HH) categorical kernel.

        Similar to EHH but with different parameterization.

        Args:
            levels: Number of categorical levels (must be >= 2).
            active_dims: Must be exactly 1 dimension (the categorical column).
        """
        return _make_categorical_node(KernelType.HH, levels, active_dims)

    @staticmethod
    def fe(levels: int, active_dims: Optional[List[int]] = None) -> KernelNode:
        """Create a Full Estimation (FE) categorical kernel.

        Most flexible: estimates the full L x L correlation matrix.
        L*(L+1)/2 parameters (L*(L-1)/2 angles + L diagonal).

        Args:
            levels: Number of categorical levels (must be >= 2).
            active_dims: Must be exactly 1 dimension (the categorical column).
        """
        return _make_categorical_node(KernelType.FE, levels, active_dims)


def _make_categorical_node(
    kernel_type: KernelType,
    levels: int,
    active_dims: Optional[List[int]] = None,
) -> KernelNode:
    """Validate and create a categorical KernelNode."""
    if levels < 2:
        raise ValueError(
            f"Categorical kernel {kernel_type.name} requires levels >= 2, got {levels}"
        )
    if active_dims is not None:
        if len(active_dims) != 1:
            raise ValueError(
                f"Categorical kernel must have exactly 1 active_dim, got {len(active_dims)}"
            )
    return KernelNode(
        kernel_type=kernel_type,
        levels=levels,
        active_dims=tuple(sorted(active_dims)) if active_dims is not None else None,
    )


def make_ard_kernel(kernel: KernelNode, dim: int) -> KernelNode:
    """Transform a kernel tree to use ARD variants with per-dimension lengthscales.

    All continuous base kernels are converted to their ARD variants.
    Categorical kernels are passed through unchanged (they don't have ARD).
    Composition operators are preserved. Initial values are preserved.

    When a kernel has active_dims, the ARD dimension is len(active_dims),
    not the full input dimension. This ensures e.g. RBF(active_dims=[0,1], ard=True)
    on 5D input gets RBFComposableARD[2] (2 lengthscales), not RBFComposableARD[5].

    Args:
        kernel: Original kernel tree
        dim: Number of continuous input dimensions (ARD_DIM)

    Returns:
        New KernelNode tree with ARD variants
    """
    if kernel.kernel_type is not None:
        # Categorical kernels don't have ARD — pass through unchanged
        if kernel.kernel_type in _CATEGORICAL_TYPES:
            return KernelNode(
                kernel_type=kernel.kernel_type,
                levels=kernel.levels,
                active_dims=kernel.active_dims,
            )

        # If this kernel has active_dims, ARD dim = number of active dims
        ard_dim = len(kernel.active_dims) if kernel.active_dims is not None else dim
        return KernelNode(
            kernel_type=kernel.kernel_type,
            ard_dim=ard_dim,
            initial_values=kernel.initial_values,
            ard=kernel.ard,
            active_dims=kernel.active_dims,
        )

    if kernel.operator == "sum":
        return KernelNode(
            operator="sum",
            left=make_ard_kernel(kernel.left, dim),
            right=make_ard_kernel(kernel.right, dim),
            active_dims=kernel.active_dims,
        )

    if kernel.operator == "product":
        return KernelNode(
            operator="product",
            left=make_ard_kernel(kernel.left, dim),
            right=make_ard_kernel(kernel.right, dim),
            active_dims=kernel.active_dims,
        )

    if kernel.operator == "scale":
        return KernelNode(
            operator="scale",
            left=make_ard_kernel(kernel.left, dim),
            scale_factor=kernel.scale_factor,
            active_dims=kernel.active_dims,
        )

    raise ValueError(f"Unknown operator: {kernel.operator}")


# Convenience aliases — continuous kernels
RBF = Kernel.rbf
Matern12 = Kernel.matern12
Matern32 = Kernel.matern32
Matern52 = Kernel.matern52
Periodic = Kernel.periodic
Linear = Kernel.linear
RQ = Kernel.rq
Polynomial = Kernel.polynomial

# Convenience aliases — categorical kernels
GD = Kernel.gd
CR = Kernel.cr
EHH = Kernel.ehh
HH = Kernel.hh
FE = Kernel.fe


if __name__ == "__main__":
    # Test the kernel builder
    print("Testing Kernel Builder API")
    print("=" * 50)

    # Test base kernels
    k1 = Kernel.rbf()
    print(f"RBF: {k1.to_mojo_type()}")
    print(f"  Params: {k1.num_params()}")
    print(f"  Names: {k1.get_param_names()}")

    # Test sum kernel
    k2 = Kernel.rbf() + Kernel.matern52()
    print(f"\nRBF + Matern52: {k2.to_mojo_type()}")
    print(f"  Params: {k2.num_params()}")
    print(f"  Names: {k2.get_param_names()}")

    # Test product kernel
    k3 = Kernel.rbf() * Kernel.linear()
    print(f"\nRBF * Linear: {k3.to_mojo_type()}")
    print(f"  Params: {k3.num_params()}")

    # Test scaled kernel
    k4 = 2.0 * Kernel.rbf()
    print(f"\n2.0 * RBF: {k4.to_mojo_type()}")
    print(f"  Params: {k4.num_params()}")

    # Test nested composition
    k5 = (Kernel.rbf() + Kernel.matern52()) * Kernel.linear()
    print(f"\n(RBF + Matern52) * Linear: {k5.to_mojo_type()}")
    print(f"  Params: {k5.num_params()}")

    # Test repr
    print(f"\nRepr: {k5!r}")

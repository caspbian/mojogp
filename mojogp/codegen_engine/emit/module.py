"""Full Mojo module assembly.

Combines GPU kernel code, the JIT adapter struct,
train_python binding, and PyInit into a complete
module source string ready for `mojo build --emit shared-lib`.

Uses the JIT training path (kernels.jit.jit_training) — no AOT modifications.
"""

from .builder import MojoBuilder
from .gpu_kernel import (
    emit_forward_matvec,
    emit_gradient_matvec,
    emit_cross_matvec,
    emit_extract_diagonal,
)
from .mojo_printer import collect_math_imports
from ..ir import IRKernel
from ..schedule import ScheduleConfig


def _emit_imports(
    b: MojoBuilder, kernel: IRKernel, kernel_type_str: str, ard_imports: str = ""
):
    """Emit all import statements for the generated module."""
    b.line("from python import Python, PythonObject")
    b.line("from python.bindings import PythonModuleBuilder")
    b.line("from gpu.host import DeviceContext, DeviceBuffer, HostBuffer")
    b.line("from gpu import block_dim, block_idx, thread_idx")
    b.line("from gpu.sync import barrier")
    b.line("from gpu.memory import external_memory, AddressSpace")
    b.line("from memory import UnsafePointer")
    b.line(collect_math_imports(kernel))
    b.line("from collections import InlineArray")
    b.line("from os import abort")
    b.blank()
    b.comment("Import kernel types")
    b.line("from kernels.composable_kernel import (")
    with b.block():
        b.line("ComposableKernel,")
        b.line("RBFComposable, Matern12Composable, Matern32Composable,")
        b.line("Matern52Composable, PeriodicComposable, LinearComposable,")
        b.line("RQComposable, PolynomialComposable,")
        b.line("SumKernel, ProductKernel, ScaleKernel,")
    b.line(")")
    if ard_imports:
        b.raw(ard_imports)
    b.blank()
    b.comment("JIT training infrastructure (no AOT modification)")
    b.line(
        "from kernels.jit.jit_training import JITGradientProvider, train_jit_with_provider"
    )
    b.line("from kernels.composite_provider import CompositeProvider")
    b.line("from kernels.training_types import TrainingResultGeneric")
    b.line("from kernels.py_conversion import bulk_copy_to_host_buffer")
    b.line("from kernels.constants import float_dtype")


def _emit_aliases(b: MojoBuilder, dim: int, kernel_type_str: str, num_params: int):
    """Emit compile-time aliases."""
    b.blank()
    b.line(f"alias DIM = {dim}")
    b.line(f"alias KernelType = {kernel_type_str}")
    b.line(f"alias NPARAMS = {num_params}")


def _emit_adapter(b: MojoBuilder, num_params: int):
    """Emit the JIT adapter struct implementing JITGradientProvider."""
    b.blank()
    b.comment("=" * 77)
    b.comment("JIT Adapter (implements JITGradientProvider)")
    b.comment("=" * 77)
    b.blank()
    b.line("struct JITAdapter(JITGradientProvider, Movable):")
    with b.block():
        b.line('"""JIT adapter: fused GPU kernels + CompositeProvider delegation."""')
        b.line("var provider: CompositeProvider[DIM, KernelType]")
        b.blank()

        # Constructor
        b.line(
            "fn __init__(out self, owned provider: CompositeProvider[DIM, KernelType]):"
        )
        with b.block():
            b.line("self.provider = provider^")
        b.blank()

        # Move constructor
        b.line("fn __moveinit__(out self, owned other: Self):")
        with b.block():
            b.line("self.provider = other.provider^")
        b.blank()

        # Forward matvec — delegates to CompositeProvider (fused kernels optional)
        b.comment("=== FORWARD MATVEC ===")
        b.line("fn forward_matvec(self, out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line(
            "    v_ptr: UnsafePointer[Float32, MutAnyOrigin], num_cols: Int) raises:"
        )
        with b.block():
            # Delegate to CompositeProvider (proven to work)
            # Fused GPU kernels can be wired in later for performance
            b.line("self.provider.forward_matvec(out_ptr, v_ptr, num_cols)")
        b.blank()

        # Fused gradient matvec
        b.comment("=== FUSED GRADIENT MATVEC ===")
        b.line(
            "fn fused_gradient_matvec(self, out_ptr: UnsafePointer[Float32, MutAnyOrigin],"
        )
        b.line(
            "    v_ptr: UnsafePointer[Float32, MutAnyOrigin], num_cols: Int) raises:"
        )
        with b.block():
            b.line("self.provider.fused_gradient_matvec(out_ptr, v_ptr, num_cols)")
        b.blank()

        # Delegated gradient_matvec
        b.line(
            "fn gradient_matvec(self, out_ptr: UnsafePointer[Float32, MutAnyOrigin],"
        )
        b.line("    v_ptr: UnsafePointer[Float32, MutAnyOrigin], num_cols: Int,")
        b.line("    param_index: Int, sync: Bool = True) raises:")
        with b.block():
            b.line(
                "self.provider.gradient_matvec(out_ptr, v_ptr, num_cols, param_index, sync)"
            )
        b.blank()

        # Simple trait methods
        b.line("fn num_gradient_params(self) -> Int: return NPARAMS")
        b.line("fn supports_fused_gradient(self) -> Bool: return True")
        b.line("fn supports_fused_ls_os(self) -> Bool: return False")
        b.line("fn supports_fused_3param(self) -> Bool: return False")
        b.blank()

        # Unsupported fused methods
        b.line(
            "fn fused_ls_os_gradient_matvec(self, ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],"
        )
        b.line("    os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line(
            "    v_ptr: UnsafePointer[Float32, MutAnyOrigin], num_cols: Int) raises:"
        )
        with b.block():
            b.line('raise Error("fused_ls_os not supported")')
        b.blank()

        b.line(
            "fn fused_3param_gradient_matvec(self, ls_out_ptr: UnsafePointer[Float32, MutAnyOrigin],"
        )
        b.line("    p1_out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line("    os_out_ptr: UnsafePointer[Float32, MutAnyOrigin],")
        b.line(
            "    v_ptr: UnsafePointer[Float32, MutAnyOrigin], num_cols: Int) raises:"
        )
        with b.block():
            b.line('raise Error("fused_3param not supported")')
        b.blank()

        # ForwardProvider delegation
        b.line("fn get_n(self) -> Int: return self.provider.n")
        b.line("fn get_ctx(self) -> DeviceContext: return self.provider.ctx")
        b.line("fn get_noise(self) -> Float32: return self.provider.noise")
        b.line("fn get_diagonal_value(self) -> Float32: return Float32(1.0)")
        b.line(
            "fn extract_diagonal(self, diag_ptr: UnsafePointer[Float32, MutAnyOrigin]) raises:"
        )
        with b.block():
            b.line("self.provider.extract_diagonal(diag_ptr)")
        b.line(
            "fn get_x_ptr(self) -> UnsafePointer[Float32, MutAnyOrigin]: return self.provider.x_ptr"
        )
        b.blank()

        # JITGradientProvider methods (param updates)
        b.line(
            "fn update_params(mut self, params_host_ptr: UnsafePointer[Float32, MutAnyOrigin]) raises:"
        )
        with b.block():
            b.line("self.provider.update_params(params_host_ptr)")
        b.line("fn update_noise(mut self, noise: Float32):")
        with b.block():
            b.line("self.provider.update_noise(noise)")


def _emit_train_binding(b: MojoBuilder):
    """Emit the train_python binding function."""
    b.blank()
    b.comment("=" * 77)
    b.comment("Python binding: train GP")
    b.comment("=" * 77)
    b.blank()
    b.line(
        "fn train_python(py_self: PythonObject, args: PythonObject) raises -> PythonObject:"
    )
    with b.block():
        b.line('"""Train GP with this JIT-compiled kernel.')
        b.line("")
        b.line("Args (from Python):")
        b.line("    args[0]: X numpy array (n, DIM) float32")
        b.line("    args[1]: y numpy array (n,) float32")
        b.line("    args[2]: max_iterations (int, optional, default=100)")
        b.line("    args[3]: learning_rate (float, optional, default=0.01)")
        b.line("    args[4]: verbose (bool, optional, default=False)")
        b.line('"""')
        b.line('var np = Python.import_module("numpy")')
        b.blank()
        b.line("var X_np = args[0]")
        b.line("var y_np = args[1]")
        b.line("var n = Int(X_np.shape[0].__int__())")
        b.blank()

        # Optional args
        b.line("var max_iterations = 100")
        b.line("var learning_rate = Float32(0.01)")
        b.line("var verbose = False")
        b.line("if len(args) > 2:")
        with b.block():
            b.line("max_iterations = Int(args[2].__int__())")
        b.line("if len(args) > 3:")
        with b.block():
            b.line("learning_rate = Float32(Float64(args[3]))")
        b.line("if len(args) > 4:")
        with b.block():
            b.line("verbose = Bool(args[4].__bool__())")
        b.blank()

        # Create buffers and copy data
        b.line("var ctx = DeviceContext()")
        b.line("var x_host = ctx.enqueue_create_host_buffer[float_dtype](n * DIM)")
        b.line("var y_host = ctx.enqueue_create_host_buffer[float_dtype](n)")
        b.line("ctx.synchronize()")
        b.blank()
        b.line("var X_c = np.ascontiguousarray(X_np, dtype=np.float32)")
        b.line("var y_c = np.ascontiguousarray(y_np, dtype=np.float32)")
        b.line("bulk_copy_to_host_buffer(X_c.ravel(), x_host, n * DIM)")
        b.line("bulk_copy_to_host_buffer(y_c, y_host, n)")
        b.blank()

        # Init params
        b.line("var params_host = ctx.enqueue_create_host_buffer[float_dtype](NPARAMS)")
        b.line("ctx.synchronize()")
        b.line("@parameter")
        b.line("for pi in range(NPARAMS):")
        with b.block():
            b.line("params_host[pi] = Float32(1.0)")
        b.blank()

        # Create provider and adapter
        b.line("var provider = CompositeProvider[DIM, KernelType](")
        b.line(
            "    ctx, x_host.unsafe_ptr(), params_host.unsafe_ptr(), n, Float32(0.1))"
        )
        b.line("var adapter = JITAdapter(provider^)")
        b.blank()

        # Train
        b.line("var result = train_jit_with_provider(")
        b.line("    adapter, ctx, y_host.unsafe_ptr(), n, NPARAMS,")
        b.line("    params_host.unsafe_ptr(), Float32(0.1),")
        b.line("    max_iterations=max_iterations,")
        b.line("    learning_rate=learning_rate,")
        b.line("    num_probes=10,")
        b.line("    max_cg_iter=100,")
        b.line("    cg_tol=Float32(1e-2),")
        b.line("    precond_rank=10,")
        b.line("    verbose=verbose,")
        b.line(")")
        b.blank()

        # Build return dict
        b.line("var out = Python.dict()")
        b.line('out["final_nll"] = Float64(result.final_nll)')
        b.line('out["noise"] = Float64(result.noise)')
        b.line('out["mean"] = Float64(result.mean)')
        b.line('out["iterations"] = result.iterations')
        b.line('out["converged"] = result.converged')
        b.blank()
        b.line("var params_list = Python.list()")
        b.line("for p in range(result.num_kernel_params):")
        with b.block():
            b.line("params_list.append(Float64(result.final_params[p]))")
        b.line('out["params"] = params_list')
        b.blank()

        # Keepalives
        b.line("_ = x_host")
        b.line("_ = y_host")
        b.line("_ = params_host")
        b.blank()
        b.line("return out")


def emit_module(
    kernel: IRKernel,
    schedule: ScheduleConfig,
    module_name: str,
    kernel_type_str: str = "",
    dim: int = 0,
    ard_imports: str = "",
) -> str:
    """Assemble a complete Mojo module source string.

    This is the main entry point for code generation. It combines all
    the pieces: imports, GPU kernels, adapter struct, train binding, and PyInit.
    """
    if dim == 0:
        dim = kernel.dim
    num_params = kernel.num_params

    b = MojoBuilder()

    # Module docstring
    b.line(f'"""Auto-generated MojoGP JIT kernel module.')
    b.blank()
    b.line(f"Kernel: {kernel_type_str}")
    b.line(f"Dimension: {dim}")
    b.line(f"Parameters: {num_params}")
    b.line(f"Schedule: TM={schedule.tm}, shmem={schedule.use_shmem}")
    b.line('"""')
    b.blank()

    # Imports
    _emit_imports(b, kernel, kernel_type_str, ard_imports)

    # Aliases
    _emit_aliases(b, dim, kernel_type_str, num_params)

    # GPU kernels (emitted as raw blocks since they manage their own indentation)
    b.raw(emit_forward_matvec(kernel, schedule))
    b.raw(emit_gradient_matvec(kernel, schedule))
    b.raw(emit_cross_matvec(kernel, schedule))
    b.raw(emit_extract_diagonal(kernel, schedule))

    # JIT adapter struct
    _emit_adapter(b, num_params)

    # Train binding
    _emit_train_binding(b)

    # PyInit
    b.blank()
    b.comment("=" * 77)
    b.comment("Module Initialization")
    b.comment("=" * 77)
    b.blank()
    b.line("@export")
    b.line(f"fn PyInit_{module_name}() -> PythonObject:")
    with b.block():
        with b.block("try:"):
            b.line(f'var m = PythonModuleBuilder("{module_name}")')
            b.line('m.def_py_function[train_python]("train", docstring="Train GP")')
            b.line("return m.finalize()")
        with b.block("except e:"):
            b.line(
                f'return abort[PythonObject]("Failed to init {module_name}: " + String(e))'
            )

    return b.build()

"""Shared Python-to-Mojo type conversion helpers.

Used by JIT engine binding files to avoid duplicating
the py_to_f32 conversion function.

Provides:
  - py_to_f32(): per-element Python-to-Float32 conversion (slow, for scalars)
  - bulk_copy_to_host_buffer(): numpy array -> HostBuffer via ctypes.memmove
  - bulk_copy_from_host_buffer(): HostBuffer -> numpy array via ctypes.memmove
  - bulk_copy_from_ptr(): UnsafePointer -> numpy array via ctypes.memmove
"""

from python import PythonObject, Python
from gpu.host import HostBuffer


fn py_to_f32(py_obj: PythonObject) raises -> Float32:
    """Convert a Python object to Float32.

    Uses Python's float() builtin to ensure we get a Python float,
    then converts to Mojo Float32.

    NOTE: This function is expensive (~5 Python interop calls per element).
    For bulk array copies, use bulk_copy_to_host_buffer() instead.
    """
    # Use Python's float() to convert to a Python float
    var builtins = Python.import_module("builtins")
    var py_float = builtins.float(py_obj)
    # Now convert to Mojo - use __index__ or direct conversion
    # The PythonObject should now be a Python float which can be converted
    var f64 = py_float.__float__()
    # Convert the Python float to Mojo Float64 using string parsing as fallback
    var str_val = String(py_float)
    return Float32(atof(str_val))


fn bulk_copy_to_host_buffer(
    np_array: PythonObject,
    host_buf: HostBuffer[DType.float32],
    num_elements: Int,
) raises:
    """Bulk copy from a contiguous float32 numpy array into a HostBuffer.

    Uses ctypes.memmove to perform a single memory copy instead of
    per-element Python-to-Mojo conversions. This is ~100-10000x faster
    than element-by-element py_to_f32() calls.

    IMPORTANT: The numpy array must be C-contiguous float32. Callers should
    ensure this with np.ascontiguousarray(arr, dtype=np.float32) before calling.

    Args:
        np_array: A C-contiguous float32 numpy array (1D or flattened 2D).
        host_buf: Destination HostBuffer with at least num_elements capacity.
        num_elements: Number of float32 elements to copy.
    """
    var ctypes = Python.import_module("ctypes")
    var dst_addr = Int(host_buf.unsafe_ptr())
    var src_addr = Int(np_array.ctypes.data)
    ctypes.memmove(dst_addr, src_addr, num_elements * 4)  # 4 bytes per float32


fn bulk_copy_from_host_buffer(
    host_buf: HostBuffer[DType.float32],
    num_elements: Int,
) raises -> PythonObject:
    """Bulk copy from a HostBuffer into a new numpy float32 array.

    Uses ctypes.memmove to perform a single memory copy instead of
    per-element Python list.append() calls. This is ~100-10000x faster.

    Args:
        host_buf: Source HostBuffer.
        num_elements: Number of float32 elements to copy.

    Returns:
        A 1D numpy float32 array of shape (num_elements,).
    """
    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var result = np.empty(num_elements, dtype=np.float32)
    var src_addr = Int(host_buf.unsafe_ptr())
    var dst_addr = Int(result.ctypes.data)
    ctypes.memmove(dst_addr, src_addr, num_elements * 4)
    return result


fn bulk_copy_from_ptr_addr(
    src_addr: Int,
    num_elements: Int,
) raises -> PythonObject:
    """Bulk copy from a raw memory address into a new numpy float32 array.

    Uses ctypes.memmove to perform a single memory copy instead of
    per-element Python list.append() calls.

    Args:
        src_addr: Source memory address (e.g., from Int(ptr)).
        num_elements: Number of float32 elements to copy.

    Returns:
        A 1D numpy float32 array of shape (num_elements,).
    """
    var np = Python.import_module("numpy")
    var ctypes = Python.import_module("ctypes")
    var result = np.empty(num_elements, dtype=np.float32)
    var dst_addr = Int(result.ctypes.data)
    ctypes.memmove(dst_addr, src_addr, num_elements * 4)
    return result

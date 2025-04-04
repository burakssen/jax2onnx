# file: jax2onnx/converter/utils.py

import numpy as np
from onnx import TensorProto
import jax.numpy as jnp

from jax.extend.core import Literal, Var

# Ensure all needed types are imported
from typing import TYPE_CHECKING, Callable

# Assuming these are correctly defined in your project:
from jax2onnx.converter.onnx_builder import OnnxBuilder


if TYPE_CHECKING:
    from jax2onnx.converter.converter import Jaxpr2OnnxConverter


def tensorproto_dtype_to_numpy(onnx_dtype: int) -> np.dtype:
    """
    Converts ONNX TensorProto data types to NumPy data types.
    """
    dtype_map = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.BOOL: np.bool_,
        TensorProto.INT8: np.int8,
        TensorProto.UINT8: np.uint8,
    }
    np_dtype = dtype_map.get(onnx_dtype)
    if np_dtype is None:
        print(
            f"Warning: Unsupported ONNX dtype {onnx_dtype} encountered in _tensorproto_dtype_to_numpy. Defaulting to np.float32."
        )
        return np.float32
    return np_dtype


def numpy_dtype_to_tensorproto(dtype):
    # If already an ONNX TensorProto dtype enum (int), just return
    if isinstance(dtype, int):
        return dtype

    try:
        np_dtype = np.dtype(dtype).type
    except TypeError:
        return TensorProto.FLOAT  # fallback

    return {
        np.float32: TensorProto.FLOAT,
        np.float64: TensorProto.DOUBLE,
        np.int32: TensorProto.INT32,
        np.int64: TensorProto.INT64,
        np.bool_: TensorProto.BOOL,
        np.int8: TensorProto.INT8,
        np.uint8: TensorProto.UINT8,
    }.get(np_dtype, TensorProto.FLOAT)


def _propagate_nested_functions(parent_builder: OnnxBuilder, sub_builder: OnnxBuilder):
    """
    Propagates nested ONNX functions from a sub-builder to a parent builder.
    Ensures nested functions are added only once.
    """
    for nested_func_name, nested_func_proto in sub_builder.functions.items():
        if nested_func_name not in parent_builder.functions:
            parent_builder.functions[nested_func_name] = nested_func_proto
            print(f"Propagated nested ONNX function: {nested_func_name}")


def function_handler(
    name: str, converter: "Jaxpr2OnnxConverter", eqn, orig_fn: Callable, params
):
    if orig_fn is None:
        raise RuntimeError(f"Original function for {name} not recorded.")

    from jax2onnx.plugin_system import get_qualified_name

    impl_key = get_qualified_name(orig_fn)
    print(f"Encountered function primitive: {impl_key}")

    instance_base_name = name.split(".")[-1]
    unique_node_name = converter.builder.get_unique_instance_name(instance_base_name)
    print(f"Generating unique ONNX node name: {unique_node_name}")

    parent_builder = converter.builder
    input_names = []
    example_args = []

    try:
        for var in eqn.invars:
            if isinstance(var, Var):
                aval = var.aval
                name = converter.get_name(var)
                input_names.append(name)
                example_args.append(
                    jnp.ones(aval.shape, dtype=aval.dtype)
                    if aval.shape
                    else jnp.zeros((), dtype=aval.dtype)
                )
                shape = tuple(aval.shape)
                dtype = aval.dtype
                parent_builder.register_value_info_metadata(name, shape, dtype)
            elif isinstance(var, Literal):
                example_args.append(var.val)
            else:
                raise TypeError(f"Unexpected input var type: {type(var)}")
    except Exception as e:
        print(f"Failed to prepare inputs for {impl_key}: {e}")
        raise

    unique_func_name = unique_node_name
    print(f"Tracing function body for: {unique_func_name}")

    sub_builder = OnnxBuilder(
        parent_builder.name_generator,
        parent_builder.opset,
        unique_func_name + "_graph",
        initializers=parent_builder.initializers,
    )
    sub_converter = converter.__class__(sub_builder)
    sub_converter.trace_jaxpr(orig_fn, example_args, preserve_graph=True)

    initializer_names = {i.name for i in parent_builder.initializers}
    used_constants = {
        inp
        for node in sub_builder.nodes
        for inp in node.input
        if inp in initializer_names
    }
    param_inputs = sorted(used_constants)
    print(f"Identified parameters (constants): {param_inputs}")

    sub_output_names = [vi.name for vi in sub_builder.outputs]

    # 🛡️ Validate all subgraph outputs have metadata
    for name in sub_output_names:
        if name not in sub_builder.value_info_metadata:
            raise RuntimeError(
                f"[❌] Subgraph output '{name}' is missing shape/type metadata. "
                f"Cannot register function '{unique_func_name}'."
            )

    internal_name = parent_builder.add_function(
        name=unique_func_name,
        sub_builder=sub_builder,
        param_input_names=param_inputs,
    )

    parent_builder.merge_value_info_metadata_from(sub_builder)

    for i, var in enumerate(eqn.outvars):
        original_parent_name = converter.get_var_name(var)
        parent_name = original_parent_name
        sub_name = sub_converter.get_name(var)

        print(
            f"[DEBUG] Mapping subgraph output '{sub_name}' → parent output '{parent_name}'"
        )

        if sub_name not in sub_builder.value_info_metadata:
            if hasattr(var, "aval"):
                shape = tuple(var.aval.shape)
                dtype = numpy_dtype_to_tensorproto(var.aval.dtype)
                sub_builder.register_value_info_metadata(
                    sub_name, shape, dtype, origin="repaired"
                )
                print(
                    f"[REPAIR] Injected missing shape/type for subgraph output '{sub_name}' using var.aval."
                )
            elif i < len(sub_output_names):
                fallback_sub_name = sub_output_names[i]
                print(
                    f"[FALLBACK] Using subgraph output by position: '{fallback_sub_name}'"
                )
                if fallback_sub_name != sub_name:
                    print(
                        f"[WARN] Output name mismatch: expected '{sub_name}', fallback used: '{fallback_sub_name}'"
                    )
                sub_name = fallback_sub_name

        if parent_name in parent_builder.value_info:
            continue

        shape_dtype = sub_builder.value_info_metadata.get(
            sub_name
        ) or parent_builder.value_info_metadata.get(parent_name)

        if not shape_dtype:
            if hasattr(var, "aval"):
                shape = tuple(var.aval.shape)
                dtype = numpy_dtype_to_tensorproto(var.aval.dtype)
                shape_dtype = (shape, dtype)
                print(f"[INFO] Recovered shape/type for {parent_name} from var.aval")

                actual = sub_builder.value_info_metadata.get(sub_name)
                if actual:
                    actual_shape, actual_dtype = actual
                    if (shape, dtype) != (actual_shape, actual_dtype):
                        print(
                            f"[WARN] Metadata mismatch for output '{parent_name}': "
                            f"subgraph has shape={actual_shape}, dtype={actual_dtype}; "
                            f"aval has shape={shape}, dtype={dtype}"
                        )

                if sub_name not in sub_builder.value_info_metadata:
                    sub_builder.register_value_info_metadata(
                        sub_name, shape, dtype, origin="recovered"
                    )

                parent_builder.register_value_info_metadata(
                    parent_name, shape, dtype, origin="recovered"
                )
            else:
                raise RuntimeError(
                    f"Output '{parent_name}' missing metadata and has no aval. Cannot infer shape/type."
                )
        else:
            shape, dtype = shape_dtype
            origin = None
            if sub_name in sub_builder.value_info_metadata_with_origin:
                _, _, origin = sub_builder.value_info_metadata_with_origin[sub_name]
            elif parent_name in parent_builder.value_info_metadata_with_origin:
                _, _, origin = parent_builder.value_info_metadata_with_origin[
                    parent_name
                ]

            parent_builder.register_value_info_metadata(
                parent_name, shape, dtype, origin=origin or "subgraph"
            )
            if origin == "repaired":
                print(
                    f"[INFO] Propagated repaired shape/type for '{parent_name}' from subgraph."
                )

        parent_builder.add_value_info(parent_name, shape, dtype)

    _propagate_nested_functions(parent_builder, sub_builder)
    print(f"Finished tracing function body: {unique_node_name}")

    call_inputs = input_names + param_inputs
    output_names = [converter.get_var_name(v) for v in eqn.outvars]

    parent_builder.add_function_call_node(
        function_name=unique_node_name,
        input_names=call_inputs,
        output_names=output_names,
        node_name=unique_node_name,
        user_display_name=name,
    )

    print(f"Added call node for: {internal_name}")

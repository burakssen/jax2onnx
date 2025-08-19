from typing import TYPE_CHECKING

import jax
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


@register_primitive(
    jaxpr_primitive=jax.lax.pow_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.pow.html",
    onnx=[
        {
            "component": "Pow",
            "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html",
        }
    ],
    since="v0.1.0",
    context="primitives.lax",
    component="pow",
    testcases=[
        {
            "testcase": "pow_test1",
            "callable": lambda x1, x2: x1 ** x2,
            "input_shapes": [(3,), (3,)],
        },
        {
            "testcase": "pow_test2",
            "callable": lambda x1, x2: x1 ** x2,
            "input_shapes": [(2, 2), (2, 2)],
        },
    ],
)
class PowPlugin(PrimitiveLeafPlugin):
    """Plugin for converting jax.lax.pow to ONNX Pow."""

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handle JAX pow primitive."""
        input_names = [s.get_name(inp) for inp in node_inputs]
        output_name = s.get_var_name(node_outputs[0])
        node = helper.make_node(
            "Pow",
            inputs=input_names,
            outputs=[output_name],
            name=s.get_unique_name("pow"),
        )
        s.add_node(node)

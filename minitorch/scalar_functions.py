from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """The forward pass simply adds the two inputs."""
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The derivative of the addition function is 1 for both inputs."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The forward pass computes the natural logarithm of the input."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the natural logarithm is 1/x."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class EQ(ScalarFunction):
    """Equality function $f(x) = $ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """The forward pass computes the equality of the two inputs."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """The derivative of the equality function is 0 for both inputs."""
        return 0.0, 0.0


class LT(ScalarFunction):
    """Less than function $f(x) = $ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """The forward pass computes the less than of the two inputs."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """The derivative of the less than function is 0 for both inputs."""
        return 0.0, 0.0


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The forward pass computes the exponential of the input."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the exponential function is e^x."""
        (a,) = ctx.saved_values
        return operators.mul(operators.exp(a), d_output)


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The forward pass computes the inverse of the input."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the inverse function is -1/x^2."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Mul(ScalarFunction):
    r"""Multiplication function $f(x, y) = x \times y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """The forward pass computes the product of the two inputs."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """The derivative of the multiplication function is y, x."""
        a, b = ctx.saved_values
        return operators.mul(b, d_output), operators.mul(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The forward pass computes the negation of the input."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the negation function is -1."""
        return operators.neg(d_output)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The forward pass computes the ReLU of the input."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the ReLU function is 1 if x > 0 else 0."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """The forward pass computes the sigmoid of the input."""
        sigmoid_val = operators.sigmoid(a)
        ctx.save_for_backward(sigmoid_val)
        return sigmoid_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """The derivative of the sigmoid function is sigmoid(x) * (1 - sigmoid(x))."""
        (sigmoid_val,) = ctx.saved_values
        return operators.mul(
            operators.mul(sigmoid_val, operators.add(1, (operators.neg(sigmoid_val)))),
            d_output,
        )

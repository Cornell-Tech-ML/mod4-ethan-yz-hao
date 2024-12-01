from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol, Dict, Set


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    x1 = list(vals)
    x2 = list(vals)
    x1[arg] = x1[arg] - epsilon
    x2[arg] = x2[arg] + epsilon
    return (f(*x2) - f(*x1)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Parents of this variable in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for this variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    order: List[Variable] = []
    visited: Set[int] = set()

    def dfs(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                dfs(parent)
        visited.add(var.unique_id)
        order.append(var)

    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable in the computation graph. The variable from which
                  the backpropagation begins.
        deriv: The derivative of the output with respect to the `variable`. This value will
               be propagated backward through the computation graph.

    Returns:
    -------
        None: The function writes computed derivatives to each leaf node using
            `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    queue = topological_sort(variable)
    derivatives: Dict[int, Any] = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives[v.unique_id] = derivatives.get(v.unique_id, 0.0) + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved values."""
        return self.saved_values

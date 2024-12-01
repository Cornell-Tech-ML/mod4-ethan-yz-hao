from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw
    
    # Step 1: Move width before height and reshape
    # batch x channel x height x width -> batch x channel x width x height
    # -> batch x channel x new_width x kw x height
    tiled = (input.permute(0, 1, 3, 2)
                 .contiguous()
                 .view(batch, channel, new_width, kw, height))
    
    # Step 2: Split height and rearrange to final shape
    # -> batch x channel x new_height x new_width x (kh * kw)
    tiled = (tiled.view(batch, channel, new_width, kw, new_height, kh)
                 .permute(0, 1, 4, 2, 5, 3)
                 .contiguous()
                 .view(batch, channel, new_height, new_width, kh * kw))
    
    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling
    Returns:
    -------
        Pooled tensor of size batch x channel x new_height x new_width
    """
    # TODO: Implement for Task 4.3.
    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    # Ensure correct output shape by explicitly reshaping after mean
    return tiled.mean(4).view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    max_values = max_reduce(input, dim)
    return (input == max_values).float()


# TODO: Implement for Task 4.4.
class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max should be max reduction"""
        # Convert dim tensor to int and compute max
        max_values = max_reduce(input, int(dim.item()))
        # Save both input and max values for backward
        ctx.save_for_backward(input, max_values)
        return max_values

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max should be argmax (see above)"""
        # Retrieve saved values
        input, max_values = ctx.saved_values
        # Create mask where input equals max values
        mask = max_values == input
        # Return gradient for input and dim
        return (grad_output * mask), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input, tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax as a tensor"""
    input_max = max_reduce(input, dim)
    shifted = input - input_max
    exp = shifted.exp()
    return exp / exp.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log of softmax using LogSumExp trick"""
    input_max = max_reduce(input, dim)
    shifted = input - input_max
    exp_sum = shifted.exp().sum(dim)
    return shifted - exp_sum.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D"""
    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    return max_reduce(tiled, 4).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise"""
    if ignore:
        return input
    
    # Handle edge case where rate = 1.0
    if rate >= 1.0:
        return input.zeros(input.shape)
    
    prob_tensor = rand(input.shape).detach() > rate
    scale = 1.0 / (1.0 - rate)
    return input * prob_tensor * scale

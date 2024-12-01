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


# TODO: Implement for Task 4.4.

import torch
from torch import Tensor
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns



def linear_q_with_scale_zero_point(tensor: Tensor, scale: Union[float, Tensor], zeropoint: int, dtype: torch.dtype = torch.int8):
    """
    Applies a linear quantization to the input Tensor

    Args:
        tensor: Torch.tensor (N, C, H, W), (C, H, W) or (H, W)
        scale: Same shape as input
        zero_point: int
        dtype: Target data type for quantization (torch.dtype)

    returns:
        A Tensor of the same dimension and shape of the input with values in the range of target quantization dtype.

    formula:
        r = s(q -z)
        r/s = q - z
        q = r/s + z
        q = int(round(r/s + z))

    """
    scaled_shifted_tensor =  tensor / scale + zeropoint

    rounded_tensor = torch.round(scaled_shifted_tensor)

    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max

    q_tensor = torch.clamp(rounded_tensor, min=q_min, max=q_max).to(dtype)

    return q_tensor

def linear_dequantization(quantized_tensor: Tensor, scale: Union[float, Tensor], zero_point: int):
    """
    Applies de-quantization to input tensor

    Args:
        quantized_tensor (Tensor): The input tensor.
        scale (Tensor or float): Scale that maps the values of the original tensor to the quantized tensor
        zeropint (int): Zero-point used for quantization
    
    returns:
        A tensor of same dimension and shape of the input and values in the range of torch.dtype float32

    formula:
        r = s(q - z)
    """
    return scale * (quantized_tensor.float() - zero_point)



def get_scale_and_zero_point(tensor: Tensor, dtype=torch.int8):

    """
    Get the scale and zero-point for the input Tensor, given target quantization dtype.

    args:
        tensor (tensor):


    formula:
        rmin = s(qmin - z)
        rmax = s(qmax - z)
        
        rmax - rmin = s(qmax -z) - (s(qmin -z))
        rmax - rmin = s(qmax - z - qmin + z)
                    = s(qmax - qmin)
        
        s = (rmax - rmin)/(qmax - qmin)

        rmin = s(qmin - z)
        rmin/s = qmin - z
        z = (qmin - rmin/s)
        z = int(round(qmin - rmin/s))
    """

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    r_min = torch.min(tensor).item()
    r_max = torch.max(tensor).item()

    scale = (r_max - r_min) / (q_max - q_min)

    zero_point = q_min - (r_min / scale)

    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else: 
        zero_point = int(round(zero_point))

    return scale, zero_point


def get_q_scale_symmetric(tensor: Tensor, dtype: torch.dtype):
    """
    Get the scale for linear quantization (symmetric mode)

    Args:
        tensor(Tensor): Input tensor
        dtype: Dtype for the quantized tensor

    Returns: scale

    """
    r_max = torch.max(torch.abs(tensor)).item()
    q_max = torch.iinfo(dtype).max
    return r_max / q_max

def linear_q_symmetric(tensor: Tensor, dtype: torch.dtype):
    """
    Applies linear quantization (symmetric mode) to the input tensor

    Args:
        tensor(Tensor): 
        dtype
    """
    scale = get_q_scale_symmetric(tensor, dtype)
    quantized_tensor = linear_q_with_scale_zero_point(tensor, scale=scale, zeropoint=0)
    return quantized_tensor, scale



def linear_q_symmetric_per_channel(r_tensor, dim, dtype=torch.int8):
    output_dim = r_tensor.shape[dim]
    scale = torch.zeros(output_dim)
    for index in range(output_dim):
        sub_tensor = torch.select(r_tensor, dim, index)
        scale[index] = get_q_scale_symmetric(sub_tensor, dtype)

    scale_shape = [1] * r_tensor.dim()
    scale_shape[dim] = -1
    scale = scale.view(scale_shape)

    quantized_tensor = linear_q_with_scale_zero_point(r_tensor, scale, zeropoint=0)

    return quantized_tensor, scale


def plot_matrix(tensor, ax, title, cmap=None):
    sns.heatmap(tensor.cpu().numpy(), annot=True, ax=ax, fmt='.2f', cbar=False, cmap=cmap)
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])



def plot_quantization_error(test_tensor, quantized_tensor, dequantized_tensor):

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))
    
    plot_matrix(test_tensor, axes[0], "test_tensor", "binary")
    plot_matrix(quantized_tensor, axes[1], "quantized_tensor", "binary")
    plot_matrix(dequantized_tensor, axes[2], "de-quantized_tensor", "binary")

    abs_error = torch.abs(test_tensor - dequantized_tensor)
    plot_matrix(abs_error, axes[3], "Quantization error", "coolwarm")
    fig.tight_layout()
    plt.show()

    
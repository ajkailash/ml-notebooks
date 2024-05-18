import torch
import matplotlib.pyplot as plt
import seaborn as sns


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

    
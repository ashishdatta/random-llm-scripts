import click
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import colored
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from typing import Literal

def create_scatter_plot(ax, sampled_2d, tokens_2d, frequencies, tokens):
    scatter = ax.scatter(sampled_2d[:, 0], sampled_2d[:, 1],
                        c=frequencies,
                        cmap='viridis',
                        alpha=0.6,
                        s=10)
    plt.colorbar(scatter, ax=ax, label='Token Frequency Rank')
    return ax

def create_contour_plot(ax, sampled_2d, tokens_2d, frequencies, tokens):
    x, y = sampled_2d[:, 0], sampled_2d[:, 1]
    H, xedges, yedges = np.histogram2d(x, y, bins=50, weights=frequencies)
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:])/2, (yedges[:-1] + yedges[1:])/2)
    contour = ax.contourf(X, Y, H.T, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='Token Density')
    return ax

def create_density_plot(ax, sampled_2d, tokens_2d, frequencies, tokens):
    x, y = sampled_2d[:, 0], sampled_2d[:, 1]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    density = ax.scatter(x, y, c=z, cmap='viridis', s=10)
    plt.colorbar(density, ax=ax, label='Point Density')
    return ax

def add_tokens_to_plot(ax, tokens_2d, tokens):
    for i, token in enumerate(tokens):
        ax.scatter(tokens_2d[i, 0], tokens_2d[i, 1],
                  color='black',
                  s=100,
                  zorder=5)
        ax.annotate(token,
                   (tokens_2d[i, 0], tokens_2d[i, 1]),
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=12,
                   color='black',
                   weight='bold')
    ax.grid(True, alpha=0.2, linestyle='-', color='gray')
    ax.set_aspect('equal', adjustable='box')
    return ax

@click.command()
@click.option('--model', required=True, help='Name or path of the model to analyze')
@click.option('--tokens', required=True, multiple=True, help='Tokens to analyze. Can be specified multiple times.')
@click.option('--baseline', default='the', help='Baseline token to compare against')
@click.option('--k', default=5, help='Number of nearest neighbors to show')
@click.option('--num_sample', default=1000, help='Number of random embeddings to sample for PCA visualization')
@click.option('--viz_type', type=click.Choice(['scatter', 'contour', 'density', 'all']), 
              default='scatter', help='Type of visualization to generate')
def main(model, tokens, baseline, k, num_sample, viz_type):
    # 1. Load tokenizer and model
    print(colored(f"Loading model: {model}", "blue"))
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    # Check baseline token first
    baseline_id = tokenizer.convert_tokens_to_ids(baseline)
    if baseline_id == tokenizer.unk_token_id:
        print(f"Baseline token '{baseline}' is not present in the tokenizer vocabulary")
        return

    # Load model only if we have valid baseline
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    embedding_matrix = model.get_input_embeddings().weight.data.to(device)
    baseline_embedding = embedding_matrix[baseline_id]
    baseline_norm = torch.norm(baseline_embedding, p=2).item()

    def find_nearest_neighbors(token_embedding, all_embeddings, k=5):
        token_embedding = token_embedding.unsqueeze(0)
        similarities = F.cosine_similarity(token_embedding, all_embeddings, dim=1)
        topk = torch.topk(similarities, k)
        topk_ids = topk.indices
        topk_values = topk.values

        results = []
        for idx, sim in zip(topk_ids, topk_values):
            token_str = tokenizer.convert_ids_to_tokens(idx.item())
            results.append((token_str, float(sim.item())))
        return results

    # Process each token
    for token in tokens:
        print(colored(f"\n=== Analyzing token: {token} ===", "blue"))
        
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            print(f"Token '{token}' is not present in the tokenizer vocabulary")
            continue

        token_embedding = embedding_matrix[token_id]
        token_norm = torch.norm(token_embedding, p=2).item()

        print(colored(f"Token norm: {token_norm:.10f}", "green"))
        print(colored(f"Baseline norm: {baseline_norm:.4f}", "green"))

        print(colored(f"\nNearest neighbors:", "cyan"))
        neighbors_token = find_nearest_neighbors(token_embedding, embedding_matrix, k)
        for token_str, sim_val in neighbors_token:
            print(f"  {token_str:20s} cos_sim={sim_val:.4f}")

    # Perform PCA visualization
    print(colored("\nGenerating PCA visualization...", "blue"))
    
    # Reset matplotlib style to default for clean look
    plt.style.use('default')
    
    # Sample random embeddings for visualization
    num_embeddings = embedding_matrix.shape[0]
    sample_indices = torch.randperm(num_embeddings)[:num_sample]
    sampled_embeddings = embedding_matrix[sample_indices].float().cpu().numpy()
    
    # Get token frequencies (using token IDs as proxy for frequency rank)
    frequencies = np.arange(num_sample)
    
    # Get embeddings for the specified tokens
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    token_embeddings = embedding_matrix[token_ids].float().cpu().numpy()
    
    # Combine sampled and token embeddings for PCA
    all_embeddings = np.vstack([sampled_embeddings, token_embeddings])
    
    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Split back into sampled and token embeddings
    sampled_2d = embeddings_2d[:num_sample]
    tokens_2d = embeddings_2d[num_sample:]
    
    # Create visualization based on selected type
    if viz_type == 'all':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
        
        # Create all three plots
        ax1 = create_scatter_plot(ax1, sampled_2d, tokens_2d, frequencies, tokens)
        ax1 = add_tokens_to_plot(ax1, tokens_2d, tokens)
        ax1.set_title('Scatter Plot View', pad=20)
        
        ax2 = create_contour_plot(ax2, sampled_2d, tokens_2d, frequencies, tokens)
        ax2 = add_tokens_to_plot(ax2, tokens_2d, tokens)
        ax2.set_title('Contour Plot View', pad=20)
        
        ax3 = create_density_plot(ax3, sampled_2d, tokens_2d, frequencies, tokens)
        ax3 = add_tokens_to_plot(ax3, tokens_2d, tokens)
        ax3.set_title('Density Estimation View', pad=20)
        
        fig.suptitle('Token Embedding Space - Multiple Views', fontsize=16, y=1.05)
    else:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # Create selected plot type
        if viz_type == 'scatter':
            ax = create_scatter_plot(ax, sampled_2d, tokens_2d, frequencies, tokens)
            title = 'Scatter Plot View'
        elif viz_type == 'contour':
            ax = create_contour_plot(ax, sampled_2d, tokens_2d, frequencies, tokens)
            title = 'Contour Plot View'
        else:  # density
            ax = create_density_plot(ax, sampled_2d, tokens_2d, frequencies, tokens)
            title = 'Density Estimation View'
        
        ax = add_tokens_to_plot(ax, tokens_2d, tokens)
        ax.set_title(title, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('token_embeddings_pca.png', dpi=300, bbox_inches='tight')
    print(colored(f"PCA visualization ({viz_type}) saved as 'token_embeddings_pca.png'", "green"))

if __name__ == '__main__':
    main()

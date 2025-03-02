import click
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import colored
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

@click.command()
@click.option('--model', required=True, help='Name or path of the model to analyze')
@click.option('--tokens', required=True, multiple=True, help='Tokens to analyze. Can be specified multiple times.')
@click.option('--baseline', default='the', help='Baseline token to compare against')
@click.option('--k', default=5, help='Number of nearest neighbors to show')
@click.option('--num_sample', default=1000, help='Number of random embeddings to sample for PCA visualization')
def main(model, tokens, baseline, k, num_sample):
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
    model = model.to(device)
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
    
    # Sample random embeddings for visualization
    num_embeddings = embedding_matrix.shape[0]
    sample_indices = torch.randperm(num_embeddings)[:num_sample]
    sampled_embeddings = embedding_matrix[sample_indices].cpu().numpy()
    
    # Get embeddings for the specified tokens
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    token_embeddings = embedding_matrix[token_ids].cpu().numpy()
    
    # Combine sampled and token embeddings for PCA
    all_embeddings = np.vstack([sampled_embeddings, token_embeddings])
    
    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Split back into sampled and token embeddings
    sampled_2d = embeddings_2d[:num_sample]
    tokens_2d = embeddings_2d[num_sample:]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot random sample points
    plt.scatter(sampled_2d[:, 0], sampled_2d[:, 1], alpha=0.1, color='gray', label='Random tokens')
    
    # Plot specified tokens with different colors
    colors = sns.color_palette("husl", len(tokens))
    for i, (token, color) in enumerate(zip(tokens, colors)):
        plt.scatter(tokens_2d[i, 0], tokens_2d[i, 1], color=color, s=100, label=token)
        plt.annotate(token, (tokens_2d[i, 0], tokens_2d[i, 1]), xytext=(5, 5), textcoords='offset points')
    
    plt.title('2D PCA Projection of Token Embeddings')
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('token_embeddings_pca.png')
    print(colored("PCA visualization saved as 'token_embeddings_pca.png'", "green"))

if __name__ == '__main__':
    main()

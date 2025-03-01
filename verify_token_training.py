import click
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from termcolor import colored

@click.command()
@click.option('--model', required=True, help='Name or path of the model to analyze')
@click.option('--tokens', required=True, multiple=True, help='Tokens to analyze. Can be specified multiple times.')
@click.option('--baseline', default='the', help='Baseline token to compare against')
@click.option('--k', default=5, help='Number of nearest neighbors to show')
def main(model, tokens, baseline, k):
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

if __name__ == '__main__':
    main()

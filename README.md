# Token Training Analyzer

A command-line tool to analyze whether tokens are well-trained in language models by examining their embedding space characteristics.

## Features

- Analyze multiple tokens at once
- Compare token embeddings with a baseline token
- Find nearest neighbors in the embedding space
- Check if tokens exist in the model's vocabulary
- Support for any HuggingFace transformer model

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install torch transformers click termcolor
```

## Usage

Basic usage:
```bash
python verify_token_training.py --model MODEL_NAME --tokens TOKEN1 --tokens TOKEN2
```

### Arguments

- `--model`: (Required) The name or path of the model to analyze (e.g., "gpt2", "facebook/opt-125m")
- `--tokens`: (Required) Tokens to analyze. Can be specified multiple times for multiple tokens
- `--baseline`: (Optional) Baseline token to compare against (default: "the")
- `--k`: (Optional) Number of nearest neighbors to show (default: 5)

### Examples

Analyze specific tokens:
```bash
# Check special tokens in GPT-2
python verify_token_training.py --model "gpt2" --tokens "<|endoftext|>" --tokens "<|pad|>"

# Check multiple tokens with custom baseline and more neighbors
python verify_token_training.py \
    --model "facebook/opt-125m" \
    --tokens "<s>" --tokens "</s>" --tokens "[PAD]" \
    --baseline "and" \
    --k 10
```

## Output

For each token, the script shows:
1. Whether the token exists in the model's vocabulary
2. The L2 norm of the token's embedding
3. Comparison with the baseline token's norm
4. K-nearest neighbors in the embedding space with cosine similarities

Example output:
```
Loading model: gpt2

=== Analyzing token: <|endoftext|> ===
Token norm: 0.1234567890
Baseline norm: 0.4567

Nearest neighbors:
  token1              cos_sim=0.9876
  token2              cos_sim=0.8765
  token3              cos_sim=0.7654
```

## Use Cases

- Verify if special tokens are properly trained in fine-tuned models
- Debug token embedding issues
- Analyze token relationships in the embedding space
- Check if custom tokens are properly integrated into the vocabulary

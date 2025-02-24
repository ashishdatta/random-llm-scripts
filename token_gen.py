import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TokenGenerator:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
    def generate_text(self, prompt: str, num_tokens: int) -> str:
        # Tokenize the input prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        total_tokens = 0
        generated_text = ""
        
        while total_tokens < num_tokens:
            # Generate tokens
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=min(50, num_tokens - total_tokens),  # Generate in smaller chunks
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            # Get the newly generated tokens
            new_tokens = outputs[0][len(input_ids[0]):]
            
            # Decode tokens to text, ignoring special tokens
            new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Update counters and text
            generated_text += new_text
            total_tokens += len(new_tokens)
            
            # Update input_ids for next iteration
            input_ids = outputs
            
            # Break if we've generated enough tokens
            if total_tokens >= num_tokens:
                break
                
        return generated_text

@click.command()
@click.option('--tokens', '-t', type=int, required=True, help='Number of tokens to generate')
@click.option('--prompt', '-p', type=str, default="default prompt", help='Initial prompt for generation')
def main(tokens: int, prompt: str):
    try:
        # Initialize generator
        generator = TokenGenerator()
        
        # Generate text
        click.echo(f"Generating {tokens} tokens...")
        generated_text = generator.generate_text(prompt, tokens)
        
        # Output results
        click.echo("\nGenerated Text:")
        click.echo("-" * 50)
        click.echo(generated_text)
        click.echo("-" * 50)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == "__main__":
    main()

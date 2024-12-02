import argparse
import torch
import transformers
from transformers import AutoTokenizer
from huggingface_hub import login


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--token',
        type=str,
        required=True,
        help="Huggingface token (must be approved for Llama)"
    )
    args = parser.parse_args()

    token = args.token
    login(token=token)

    model = "meta-llama/Llama-3.2-1B"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.float16},
        device=device
    )

    prompt = 'I have tomatoes, basil and cheese at home. What can I cook for dinner?'

    tokenizer = AutoTokenizer.from_pretrained(model)
    sequences = pipeline(
        prompt,
        # do_sample=True,
        top_k=10,
        num_return_sequences=5,
        eos_token_id=tokenizer.eos_token_id,
        # truncation=True,
        return_full_text=False,
        # max_length=512,
    )

    with open("llama_output.txt", "w") as f:
        f.write(prompt + "\n")
        for i, seq in enumerate(sequences):
            f.write(f"Result {i}:\n {seq['generated_text']}\n\n")
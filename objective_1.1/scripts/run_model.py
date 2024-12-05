import argparse
import torch
import transformers
from transformers import AutoTokenizer
from huggingface_hub import login
#import tensorflow as tf
#from tensorflow.python.client import device_lib
import csv

# Function to check GPU and CUDA status
# CUDA is a parallel computing platform and application programming interface model created by Nvidia
# MPS is the Metal Performance Shaders framework, which is a highly optimized GPU-accelerated framework for image and matrix math operations on Apple M series chips.
# CPU is fallback if no GPU is available
def check_gpu():
    print("CUDA Available: ", torch.cuda.is_available())
    if (torch.cuda.is_available()) : 
        print("CUDA Version: ", torch.version.cuda)
        print("Current Device: ", torch.cuda.current_device())
        print("Device Name: ", torch.cuda.get_device_name())
        print("Number of GPUs: ", torch.cuda.device_count())
    print("MPS Available: ", torch.mps.is_available())
    if (torch.mps.is_available()) : 
        print("Number of GPUs: ", torch.mps.device_count())
    #print("TensorFlow GPUs: ", tf.config.list_physical_devices('GPU'))
    #print("Local Devices: ", device_lib.list_local_devices())

# Function to generate text using the provided model and prompt
def generate_text(hf_token, model_name, text_prompt):
    # Login to Hugging Face
    login(hf_token)

    # Load the model and tokenizer
    #model_name = "meta-llama/Meta-Llama-3.1-8B"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(torch.mps.empty_cache())
        print(torch.mps.current_allocated_memory())
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()  # Clear the GPU cache
        print(torch.cuda.memory_summary())
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Initialize the text generation pipeline
    pipeline = transformers.pipeline(
        "text-generation", 
        model=model_name, 
        model_kwargs={"torch_dtype": torch.float16}, 
        device=device
    )

    if torch.backends.mps.is_available():
        print(torch.mps.current_allocated_memory())
    elif torch.cuda.is_available():
        print(torch.cuda.memory_summary())

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generate text based on the input prompt
    sequences = pipeline(
        text_prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        truncation=True,
        max_length=4000,
    )

    # Return an array of generated texts
    return [seq['generated_text'] for seq in sequences]

# Function to write the generated text to a TSV file
def write_to_tsv(filename, results):
    with open(filename, mode='w', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        for result in results:
            tsv_writer.writerow([result])

# Main function to handle arguments
def main():
    parser = argparse.ArgumentParser(description="Generate text using the Meta-Llama model.")
    parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face API token")
    parser.add_argument('--prompt', type=str, required=True, help="Text prompt to feed to the model")
    parser.add_argument('--output_file', type=str, required=True, help="Output TSV file to write the results")
    parser.add_argument('--model', type=str, required=False, help="Model name to use for text generation, the HugginFace model hub name. Defaults to 'meta-llama/Meta-Llama-3.1-8B'", default="meta-llama/Meta-Llama-3.1-8B")

    args = parser.parse_args()

    # Check GPU status
    check_gpu()

    # Generate text
    results = generate_text(args.hf_token, args.model, args.prompt)

    # Write results to a TSV file
    write_to_tsv(args.output_file, results)

if __name__ == "__main__":
    main()

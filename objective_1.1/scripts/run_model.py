import argparse
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM    
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
def generate_text(hf_token, model_name, text_prompt, top_k, max_length, num_return_seq, low_cpu_mem, dtype):

    # Login to Hugging Face
    login(hf_token)

    # Load the model and tokenizer, mps is Apple M-series, CUDA is Nvidia GPU, CPU is fallback
    # Switched to using accelerate which I think actually takes care of the device via the device_map=auto parameter
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(torch.mps.empty_cache())
        print(torch.mps.current_allocated_memory())
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()  # Clear the GPU cache
        torch.cuda.synchronize()
        print(torch.cuda.memory_summary())
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # check for the dtype, controls the precision of the model
    dtypeObj = torch.float16
    if (dtype == "bfloat16"):
        dtypeObj = torch.bfloat16
    elif (dtype == "float32"):
        dtypeObj = torch.float32
    elif (dtype == "float64"):
        dtypeObj = torch.float64

    # Load model and tokenizer with device_map=auto, Automatically splits the model across multiple devices (e.g., multiple GPUs or GPU + CPU) based on their available memory.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto", 
        torch_dtype=dtypeObj
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize the text generation pipeline
    pipeline = transformers.pipeline(
        "text-generation", 
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": dtypeObj, "low_cpu_mem_usage": low_cpu_mem}
    )

    if torch.backends.mps.is_available():
        print(torch.mps.current_allocated_memory())
    elif torch.cuda.is_available():
        print(torch.cuda.memory_summary())

    # Generate text based on the input prompt
    sequences = pipeline(
        text_prompt,
        do_sample=True,
        top_k=top_k,
        num_return_sequences=num_return_seq,
        eos_token_id=tokenizer.eos_token_id,
        truncation=True,
        max_length=max_length
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
    parser = argparse.ArgumentParser(description="Generate text using the Meta-Llama model")
    parser.add_argument('--token', type=str, required=True, help="Hugging Face API token")
    parser.add_argument('--prompt', type=str, required=True, help="Text prompt to feed to the model")
    parser.add_argument('--output-file', type=str, required=True, help="Output TSV file to write the results")
    parser.add_argument('--model', type=str, required=False, help="Model name to use for text generation, the HugginFace model hub name. Defaults to 'meta-llama/Meta-Llama-3.1-8B'", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument('--top-k', type=int, required=False, help="How many top probable tokens to consider", default=10)
    parser.add_argument('--max-length', type=int, required=False, help="What is the max length of the result", default=400)
    parser.add_argument('--num-return-seq', type=int, required=False, help="How many independently generated sequences should be returned", default=1)
    parser.add_argument('--low-cpu-mem', action='store_true', help="Sets the model argument low_cpu_mem_usage to True")
    parser.add_argument('--dtype', type=str, required=False, help="For specifying the precision (or data types) of tensors. Recognized values now include float16, bfloat16, float32, float64", default="float16")

    args = parser.parse_args()

    # Check GPU status
    check_gpu()

    # Generate text
    results = generate_text(args.token, args.model, args.prompt, args.top_k, args.max_length, args.num_return_seq, args.low_cpu_mem, args.dtype)

    # Write results to a TSV file
    write_to_tsv(args.output_file, results)

if __name__ == "__main__":
    main()

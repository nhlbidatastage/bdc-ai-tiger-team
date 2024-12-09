version 1.0

workflow HelloWorldWithDocker {
    input {
        String token
    }
    call run_python_script {
        input:
            token = token
    }

    output {
        File log = run_python_script.log_file
    }
}  
       
task run_python_script {

    input {
        String token
    }

    command <<<
        python3 <<CODE
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

    model = "meta-llama/Meta-Llama-3.1-8B"

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

CODE
            --token ${token}
       >>>

    output {      
        File log_file = "llama_output.txt"
    }
    runtime {    
        # Use this container, pull from DockerHub   
        docker: "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-2.py310" 
        memory: "16 GB"
        cpu: "8" 
        gpuType: "nvidia-tesla-v100"
        gpuCount: 1
        nvidiaDriverVersion: "418.87.00" # The official hugging face PyTorch gpu uses Nvidia drivers 470-471
        zones: ["us-central1-c"]  
    } 
}
version 1.0

workflow HelloHuggingFace {
    input {
        String token
        String prompt
        String model
        Int top_k
        Int max_length
        Int num_return
        String dtype
    }
    call run_python_script {
        input:
            token = token,
            prompt = prompt,
            model = model,
            top_k = top_k,
            max_length = max_length,
            num_return = num_return,
            dtype = dtype
    }

    output {
        File log = run_python_script.log_file
    }
}  
       
task run_python_script {

    input {
        String token
        String prompt
        String model
        Int top_k
        Int max_length
        Int num_return
        String dtype
    }

    command {
        python3 /run_model.py --token ${token} --prompt ${prompt} --output-file llama_output.txt --model ${model} --top-k ${top_k} --max-length ${max_length} --num-return-seq ${num_return} --dtype ${dtype}
    }

    output {      
        File log_file = "llama_output.txt"
    }
    runtime {     
        docker: "nimbusinformatics/bdc-ai-tiger-team:latest"
        memory: "64 GB"
        cpu: "8" 
        gpuType: "nvidia-tesla-v100"
        gpuCount: 2
        nvidiaDriverVersion: "418.87.00" # The official hugging face PyTorch gpu uses Nvidia drivers 470-471
        zones: ["us-central1-c"]  
        disks: "local-disk 100 SSD"
        bootDiskSizeGb: 50
    } 
}
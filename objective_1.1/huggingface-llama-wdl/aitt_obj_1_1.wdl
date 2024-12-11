version 1.0

workflow HelloWorldWithDocker {
    input {
        String token
        String prompt
        String model
    }
    call run_python_script {
        input:
            token = token,
            prompt = prompt,
            model = model 
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
    }

    command {
        python3 /run_model.py --hf_token ${token} --prompt ${prompt} --output_file llama_output.txt --model ${model}
    }

    output {      
        File log_file = "llama_output.txt"
    }
    runtime {     
        docker: "nimbusinformatics/bdc-ai-tiger-team:latest"
        memory: "64 GB"
        cpu: "8" 
        gpuType: "nvidia-tesla-v100"
        gpuCount: 1
        nvidiaDriverVersion: "418.87.00" # The official hugging face PyTorch gpu uses Nvidia drivers 470-471
        zones: ["us-central1-c"]  
        disks: "local-disk 100 SSD"
        bootDiskSizeGb: 50
    } 
}
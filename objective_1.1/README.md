# Objective 1.1

For a more in-depth description of what we're hoping to accomplish see our [Charter](https://docs.google.com/document/d/1JbukDIQj_M92IOuf0hiV2ZPqz3sGBQGK5vmzfNirJzk/edit)

The basics are 1) get a "hello world" working in Terra/SBG notebook environments for a model pulled from HuggingFace and 
2) take that proof of concept a bit farther by tuning the model with content (open access) from BDC.

## Experiment 1 - Notebook on SBG (blocked)

Basic “hello world” for [Llama 3](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) in SBG notebooks

As of 20241021 this experiment is on pause since the SBG Notebook environment image does not have the CUDA drivers
installed properly. See the [notes](https://docs.google.com/document/d/1S1aVbmvZ4wiXFc2YzGRdIwYfvgh4hAGFdjlIwAzD9wU/edit?tab=t.0#bookmark=id.e0tv2cb680e4) from today's AI hackathon session for more detail.  David Roberson reports the images won't be updated until Q1 2025.  However, CWL execution can make use of CUDA so we'll try again in experiment 3 below.

## Experiment 2 - Notebook on Terra (working!)

Basic “hello world” for [Llama 3](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) in Terra notebooks.

Alisa Manning has created a couple notebooks to explore Llama3 in Terra notebooks:
* [Running_Llama_on_HF_transformers.ipynb](Running_Llama_on_HF_transformers.ipynb) 
* ['llama test.ipynb'](llama%20test.ipynb)

_These are not yet working, use the notebook below from Brian for a working version_

Brian tried a different base image and GPU type and was able to get a basic "hello world" working (prompt for Llama3 to create a recipe).

The environment was:
<img width="522" alt="image" src="https://github.com/user-attachments/assets/88fa4db3-7d28-4f3c-9ded-ee297610f35f">

**Brian was then able to succesfully run this notebook**:
* ['20241021 Llama3.ipynb'](20241021%20Llama3.ipynb)

The notebook itself has details about dependency, account setup, etc.

Some things to consider:
* I needed to use the NVIDIA Tesla V100 GPU (I tried 2, not sure if 2 are required)
* When I tried the NVIDIA T4 I ran out of GPU memory
* I used 8 CPUs for 52GB of RAM, not certain if this much RAM is needed
* You need a valid Huggingface token
* You need to apply for access to the Llama 3 model via their Huggingface page

## Experiment 3 - Docker + Workflow (working on SBG, Terra in progress) <-- we are here...

Once we had a working hello world above we:
* created a Docker image 
* wrote a wrapper script to call the model 
* wrote the same workflow as the notebook but in both CWL and WDL
* show we can run these in their respective environments, CWL is working on SBG 

### Building Docker Image Locally

Build from the `objective_1.1` directory (you can drop `--no-cache` if you're building this frequently):

```bash
docker build --no-cache -t bdc-ai-tiger-team -f docker/Dockerfile . 
```

See below "Hosting the Docker Image on DockerHub" for more details on building and pushing to DockerHub

### Script

We created a Python script to do the heavy lifting on behind the CWL/WDL workflows.

  scripts/run_model.py

You would run this script with:

```bash
python3 run_model.py --hf_token <your_hf_token> --prompt "I have tomatoes, basil and cheese at home. What can I cook for dinner?" --output_file output.tsv --model meta-llama/Meta-Llama-3.1-8B
```

### Testing Locally

You can actually run this locally, I tested on a Mac M2 Ultra GPU as well as CPU (no acceleration) via Docker. It should automatically detect an NVIDIA GPU if you are running a Linux/Windows host with that GPU type with PyTorch correctly configured.

#### Mac M-Series Processor

I used Anaconda to install depdencies and setup an environment to keep this work isolated.

```bash
# create a clean conda environment for this work
conda create -n llama-testing python=3.10 -y
# activate it
conda activate llama-testing
# install python dependencies 
pip3 install torch torchvision torchaudio; pip3 install transformers; pip3 install tensorflow
# now run the script, this should automatically use the M2 GPU cores (you can see this in Activity Monitor)
python scripts/run_model.py --hf_token <your_hf_token> --prompt "I have tomatoes, basil and cheese at home. What can I cook for dinner?" --output_file output.tsv --model meta-llama/Meta-Llama-3.1-8B
```

This should run pretty quickly and use the GPU cores on the M-series processor.

#### Via Docker on Mac (or other platform)

This approach uses Docker, so the GPU acceleration won't be used.  Rather the CPUs will be used.

```bash
# build the docker using the 'docker build' command above 
# launch into the docker container
docker run -it bdc-ai-tiger-team /bin/bash
# within that Docker container, run the python script
root@68f713819d68:/# python3 run_model.py --hf_token <your_hf_token> --prompt "I have tomatoes, basil and cheese at home. What can I cook for dinner?" --output_file output.tsv --model meta-llama/Meta-Llama-3.1-8B
```

This runs more slowly since it's using the CPU instead.  It also re-downloads the model files each time you build and run it.

### Hosting the Docker Image on DockerHub

For right now, I'm hosting the docker image at [nimbusinformatics/bdc-ai-tiger-team](https://hub.docker.com/repository/docker/nimbusinformatics/bdc-ai-tiger-team/general)

This is not ideal since 1) this is not an official NHLBI Docker repository and 2) I can't build the Docker image directly
from GitHub (and trigged by checkins in this repo).  The latter is because my Github user does not have permissions to 
setup the Dockerhub integration with GitHub for the NHLBIDataStage organization.

That being said, here's now I built the image locally (I'm on an M2 Mac) with multi-platform buidling enabled on Docker:

```bash
cd bdc-ai-tiger-team/objective_1.1
docker buildx create --use
docker buildx inspect --bootstrap
# to build locally amd64
docker buildx build --platform linux/amd64 -t bdc-ai-tiger-team:latest -f docker/Dockerfile --load .
# to build locally arm64
docker buildx build --platform linux/arm64 -t bdc-ai-tiger-team:latest -f docker/Dockerfile --load .
# to build and push to the DockerHub repo
docker buildx build --platform linux/amd64,linux/arm64 -t nimbusinformatics/bdc-ai-tiger-team:latest -f docker/Dockerfile --push .
```

You can add `--no-cache` if you like to make sure the build is totally fresh.

### CWL

See the CWL in `huggingface-llama-cwl/huggingface_test-tool-0.2.cwl`.  This has been reported to work on SBG.

### WDL

See the WDL in `huggingface-llama-wdl/aitt_obj_1_1.wdl`, this isn't working yet.


LEFT OFF WITH: the job failed, looks like CUDA is setup correctly and the model is downloading.  However I think there may not be enough system memory?  I bumped it to 64GB to be safe.  Details of the V100 instance types here: https://cloud.google.com/compute/docs/gpus

```
2024/12/11 08:50:49 Starting container setup.
2024/12/11 08:50:51 Done container setup.
2024/12/11 08:50:55 Starting localization.
2024/12/11 08:51:13 Localization script execution started...
2024/12/11 08:51:13 Localizing input gs://fc-8fc8bb19-39e2-4911-9d56-1643a820d931/submissions/ed03b643-f879-4e82-8b3e-d4778705dda0/HelloWorldWithDocker/97de4541-d093-4f76-af07-079cbe35afcb/call-run_python_script/script -> /cromwell_root/script
2024/12/11 08:51:18 Localization script execution complete.
2024/12/11 08:51:22 Done localization.
2024/12/11 08:51:22 Running user action: docker run -v /mnt/local-disk:/cromwell_root --entrypoint=/bin/bash nimbusinformatics/bdc-ai-tiger-team@sha256:4130a23af3e8234d369e1143518b8bf106bf5e83fe7551177141b03f0598fb17 /cromwell_root/script
CUDA Available:  True
CUDA Version:  12.1
Current Device:  0
Device Name:  Tesla V100-SXM2-16GB
Number of GPUs:  1
MPS Available:  False
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Active memory         |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Requested memory      |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| GPU reserved memory   |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Non-releasable memory |      0 B   |      0 B   |      0 B   |      0 B   |
|       from large pool |      0 B   |      0 B   |      0 B   |      0 B   |
|       from small pool |      0 B   |      0 B   |      0 B   |      0 B   |
|---------------------------------------------------------------------------|
| Allocations           |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Active allocs         |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| GPU reserved segments |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |       0    |       0    |       0    |       0    |
|       from large pool |       0    |       0    |       0    |       0    |
|       from small pool |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|

Using device: cuda

Downloading shards:   0%|          | 0/4 [00:00<?, ?it/s]
Downloading shards:  25%|██▌       | 1/4 [03:07<09:13, 184.41s/it]
Downloading shards:  50%|█████     | 2/4 [09:49<10:28, 314.02s/it]
Downloading shards:  75%|███████▌  | 3/4 [16:33<05:55, 355.18s/it]
Downloading shards: 100%|██████████| 4/4 [18:02<00:00, 250.26s/it]
Downloading shards: 100%|██████████| 4/4 [18:02<00:00, 270.68s/it]

Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▌       | 1/4 [10:49<32:28, 649.56s/it]
Loading checkpoint shards:  50%|█████     | 2/4 [21:51<21:52, 656.40s/it]
Loading checkpoint shards:  75%|███████▌  | 3/4 [32:03<10:36, 636.22s/it]/cromwell_root/script: line 27:    15 Killed                  python3 /run_model.py --hf_token <TOKEN> --prompt 'I have tomatoes, basil and cheese at home. What can I cook for dinner?' --output_file llama_output.txt --model meta-llama/Meta-Llama-3.1-8B
2024/12/11 09:55:13 Starting delocalization.
2024/12/11 09:55:16 Delocalization script execution started...
2024/12/11 09:55:16 Delocalizing output /cromwell_root/memory_retry_rc -> gs://fc-8fc8bb19-39e2-4911-9d56-1643a820d931/submissions/ed03b643-f879-4e82-8b3e-d4778705dda0/HelloWorldWithDocker/97de4541-d093-4f76-af07-079cbe35afcb/call-run_python_script/memory_retry_rc
2024/12/11 09:55:25 Delocalizing output /cromwell_root/rc -> gs://fc-8fc8bb19-39e2-4911-9d56-1643a820d931/submissions/ed03b643-f879-4e82-8b3e-d4778705dda0/HelloWorldWithDocker/97de4541-d093-4f76-af07-079cbe35afcb/call-run_python_script/rc
2024/12/11 09:55:26 Delocalizing output /cromwell_root/stdout -> gs://fc-8fc8bb19-39e2-4911-9d56-1643a820d931/submissions/ed03b643-f879-4e82-8b3e-d4778705dda0/HelloWorldWithDocker/97de4541-d093-4f76-af07-079cbe35afcb/call-run_python_script/stdout
2024/12/11 09:55:28 Delocalizing output /cromwell_root/stderr -> gs://fc-8fc8bb19-39e2-4911-9d56-1643a820d931/submissions/ed03b643-f879-4e82-8b3e-d4778705dda0/HelloWorldWithDocker/97de4541-d093-4f76-af07-079cbe35afcb/call-run_python_script/stderr
2024/12/11 09:55:29 Delocalizing output /cromwell_root/llama_output.txt -> gs://fc-8fc8bb19-39e2-4911-9d56-1643a820d931/submissions/ed03b643-f879-4e82-8b3e-d4778705dda0/HelloWorldWithDocker/97de4541-d093-4f76-af07-079cbe35afcb/call-run_python_script/llama_output.txt
Required file output '/cromwell_root/llama_output.txt' does not exist.
```

### NEXT STEPS

* need to converge both CWL and WDL on the same script/Dockerfile, we have a single Docker but it needs to be tested/re-tested on both SBG and Terra
* need to host the Dockerfile in a location both SBG/Terra environments can pull from (e.g. Dockerhub) -- done
* need to confirm the WDL works on Terra (and CWL continues to work on SBG) -- in progress now


## Experiment 4

Tuning of the model based on “Discovery page” content in BDC.  We’ll use this [tutorial](https://www.datacamp.com/tutorial/llama3-fine-tuning-locally) as a jumping off point.  More to come on this.



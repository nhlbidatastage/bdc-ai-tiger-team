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

### Dockerfile

Build from the `objective_1.1` directory (you can drop `--no-cache` if you're building this frequently):

```bash
docker build --no-cache -t llama-testing -f docker/Dockerfile . 
```

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
docker run -it llama-testing /bin/bash
# within that Docker container, run the python script
root@68f713819d68:/# python3 run_model.py --hf_token <your_hf_token> --prompt "I have tomatoes, basil and cheese at home. What can I cook for dinner?" --output_file output.tsv --model meta-llama/Meta-Llama-3.1-8B
```

This runs more slowly since it's using the CPU instead.  It also re-downloads the model files each time you build and run it.

### Hosting the Docker Image

TODO: we need to host the docker image somewhere others can pull it.  For now you'll have to build and host it yourself.

### CWL

See the CWL in `huggingface-llama-cwl/huggingface_test-tool-0.2.cwl`.  This has been reported to work on SBG.

### WDL

See the WDL in `huggingface-llama-wdl/aitt_obj_1_1.wdl`, this isn't working yet.

### NEXT STEPS

* need to converge both CWL and WDL on the same script/Dockerfile
* need to host the Dockerfile in a location both SBG/Terra environments can pull from (e.g. Dockerhub)
* need to confirm the WDL works on Terra (and CWL continues to work on SBG)

## Experiment 4

Tuning of the model based on “Discovery page” content in BDC.  We’ll use this [tutorial](https://www.datacamp.com/tutorial/llama3-fine-tuning-locally) as a jumping off point.  More to come on this.

# Objective 1.1

For a more in-depth description of what we're hoping to accomplish see our [Charter](https://docs.google.com/document/d/1JbukDIQj_M92IOuf0hiV2ZPqz3sGBQGK5vmzfNirJzk/edit)

The basics are 1) get a "hello world" working in Terra/SBG notebook environments for a model pulled from HuggingFace and 
2) take that proof of concept a bit farther by tuning the model with content (open access) from BDC.

## Experiment 1

Basic “hello world” for [Llama 3](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) in SBG notebooks

As of 20241021 this experiment is on pause since the SBG Notebook environment image does not have the CUDA drivers
installed properly. See the [notes](https://docs.google.com/document/d/1S1aVbmvZ4wiXFc2YzGRdIwYfvgh4hAGFdjlIwAzD9wU/edit?tab=t.0#bookmark=id.e0tv2cb680e4) from today's AI hackathon session for more detail.  David Roberson reports the images won't be updated until Q1 2025.  However, CWL execution can make use of CUDA so we'll try again in experiment 3 below.

## Experiment 2

Basic “hello world” for [Llama 3](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) in Terra notebooks.

Alisa Manning has created a couple notebooks to explore Llama3 in Terra notebooks:
* [Running_Llama_on_HF_transformers.ipynb]
* [llama test.ipynb]

These are not yet working

Brian tried a different base image and GPU type and was able to get a basic "hello world" working (prompt for Llama3 to create a recipe).

The environment was:
<img width="522" alt="image" src="https://github.com/user-attachments/assets/88fa4db3-7d28-4f3c-9ded-ee297610f35f">

I was then able to succesffully run this notebook:
* [Uploading 20241021 Llama3.ipynb…]()


## Experiment 3

## Experiment 4

Tuning of the model based on “Discovery page” content in BDC.  We’ll use this [tutorial](https://www.datacamp.com/tutorial/llama3-fine-tuning-locally) as a jumping off point.

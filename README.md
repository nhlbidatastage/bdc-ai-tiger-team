# bdc-ai-tiger-team

A GitHub repository for the AI Tiger Team, see charter [here](https://docs.google.com/document/d/1JbukDIQj_M92IOuf0hiV2ZPqz3sGBQGK5vmzfNirJzk/edit)

The Tiger Team is organized into sub-objectives, documented in the Charter.  Each sub-objective has a folder in this repository where code for that objective is stored.  We also created Tutorials that will be directly useful for researchers, see the links below.

## Objective 1.1

The goal for objective 1.1 is to see if we can replicate a simple AI-based analysis in Terra & SBG workspaces.  From the charter:

**Who**: Brian O’Connor, David Beaumont, Matt Satusky, Ashok Krishnamurthy, Alisa Manning [for Terra workspace setup], Steven Guo, and others please add yourself here

**What**: As a researcher I want to use the Llama 3 model from HuggingFace to generate answers to questions I have about the data available on BioData Catalyst.  Specifically, I’d like to use a Jupyter notebook in Terra/Velsera to pull the text descriptions and other information from the Gen3 Discovery service (which is open access).  I will then use these descriptions to fine-tune the model and then ask it to generate a response that will hopefully then direct me to datasets of interest given certain criteria.  

**Where**: Terra/Velsera workspace environments running in Jupyter

**Why**: This will demonstrate if 1) our workspace environments can be used to perform the full breadth of work described in the fine-tuning tutorial above and 2) a researcher can use a similar approach to explore BDC data to answer their hypothesis generation questions.  An outcome will be a tutorial on how this was done in BDC and the required resources used (or, if not feasible, why not).

For more details see the Objective 1.1 [README.md](objective_1.1/README.md)

## Tutorials

Object 1.1 allowed us to explore the AI capabilities of the NHLBI BDC system.  As an outcome of this work 
we developed two tutorials that can be jumping off points for researchers that want to use AI models,
such as the Llama3 LLM model, in their research via the BDC environment.

## Tutorial 1 - Hello Llama

This [tutorial](tutorials/tutorial_1_hello_llama/README.md) walks you through doing a very basic interaction with 
the Llama3 LLM on the BDC environment.  We show you how to do this interactively via a notebook on the Terra environment
and in a workflow on Terra and also Velsera.  The notebook approach is great for walking through the code and how
the whole process works while the workflows show how you might scale up and parallelize running models like Llama3.
While the actual demonstration we do is trivial (asking Llama3 to come up with a recipe given ingredients), you 
can use this as a jumping off point for interacting with a sophisticated LLM within our secure environment.

## Tutorial 2 - RAG Llama

This [tutorial](tutorials/tutorial_2_rag_llama/README.md) picks up where the Hello Llama tutorial leaves off.  Given the domain-specific information most
researchers will work with, there's a desire to provide contextual information to models like Llama3 in order
to improve the quality of the response.  For example, you may want the LLM to have background information on 
your area of research, terminology used, and scientific experimental designs in order to generate 
a useful response to your question.  Techniques such as fine-tuning and RAG provide ways to provide 
contextual information.  In this tutorial we use RAG since it's a very easy process that is highly adaptable.
In this case we're using an open source database of patient/physician conversations to ask a medical question, however
this is just a stand-in for more complex and realistic uses.
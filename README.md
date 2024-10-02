# bdc-ai-tiger-team

A GitHub repository for the AI Tiger Team, see charter [here](https://docs.google.com/document/d/1JbukDIQj_M92IOuf0hiV2ZPqz3sGBQGK5vmzfNirJzk/edit)

The Tiger Team is organized into sub-objectives, documented in the Charter.

Each sub-objective has a folder in this repository where code for that objective is stored.

## Objective 1.1

The goal for objective 1.1 is to see if we can replicate a simple AI-based analysis in Terra & SBG workspaces.  From the charter:


**Who**: Brian O’Connor, David Beaumont, Matt Satusky, Ashok Krishnamurthy, Alisa Manning [for Terra workspace setup], Steven Guo, and others please add yourself here

**What**: As a researcher I want to use the Llama 3 model from HuggingFace to generate answers to questions I have about the data available on BioData Catalyst.  Specifically, I’d like to use a Jupyter notebook in Terra/Velsera to pull the text descriptions and other information from the Gen3 Discovery service (which is open access).  I will then use these descriptions to fine-tune the model and then ask it to generate a response that will hopefully then direct me to datasets of interest given certain criteria.  

**Where**: Terra/Velsera workspace environments running in Jupyter

**Why**: This will demonstrate if 1) our workspace environments can be used to perform the full breadth of work described in the fine-tuning tutorial above and 2) a researcher can use a similar approach to explore BDC data to answer their hypothesis generation questions.  An outcome will be a tutorial on how this was done in BDC and the required resources used (or, if not feasible, why not).
# Tutorial 1 - Hello Llama

## LLMs and You

As an NHLBI researcher, you can leverage an LLM model like Llama3 in BDC to accelerate data analysis, hypothesis generation, and literature review. By training or fine-tuning the model on specialized datasets, such as genomic, proteomic, or clinical trial data, researchers like you can query and synthesize complex relationships between genes, pathways, and diseases like heart failure, pulmonary hypertension, or sickle cell anemia. The model can also assist in natural language tasks like summarizing large volumes of scientific literature, generating insights from unstructured clinical data, or creating tailored communications materials. Additionally, an LLM running in the secure BDC environment ensures data privacy and compliance with regulations when working with sensitive or proprietary datasets, making it a valuable tool in advancing personalized medicine and translational research in cardiovascular, pulmonary, and hematological domains.

## About this Tutorial

This tutorial picks up where the Hello Llama tutorial leaves off.  Given the domain-specific information most
researchers will work with, there's a desire to provide contextual information to models like Llama3 in order
to improve the quality of the response.  For example, you may want the LLM to have background information on 
your area of research, terminology used, and scientific experimental designs in order to generate 
a useful response to your question.  Techniques such as fine-tuning and RAG provide ways to provide 
contextual information.  In this tutorial we use RAG since it's a very easy process that is highly adaptable.
In this case we're using an open source database of patient/physician conversations to ask a medical question, however
this is just a stand-in for more complex and realistic uses.

TODO: need to finish this tutorial.  in the meantime you can check out our [README](../../objective_1.1/README.md) which documents this 
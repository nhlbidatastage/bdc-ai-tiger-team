from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from huggingface_hub import login
import torch
import argparse
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import csv


# Function to write the generated text to a TSV file
def write_to_tsv(filename, results):
    with open(filename, mode='w', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        for result in results:
            tsv_writer.writerow([result])

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
def do_rag(hf_token, lc_token, model_name, text_prompt, top_k, max_length, num_return_seq, low_cpu_mem, dtype):
    # Login to Hugging Face
    login(hf_token)

    # load dataset
    # Specify the dataset name
    dataset_name = "ruslanmv/ai-medical-chatbot"

    # Create a loader instance using dataset columns
    loader_doctor = HuggingFaceDatasetLoader(dataset_name,"Doctor")

    # Load the data
    print("Loading data...")
    doctor_data = loader_doctor.load()

    # Select the first 100 entries
    # here's where I can reduce down... need to param this
    doctor_data = doctor_data[:1000]

    print(doctor_data[:2])

    # Define the path to the embedding model
    modelPath = "sentence-transformers/all-MiniLM-L12-v2"

    # GPU acceleration
    model_kwargs = {'device':'mps'}

    # Create a dictionary with encoding options
    encode_kwargs = {'normalize_embeddings': False}

    print("Initialize an instance of HuggingFaceEmbeddings with the specified parameters...")
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )
    text = "Why are you a doctor?"
    query_result = embeddings.embed_query(text)
    print(query_result[:3])

    print("Create an instance of FAISS...")
    vector_db = FAISS.from_documents(doctor_data, embeddings)
    vector_db.save_local("faiss_doctor_index")

    print("Perform a similarity search with the question...")
    question = "Hi Doctor, I have a headache, help me."
    top_k = 30
    searchDocs = vector_db.similarity_search(question, top_k)
    print(searchDocs[0].page_content)

    print("Create a retriever instance...")
    retriever = vector_db.as_retriever()
    #base_model = "/kaggle/input/llama-3/transformers/8b-chat-hf/1"
    #base_model = "meta-llama/llama-3-8b-chat-hf"
    #base_model = "meta-llama/Meta-Llama-3.1-8B"
    base_model = "meta-llama/Llama-3.2-1B-Instruct"

    print("Initialize the model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print("Load the model...")
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
    )

    print("Initialize the text generation pipeline...")
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=120
    )

    print("Create a pipeline instance...")
    # llm = OpenAI(api_key=openai_api_key, model="gpt-4")
    llm = HuggingFacePipeline(pipeline=pipe)

    print("Create a retriever instance...")
    rag_prompt = hub.pull("rlm/rag-prompt")

    print("Create a QA chain...")
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    print("Preparing question for the QA chain...")
    question = "Hi Doctor, I have a headache, help me."
    print("Question: "+question)

    print("Invoke the QA chain...")
    result = qa_chain.invoke(question)
    print(result)

    print("Return the answer...")
    print(result.split("Answer: ")[1])
    return(result.split("Answer: "))


# Main function to handle arguments
def main():
    parser = argparse.ArgumentParser(description="Generate text using RAG with the Meta-Llama model")
    parser.add_argument('--token', type=str, required=True, help="Hugging Face API token")
    parser.add_argument('--lc-token', type=str, required=True, help="LangChain API token")
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
    results = do_rag(args.token, args.lc_token, args.model, args.prompt, args.top_k, args.max_length, args.num_return_seq, args.low_cpu_mem, args.dtype)

    # Write results to a TSV file
    write_to_tsv(args.output_file, results)

if __name__ == "__main__":
    main()
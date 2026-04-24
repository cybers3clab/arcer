import os
import requests
import yaml
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def _set_env():
    load_dotenv()
    required_vars = [
        "OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN"
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


def main():
    arcer = arcer()
    
    # setup the configuration of the thread used for this conversation with arcer
    config = {"configurable": {"thread_id":"thread001"}}

    #setup the user input messages
    messages = []

    #m1 = """Hi, I want to create a CyRIS-based cyber range with id '1' running on the host 'master_ws' on 'localhost' address. Use the virtual bridge at 192.168.122.1 for the communication among the host and the cyber range VMs. The account on the host is 'host_user'.
    #Create an CentOS-based VM called 'desktop' which IP is 192.168.122.10 and add an account 'training_user' with password 's3cur3p4s5wd'. Install 'nmap'. Create another CentOS-based VM called 'server' and add an account 'admin' with password '4dm1np4s5wd'. Emulate a malware running as a process 'proc123' listening on port 9001. Please note that both VMs are KVM-based and use the configuration file 'home/matteolupinacci/Desktop/cyris/images/basevm.xml'.
    #Clone one instance of the CR containing one instance of each guest machine. The desktop VM is the entry point for the CR. Create a network segment called 'netseg1' and add all the VMs to it."""
    m2 = "Fix any errors and do the syntax verification step until the output is correct."
    #m3 = "Deploy the CR."

    create_new_scenario = "Hi, please write me a configuration file for CyRIS-based CR. You choose all the scenario characteristics."

    messages.append(create_new_scenario)
    #messages.append(m1)
    messages.append(m2)
    #messages.append(m3)

    for message in messages:
        if not correct_syntax:
            inputs = {"messages": [("user", message)]}
            print_stream(arcer.stream(inputs, stream_mode="values", config=config))
        else:
            break
            
            
def arcer():
    _set_env()
    
    # setup the splitter for splitting documents in chunks
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

    # setup the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # setup the vectore store based on the embeddings model
    cyris_vector_store = InMemoryVectorStore(embeddings)

    # load the documents (or directly the page_content (text))
    documents_page_list = [] #every Document has a 'page_content' and a 'metadata' property
    folder = "./cyris_docs"

    for filename in os.listdir(folder):
        if filename.endswith(".yml"):
            print(filename)
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                content = yaml.dump(data, default_flow_style=False, sort_keys=False)
                documents_page_list.append(Document(page_content=content, metadata={"source": file_path}))

        if filename.endswith(".pdf"):
            print(filename)
            file_path = os.path.join(folder, filename)
            loader = PyMuPDFLoader(file_path)
            documents_page_list.extend(loader.load())

    print(f"Number of loaded pdf pages:", len(documents_page_list))

    cyris_chunks = splitter.split_documents(documents_page_list)

    #store chunks in a vectorstore
    cyris_stored_chunks = cyris_vector_store.add_documents(cyris_chunks)

    vectore_store_dictionary = dict()
    vectore_store_dictionary["cyris"] = cyris_vector_store

    # setup list of external tools
    tools = []

    # boolean used during tool calling
    correct_syntax = False

    # create tools for leveraging calling-tools functionality during retrieval phase using langgraph
    @tool
    def retrieval(query):
        """Retrieve information related to a query from the vectorstore"""
        q = query.lower()
        vector_store = None
        for key, vs in vectore_store_dictionary.items():
            if key in q:
                vector_store = vs
                print(f"Using vectorstore: {key}")
                break

        # Create a retriever with MMR to reduce redundancy
        mmr_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,  # Total number of documents to retrieve
                "fetch_k": 20,  # Number of documents to fetch before filtering
                "lambda_mult": 0.5  # Balance between relevance and diversity
            }
        )
        retrieved_chunks = mmr_retriever.invoke(query)
        # Remove exact duplicates
        unique_chunks = []
        seen_contents = set()
        for chunk in retrieved_chunks:
            if chunk.page_content not in seen_contents:
                unique_chunks.append(chunk)
                seen_contents.add(chunk.page_content)

        ret = "\n\n".join((f"Content: {chunk.page_content}") for chunk in unique_chunks)
        return ret


    @tool
    def verify_cyris_description_file_syntax(content_file):
        """Test the syntax of the generated output"""
        api_url = api_base_url+"verify_cyris_description_file_syntax"
        #print(content_file)
        response = requests.post(api_url, json={"file": content_file})
        output = response.json()["output"]
        if "CORRECT FILE SYNTAX" in output:
            global correct_syntax
            correct_syntax = True
            return output


    @tool
    def deploy_cyber_range():
        """Deploy the cyber range on the remote host"""
        api_url = api_base_url+"deploy_cyber_range"
        response = requests.get(api_url)
        output = response.json()["output"]
        return output


    # add every tool to the list "tools"
    tools.append(retrieval)
    tools.append(verify_cyris_description_file_syntax)
    tools.append(deploy_cyber_range)

    # setup the url for the API
    api_base_url = "URL_OF_CYRIS"

    # setup the LLM model
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    # setup memory
    memory = MemorySaver()

    # setup the system promt
    system_prompt = """You are an assistant for the generation of cyber range (CR) configuration file and for the automatic deployment given a textual description of the scenario. Retrieve information about the CR framework and also use the provided example files.
    File format may change depending on the CR framewrok. You MUST USE ONLY THE USER REQUESTED FRAMEWORK and pay attention to the tag-name!"""
    
    # create the agent using langgraph
    arcer = create_react_agent(model, tools, prompt = system_prompt, checkpointer=memory)
    return arcer
    
        
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
            
if __name__ == "__main__":
    main()
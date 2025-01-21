""" 
Is create_conversational_retrieval_agent only supported by OpenAI?
"""

import argparse

import openai
from dotenv import load_dotenv
from kwwutils import (
    clock,
    execute,
    get_documents_by_path,
    get_embeddings,
    get_llm,
    printit,
)
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent,
    create_retriever_tool,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain_community.vectorstores import FAISS

_path = "../"

load_dotenv()


@clock
@execute
def main(options):
    chat_llm = get_llm(options)
    embeddings = get_embeddings(options)
    pathname = options["pathname"]
    documents = get_documents_by_path(pathname)
    vectordb = FAISS.from_documents(documents, embeddings)
    tools = [
        create_retriever_tool(
            retriever=vectordb.as_retriever(), 
            name="italy_travel",
            description="Searches and returns documents regarding Italy."
        )
    ]
    agent_executor = create_conversational_retrieval_agent(chat_llm, tools, memory_key='chat_history', verbose=True)
    agent_executor.invoke("what can I visit in India in 3 days?")
    """
    """ 

    

    port = options["port"]
    model = f"ollama/{options['model']}"
    question = options["question"]
    client = openai.OpenAI(
        api_key="anything", 
        base_url=f"http://0.0.0.0:{port}"
    )
    printit(model, client)
    printit("client type", type(client))
    printit("client dir", dir(client))
    response = client.chat.completions.create(
        model=model, 
        messages = [
        {
            "role": "user",
            "content": question,
        }
    ])
    
    """ 
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=client,
        tools=tools,
        prompt=prompt,
    )
    """


        

def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedmodel: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--pathname', type=str, help='pathname', default='italy_travel.pdf')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='chat')
    parser.add_argument('--port', type=int, help='repeatcnt', default=8000)
    parser.add_argument('--question', type=str, help='question', default='tell me a joke about science')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--vectordb_type', type=str, help='vectordb_type', default='memory')
    parser.add_argument('--vectorstore', type=str, help='vectorstore: Chroma, FAISS', default='Chroma')
    parser.add_argument('--model', type=str, help='model', default="openhermes")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument('--models', nargs='+', default=[
        "codellama:7b",        
        "codellama:7b-python",        
        "codellama:13b",        
        "codellama:13b-python",        
        "codellama:34b",        
        "codellama:34b-python",        
        "llama2:latest",           
        "llama2-uncensored:latest",           
        "medllama2:latest",        
        "medllama2:latest",        
        "mistral:instruct",        
        "mistrallite:latest",      
        "openchat:latest",         
        "orca-mini:latest",        
        "phi:latest",        
        "vicuna:latest",           

        "wizardcoder:latest",
        "wizardlm-uncensored:latest",        
        "yarn-llama2:latest",        
        "yarn-mistral:latest",
    ])
    return vars(parser.parse_args())


if __name__ == '__main__':
    options = Options()
    main(**options)

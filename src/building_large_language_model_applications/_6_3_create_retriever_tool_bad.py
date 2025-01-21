""" 
label: >>>{'input': 'what can I visit in Italy in 3 days?'}<<<:
values: >>>>>>{'input': 'what can I visit in Italy in 3 days?',
 'output': 'Agent stopped due to iteration limit or time limit.'}<<<<<<
"""

import argparse

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
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS

_path = "../"

questions = [
#   {"input": "Tell me something about Pantheon"},
#   {"input": "what can I visit in Italy in 3 days?"},
]
questions = [
    {"input": "Tell me something about Pantheon"},
    {"input": "Where can I visit in Italy in 3 days? Do not repeat more then 5 times to find an answer."},
]

@clock
@execute
def main(options):
    chat_llm = get_llm(options)
    embeddings = get_embeddings(options)
    filename = options["filename"]
    documents = get_documents_by_path(filename)
    vectordb = FAISS.from_documents(documents, embeddings)

    tool = create_retriever_tool(
        retriever=vectordb.as_retriever(),
        name="italy_travel",
        description="Searches and returns documents regarding Italy.",
    )
    tools = [tool]


    memory_key = "chat_history"
    memory = ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=True,
    )


    prompt = hub.pull("hwchase17/react")
    printit("+++++++++++++++++++++++++prompt", prompt)
    agent = create_react_agent(
        llm=chat_llm,
        tools=tools,
        prompt=prompt,
    )
    
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory_key=memory_key,
        handle_parsing_errors=True,
        verbose=True,
    )

    responses = []
    for question in questions:
        response = agent_executor.invoke(question)
        responses.append(response)
        printit(question, response)

    printit("prompt", prompt)
    printit("type(agent)", type(agent))
    printit("type(agent_executor)", type(agent_executor))
    return responses


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedmodel: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--filename', type=str, help='filename', default='italy_travel.pdf')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='chat')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="mistral")
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

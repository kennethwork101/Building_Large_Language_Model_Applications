import argparse

from kwwutils import (
    clock,
    execute,
    get_documents_by_path,
    get_embeddings,
    get_llm,
    printit,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

_path = "../"

custom_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. 
If you cannot find the answer in the document provided, ignore the document and answer anyway.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""


@clock
@execute
def main(options):
    chat_llm = get_llm(options)
    embeddings = get_embeddings(options)
    filename = options["filename"]
    documents = get_documents_by_path(filename)
    vectordb = FAISS.from_documents(documents, embeddings)

    memory_key = "chat_history"
    memory = ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=True,
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm, 
        retriever=vectordb.as_retriever(), 
        memory=memory, 
        verbose=True,
    )
    question = {'question': 'Give me some review about the Pantheon'}
    response = qa_chain.invoke(question)
    printit(question, response)

    
    # Custom prompt
    custom_prompt = PromptTemplate.from_template(template=custom_template)
    memory = ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=True,
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=vectordb.as_retriever(),
        condense_question_prompt=custom_prompt,
        memory=memory,
        verbose=True,
    )
    question = {'question':'What can I visit in India?'}
    response = qa_chain.invoke(question)
    printit(question, response)
    return response


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

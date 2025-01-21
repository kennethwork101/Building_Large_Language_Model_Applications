import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

_path = "../"

questions = [
    "Hi there!",
    "what is the most iconic place in southern Japan?",
    "What kind of other events?",
]

@clock
@execute
def main(options):
    chat_llm = get_llm(options)
    messages = [
        SystemMessage(content="You are a helpful assistant that help the user to plan an optimized itinerary."),
        HumanMessage(content="I'm going to Rome for 2 days, what can I visit?")
    ]
    response = chat_llm(messages)
    printit(messages, response)

    # Add memory to the conversation
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=chat_llm,
        memory=memory,
        verbose=True,
    )
    for question in questions:
        conversation.invoke(question)

    response = memory.load_memory_variables(inputs="")
    response = memory.load_memory_variables({})
    printit("1 memory", memory)
    printit("1 conversation", conversation)
    printit("1 inputs load_memory_variables", response)
    printit("1 {} load_memory_variables", response)

    
    # Use messges in a prompt in the conversation 
    template = "You are a helpful assistant that help the user to plan an optimized itinerary."
    memory_key = "chat_history"
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    messages = [
        SystemMessagePromptTemplate.from_template(template=template),
        MessagesPlaceholder(variable_name=memory_key),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    conversation = LLMChain(
        llm=chat_llm,
        prompt=prompt,
        verbose=False,
        memory=memory
    )
    for question in questions:
        response = conversation.invoke(question)
    mem_var = memory.load_memory_variables({})
    printit("2 memory", memory)
    printit("2 conversation", conversation)
    printit("2 inputs load_memory_variables", mem_var)
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedmodel: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='chat')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="mistral")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument('--models', nargs='+', default=[
#       "codellama:7b",        
#       "codellama:7b-python",        
#       "codellama:13b",        
#       "codellama:13b-python",        
#       "codellama:34b",        
#       "codellama:34b-python",        
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

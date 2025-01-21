import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate

_path = "../"

itinerary_template = """
You are a vacation itinerary assistant.
You help customers finding the best destinations and itinerary.
You help customer screating an optimized itinerary based on their preferences.
Here is a question:
{input}
"""


restaurant_template = """
You are a restaurant booking assitant.
You check with customers number of guests and food preferences.
You pay attention whether there are special conditions to take into account.
Here is a question:
{input}
"""


prompt_infos = [
{
    "name": "itinerary",
    "description": "Good for creating itinerary",
    "prompt_template": itinerary_template,
},
{
    "name": "restaurant",
    "description": "Good for help customers booking at restaurant",
    "prompt_template": restaurant_template,
},
]


questions = [
    "I'm planning a trip from Seattle to Colorado Springs by car. What can I visit in between?",
    "I want to book a table for tonight",
    "What was the first Disney movie?"
]


@clock
@execute
def main(options):
    chat_llm = get_llm(options)
    destination_chains = {}

    # prompt_infos map a name to the prompt
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = PromptTemplate.from_template(template=prompt_template)
        destination_chains[name] = LLMChain(llm=chat_llm, prompt=prompt)

    # default chain is a ConversationChain
    default_chain = ConversationChain(llm=chat_llm, output_key="text")

    # Destintions is the prompt_infos formated into a string to pass into router_template
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    printit("MULTI_PROMPT_ROUTER_TEMPLATE ", MULTI_PROMPT_ROUTER_TEMPLATE)

    """ 
# ### The following 2 entries are inserted into the router_template prompt under CANDIDATE PROMPTS
 '<< CANDIDATE PROMPTS >>\n'
 'itinerary: Good for creating itinerary\n'
 'restaurant: Good for help customers booking at restaurant\n'
    """
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    # Router chain has a list of destinations
    router_chain = LLMRouterChain.from_llm(chat_llm, router_prompt)

    # Multi chain
    multi_chain = MultiPromptChain(
        default_chain=default_chain,
        destination_chains=destination_chains,
        router_chain=router_chain,
        verbose=True,
    )

    printit("router_template", router_template)
    printit("router_prompt", router_prompt)
    printit("destinations", destinations)
    printit("destinations_str", destinations_str)

    printit("default_chain", default_chain)
    printit("destination_chains", destination_chains)
    printit("router_chain", router_chain)
    printit("multi_chain", multi_chain)

    responses = []
    for question in questions:
        response = multi_chain.invoke(question)
        responses.append(response)
        printit(question, response)
    return responses



def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedmodel: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='chat')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="llama2")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument('--models', nargs='+', default=[
        "codellama:7b",        
#       "codellama:7b-python",        
        "codellama:13b",        
        "codellama:13b-python",        
        "codellama:34b",        
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
#       "yarn-llama2:latest",        
#       "yarn-mistral:latest",
    ])
    return vars(parser.parse_args())


if __name__ == '__main__':
    options = Options()
    main(**options)

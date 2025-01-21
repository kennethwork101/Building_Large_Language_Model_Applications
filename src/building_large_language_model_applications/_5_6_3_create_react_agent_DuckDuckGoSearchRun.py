""" 
- This test is not reliable. 
  More than 50% will failed due to
  Observation: Invalid Format: Missing 'Action:' after 'Thought:
  Agent stopped due to iteration limit or time limit.
  
"""

import argparse

from dotenv import load_dotenv
from kwwutils import clock, execute, get_llm, printit
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

_path = "../"

load_dotenv()


@clock
@execute
def main(options):
    llm = get_llm(options)
    #search = SerpAPIWrapper()
    search = DuckDuckGoSearchRun()
    question = options["question"]


    # Set up the tools
    tools = [
        Tool.from_function(
            func=search.run,
            name="Search",
            description="Useful for when you need to answer questions about current events",
        ),
    ]

    # React agent
    prompt = hub.pull("hwchase17/react")
    printit("prompt", prompt)
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        handle_parsing_errors=True, 
        verbose=True
    )
    response = agent_executor.invoke({"input": question})
    printit(f"create_react_agent {question}", response)
    printit(f"agent ", agent)
    printit(f"agent type", type(agent))
    printit(f"agent_executor ", agent_executor)
    printit(f"agent_executor type", type(agent_executor))
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedmodel: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--filename', type=str, help='filename', default='Cats&Dogs.txt')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='llm')
    parser.add_argument('--question', type=str, help='question', default='When was Avatar 2 released?')
#   parser.add_argument('--question', type=str, help='question', default='What did they say about matlab?')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="mistral:instruct")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument('--models', nargs='+', default=[
    #   "codellama:7b",        
    #   "codellama:7b-python",        
    #   "codellama:13b",        
    #   "codellama:13b-python",        
    #   "codellama:34b",        
    #   "codellama:34b-python",        
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

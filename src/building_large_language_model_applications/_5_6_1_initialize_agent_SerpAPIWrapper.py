"""
- This test is not reliable.
  More than 50% will failed due to
  Observation: Invalid Format: Missing 'Action:' after 'Thought:
  Agent stopped due to iteration limit or time limit.

"""

import argparse

from dotenv import load_dotenv
from kwwutils import clock, execute, get_llm, printit
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper

_path = "../"

load_dotenv()


@clock
@execute
def main(options):
    llm = get_llm(options)
    search = SerpAPIWrapper()
    question = options["question"]

    # Set up the tools
    tools = [
        Tool.from_function(
            func=search.run,
            name="Search",
            description="Useful for when you need to answer questions about current events",
        ),
    ]

    agent_executor = initialize_agent(
        llm=llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )
    printit("agent_executor type ", type(agent_executor))
    response = agent_executor.invoke({"input": question})
    printit(f"agent_executor {question}", response)
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persist_directory",
        type=str,
        help="persist_directory",
        default=f"{_path}mydb/data_all/",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="embedding: chroma gpt4all huggingface",
        default="chroma",
    )
    parser.add_argument(
        "--embedmodel", type=str, help="embedmodel: ", default="all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="llm"
    )
    parser.add_argument(
        "--question", type=str, help="question", default="When was Avatar 2 released?"
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    parser.add_argument("--model", type=str, help="model", default="mistral:instruct")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
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
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)

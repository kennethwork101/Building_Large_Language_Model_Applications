import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

_path = "../"

template1 = "You are a comedian. Generate a joke on the following {topic} Joke:"
template2 = (
    "You are translator. Given a text input, translate it to {language} Translation:"
)


@clock
@execute
def main(options):
    llm = get_llm(options)
    question = options["question"]

    prompt1 = PromptTemplate.from_template(template=template1)
    prompt2 = PromptTemplate.from_template(template=template2)

    chain1 = LLMChain(prompt=prompt1, llm=llm)
    chain2 = LLMChain(prompt=prompt2, llm=llm)

    overall_chain = SimpleSequentialChain(chains=[chain1, chain2])

    # Simple sequential chain we can pass in question string directly
    response = overall_chain.invoke(question)
    printit(f"llmchain: invoke {question}", response)

    # Simple sequential chain we can pass in question using input key
    response = overall_chain.invoke({"input": question})

    printit(f"llmchain: invoke {question}", response)
    printit("prompt1", prompt1)
    printit("prompt2", prompt2)
    printit("chain1", chain1)
    printit("chain2", chain2)
    printit("overall_chain", overall_chain)
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
        "--question", type=str, help="question", default="Cats and Dogs"
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    parser.add_argument("--model", type=str, help="model", default="llama2")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
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
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)

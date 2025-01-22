"""
Defind a function rename_cat and call it via TransformChain
"""

import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.chains import LLMChain, SimpleSequentialChain, TransformChain
from langchain.prompts import PromptTemplate

_path = "../"

template = "Use the actual name and include it in the output to Summarize this text: {output_text}\n\nSummary:"


def rename_cat_fn(inputs: dict) -> dict:
    printit("inputs", inputs)
    text = inputs["input_text"]
    new_text = text.replace("cat", "Silvester the Cat")
    printit("new_text", new_text)
    return {"output_text": new_text}


@clock
@execute
def main(options):
    llm = get_llm(options)
    filename = options["filename"]

    with open(filename) as fp:
        cats_and_dogs = fp.read()

    # Transform chain is used in simple sequential chain
    transform_chain = TransformChain(
        input_variables=["input_text"],
        output_variables=["output_text"],
        transform=rename_cat_fn,
    )
    prompt = PromptTemplate.from_template(template=template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])
    response = sequential_chain.invoke(cats_and_dogs)
    printit(filename, response)
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
        "--filename", type=str, help="filename", default="Cats&Dogs.txt"
    )
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="llm"
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

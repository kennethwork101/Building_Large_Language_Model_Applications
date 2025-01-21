""" 
Defind a function rename_cat and call it via TransformChain
"""

import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.chains import TransformChain
from langchain.prompts import PromptTemplate

_path = "../"

template = "Use the actual name and include it in the output to Summarize this text: {output_text}\n\nSummary:"


def rename_cat_fn(inputs: dict) -> dict:
    text = inputs["input_text"]
    new_text = text.replace('cat', 'Silvester the Cat')
    output = {"output_text": new_text}
    return output


@clock
@execute
def main(options):
    llm = get_llm(options)

    filename = options["filename"]
    with open(filename) as fp:
        cats_and_dogs = fp.read()

    transform_chain = TransformChain(
        input_variables=["input_text"], 
        output_variables=["output_text"],
        transform=rename_cat_fn,
    )

    prompt = PromptTemplate.from_template(template=template)
    chain = {"output_text": transform_chain} | prompt | llm

    response = chain.invoke(cats_and_dogs)
    printit(222, response)
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--persist_directory', type=str, help='persist_directory', default=f'{_path}mydb/data_all/')
    parser.add_argument('--embedding', type=str, help='embedding: chroma gpt4all huggingface', default='chroma')
    parser.add_argument('--embedmodel', type=str, help='embedmodel: ', default='all-MiniLM-L6-v2')
    parser.add_argument('--filename', type=str, help='filename', default='Cats&Dogs.txt')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='llm')
    parser.add_argument('--repeatcnt', type=int, help='repeatcnt', default=1)
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    parser.add_argument('--model', type=str, help='model', default="openchat")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument('--models', nargs='+', default=[
        "codebooga:latest",
        "codellama:13b",
        "codellama:13b-python",
        "codellama:34b",
        "codellama:34b-python",
        "codellama:7b",
        "codellama:7b-python",
        "codeup:latest",
        "deepseek-coder:latest",
        "dolphin-mistral:latest",
        "dolphin-mixtral:latest",
##      "falcon:latest",
        "llama-pro:latest",
        "llama2-uncensored:latest",
        "llama2:latest",
        "magicoder:latest",
##      "meditron:latest",
        "medllama2:latest",
        "mistral-openorca:latest",
        "mistral:instruct",
        "mistral:latest",
        "mistrallite:latest",
        "mixtral:latest",
        "openchat:latest",
        "orca-mini:latest",
        "orca2:latest",
        "phi:latest",
        "phind-codellama:latest",
        "sqlcoder:latest",
##      "stable-code:latest",
##      "starcoder:latest",
        "starling-lm:latest",
        "tinyllama:latest",
        "vicuna:latest",
        "wizardcoder:latest",
        "wizardlm-uncensored:latest",
##      "yarn-llama2:latest",
        "yarn-mistral:latest",
    ])
    return vars(parser.parse_args())


if __name__ == '__main__':
    options = Options()
    main(**options)

import argparse

from dotenv import load_dotenv
from kwwutils import clock, execute, printit
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

_path = "../"

load_dotenv()

template = "Question: {question}\n\nAnswer: give a direct answer"


@clock
@execute
def main(options):
    question = options["question"]
    repo_id = "tiiuae/falcon-7b-instruct"
    hfh_llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_lenghth": 1000}
    )

    # Use hfh_llm to ask questions
    response = hfh_llm.invoke(question)
    printit(question, response)

    # We can also use hfh_llm to set up a chain in case we have more complicated input prompt
    prompt = PromptTemplate.from_template(template=template)
    chain = prompt | hfh_llm
    response = chain.invoke({"question": question})
    printit(question, response)
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
    parser.add_argument(
        "--question",
        type=str,
        help="question",
        default="What was the first and second Disney movie?",
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    parser.add_argument("--model", type=str, help="model", default="mistral")
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

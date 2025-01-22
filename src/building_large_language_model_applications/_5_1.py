import argparse
import csv
import os

from kwwutils import clock, execute, get_embeddings, get_llm, printit
from langchain.chains import LLMChain, RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.vectorstores import FAISS

_path = "../"

data1 = [
    ["Name", "Age", "City"],
    ["John", 25, "New York"],
    ["Emily", 28, "Los Angeles"],
    ["Michael", 22, "Chicago"],
]


content = """
Amidst the serene landscape, towering mountains stand as majestic guardians of nature's beauty.
The crisp mountain air carries whispers of tranquility, while the rustling leaves compose a symphony of wilderness.
Nature's palette paints the mountains with hues of green and brown, creating an awe-inspiring sight to behold.
As the sun rises, it casts a golden glow on the mountain peaks, illuminating a world untouched and wild.
"""


dialogue_lines = [
    "Good morning!",
    "Oh, hello!",
    "I want to report an accident",
    "Sorry to hear that. May I ask your name?",
    "Sure, Mario Rossi.",
]

template = "Sentence: {sentence} Translation in {language}:"


@clock
@execute
def main(options):
    llm = get_llm(options)
    embeddings = get_embeddings(options)

    prompt = PromptTemplate.from_template(template=template)
    prompt_str = prompt.format(sentence="the cat is on the table", language="spanish")

    response = llm.invoke(prompt_str)
    printit(prompt_str, response)

    response = llm.generate([prompt_str])
    printit(prompt_str, response)

    filename = options["filename"]
    dirpath = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirpath, filename)
    with open(filename, "w", newline="") as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerows(data1)
    loader = CSVLoader(file_path=options["filename"])
    data = loader.load()
    printit(options["filename"], data)

    # Write content
    file_name = "mountain.txt"
    with open(file_name, "w") as fp:
        fp.write(content)

    # Read content
    with open("mountain.txt") as fp:
        mountain = fp.read()

    # Retrieve texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=20, length_function=len
    )
    texts = text_splitter.create_documents([mountain])
    printit("6 texts type", type(response))
    printit(file_name, texts)

    embedding_data = embeddings.embed_documents(dialogue_lines)
    print(
        f"Number of vector: {len(embedding_data)}; Dimension of each data vector: {len(embedding_data[-1])}"
    )
    embedded_query = embeddings.embed_query(
        "What was the name mentioned in the conversation?"
    )
    print(f"Dimension of the query vector: {len(embedded_query)}")
    print(f"Sample of the first 4 elements of the vector: {embedded_query[:5]}")

    file_name = "dialogue.txt"
    # with open(file_name, 'w') as fp:
    with open(file_name, "w") as fp:
        for line in dialogue_lines:
            fp.write(line + "\n")
    print(f'Dialogue text file "{file_name}" generated and saved.')

    text_splitter = CharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separator="\n",
    )
    raw_documents = TextLoader("dialogue.txt").load()
    documents = text_splitter.split_documents(raw_documents)
    vectordb = FAISS.from_documents(documents, embeddings)
    retriever = vectordb.as_retriever()
    printit("vectordb", vectordb)
    printit("retriever", retriever)

    question = "What is the reason for calling?"
    docs = vectordb.similarity_search(question)
    printit(question, docs[0].page_content)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = qa.invoke(question)
    printit(f"{question} qa.invoke", response)
    printit("type response qa.invoke", type(response))
    printit(f"{question} qa", response)
    printit("type response qa", type(response))

    memory = ConversationSummaryMemory(llm=llm)
    memory.save_context(
        {"input": "hi, I'm looking for some ideas to write an essay in AI"},
        {"output": "hello, what about writing on LLMs?"},
    )
    printit("memory", memory.load_memory_variables({}))

    inputs = {"sentence": "the cat is on the table", "language": "spanish"}
    prompt = PromptTemplate.from_template(template=template)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.invoke(
        {"sentence": "the cat is on the table", "language": "spanish"}
    )
    printit("llm chain response", response)

    chain2 = prompt | llm
    response = chain2.invoke(inputs)
    printit("lcel chain response", response)
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
    parser.add_argument("--filename", type=str, help="filename", default="sample.csv")
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
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)

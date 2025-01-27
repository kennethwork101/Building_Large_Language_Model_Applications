import argparse
import ast

import lancedb
import pandas as pd

_path = "../"


from kwwutils import clock, execute, printit
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chains import (
    RetrievalQA,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.vectorstores import LanceDB


def calculate_weighted_rate(vote_average, vote_count, min_vote_count=10):
    """
    Calculate weighted rate (IMDb formula)
    """
    return (vote_count / (vote_count + min_vote_count)) * vote_average + (
        min_vote_count / (vote_count + min_vote_count)
    ) * 5.0


@clock
@execute
def main(options):
    md = pd.read_csv("movies_metadata.csv")
    print(md.head())
    # Convert string representation of dictionaries to actual dictionaries
    md["genres"] = md["genres"].apply(ast.literal_eval)
    # Transforming the 'genres' column
    md["genres"] = md["genres"].apply(lambda x: [genre["name"] for genre in x])
    # Minimum vote count to prevent skewed results
    vote_counts = md[md["vote_count"].notnull()]["vote_count"].astype("int")
    min_vote_count = vote_counts.quantile(0.95)
    # Create a new column 'weighted_rate'
    md["weighted_rate"] = md.apply(
        lambda row: calculate_weighted_rate(
            row["vote_average"], row["vote_count"], min_vote_count
        ),
        axis=1,
    )
    md = md.dropna()
    md_final = md[["genres", "title", "overview", "weighted_rate"]].reset_index(
        drop=True
    )
    printit("md_final", md_final)

    # Create a new column by combining 'title', 'overview', and 'genre'
    md_final["combined_info"] = md_final.apply(
        lambda row: f"Title: {row['title']}. Overview: {row['overview']} Genres: {', '.join(row['genres'])}. Rating: {row['weighted_rate']}",
        axis=1,
    )
    print(md_final["combined_info"][9])

    # embedding model parameters
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
    encoding = tiktoken.get_encoding(embedding_encoding)
    # omit reviews that are too long to embed
    md_final["n_tokens"] = md_final.combined_info.apply(
        lambda x: len(encoding.encode(x))
    )
    md_final = md_final[md_final.n_tokens <= max_tokens]
    len(md_final)
    md_final.head()
    md_final["embedding"] = md_final.overview.apply(
        lambda x: get_embedding(x, engine=embedding_model)
    )
    md_final.head()
    md_final.rename(columns={"embedding": "vector"}, inplace=True)
    md_final.rename(columns={"combined_info": "text"}, inplace=True)
    md_final.to_pickle("movies.pkl")
    print(md.head(2))

    uri = "data/sample-lancedb"
    db = lancedb.connect(uri)
    table = db.create_table("movies", md)
    embeddings = OpenAIEmbeddings()
    docsearch = LanceDB(connection=table, embedding=embeddings)
    query = "I'm looking for an animated action movie. What could you suggest to me?"
    docs = docsearch.similarity_search(query)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    query = "I'm looking for an animated action movie. What could you suggest to me?"
    result = qa({"query": query})
    df_filtered = md[md["genres"].apply(lambda x: "Comedy" in x)]
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"data": df_filtered}),
        return_source_documents=True,
    )
    query = "I'm looking for a movie with animals and an adventurous plot."
    result = qa({"query": query})
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"filter": {"adult": "False"}}),
        return_source_documents=True,
    )
    query = "I'm looking for a movie with animals and an adventurous plot."
    result = qa({"query": query})

    system_message = SystemMessage(
        content=""" 
                Do your best to answer the questions.
                if there are more than one argument for the single-input tool, 
                reason step by step and treat them as single input.
                relevant information, only if neccessary
            """
    )
    # This is needed for both the memory and the prompt
    memory_key = "history"
    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)
    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    )
    agent_executor = create_conversational_retrieval_agent(
        llm=llm, tools=tools, prompt=prompt, verbose=True
    )
    result = agent_executor(
        {
            "input": "I liked a lot kung fu panda 1 and 2. Could you suggest me some similar movies?"
        }
    )

    template = """
    You are a movie recommender system that help users to find movies that match their preferences. 
    Use the following pieces of context to answer the question at the end. 
    For each question, suggest three movies, with a short description of the plot and the reason why the user migth like it.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Your response:
    """
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    query = "I'm looking for a funny action movie, any suggestion?"
    result = qa({"query": query})
    print(result["result"])

    template_prefix = """
    You are a movie recommender system that help users to find movies that match their preferences. 
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    """
    user_info = """
    This is what we know about the user, and you can use this information to better tune your research:
    Age: {age}
    Gender: {gender}
    """
    template_suffix = "Question: {question} Your response:"
    user_info = user_info.format(age=18, gender="female")
    COMBINED_PROMPT = template_prefix + "\n" + user_info + "\n" + template_suffix
    print(COMBINED_PROMPT)

    PROMPT = PromptTemplate(
        template=COMBINED_PROMPT, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    query = "Can you suggest me some action movie?"
    result = qa({"query": query})
    print(result["result"])
    print(result["source_documents"])

    data = {
        "username": ["Alice", "Bob"],
        "age": [25, 32],
        "gender": ["F", "M"],
        "movies": [
            [
                ("Transformers: The Last Knight", 7),
                ("Pokémon: Spell of the Unknown", 5),
            ],
            [("Bon Cop Bad Cop 2", 8), ("Goon: Last of the Enforcers", 9)],
        ],
    }
    # Convert the "movies" column into dictionaries
    for i, row_movies in enumerate(data["movies"]):
        movie_dict = {}
        for movie, rating in row_movies:
            movie_dict[movie] = rating
        data["movies"][i] = movie_dict
    # Create a pandas DataFrame
    df = pd.DataFrame(data)
    print(df.head())

    template_prefix = """
    You are a movie recommender system that help users to find movies that match their preferences. 
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}
    """
    user_info = """
    This is what we know about the user, and you can use this information to better tune your research:
    Age: {age}
    Gender: {gender}
    Movies already seen alongside with rating: {movies}
    """
    template_suffix = "Question: {question} Your response:"
    age = df.loc[df["username"] == "Alice"]["age"][0]
    gender = df.loc[df["username"] == "Alice"]["gender"][0]
    movies = ""
    # Iterate over the dictionary and output movie name and rating
    for movie, rating in df["movies"][0].items():
        output_string = f"Movie: {movie}, Rating: {rating}" + "\n"
        movies += output_string
        # print(output_string)
    user_info = user_info.format(age=age, gender=gender, movies=movies)
    COMBINED_PROMPT = template_prefix + "\n" + user_info + "\n" + template_suffix
    print(COMBINED_PROMPT)

    PROMPT = PromptTemplate(
        template=COMBINED_PROMPT, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    query = "Can you suggest me some action movie based on my background?"
    response = qa({"query": query})
    print(response["result"])
    print(response["source_documents"])
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
        "--llm_type", type=str, help="llm_type: chat or llm", default="chat"
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

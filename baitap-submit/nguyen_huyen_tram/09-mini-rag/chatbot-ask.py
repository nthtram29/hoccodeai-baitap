import requests
import json
import os
from openai import OpenAI
from pydantic import TypeAdapter
import inspect
from dotenv import load_dotenv
from pprint import pprint
import chromadb
from chromadb.utils import embedding_functions
from wikipediaapi import Wikipedia

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
COLLECTION_NAME = "celebrity"
client = chromadb.PersistentClient(path="./data-new")
# client.heartbeat()

embedding_function = embedding_functions.DefaultEmbeddingFunction()

collections = client.list_collections()
if COLLECTION_NAME not in collections:
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
else:
    collection = client.get_collection(COLLECTION_NAME)

wiki = Wikipedia('TramWikipedia/0.0 (http://github.com/nthtram29)', 'en')

def search_wikipedia(query: str) -> str:
    """Find the Wikipedia article for a given person's name or topic and return the content of the most relevant article."""
   
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1
    }

    res = requests.get(url, params=params)
    data = res.json()

    if "query" not in data or not data["query"].get("search"):
        return "Không tìm thấy kết quả phù hợp trên Wikipedia."

    # Lấy kết quả tìm kiếm đầu tiên
    title = data["query"]["search"][0]["title"]
    doc = wiki.page(title).text
    
    paragraphs = doc.split('\n\n')

    for index, paragraph in enumerate(paragraphs):
        collection.add(documents=[paragraph], ids=[str(index)])
    # print(f"doc: {doc}")
    return doc

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": inspect.getdoc(search_wikipedia),
            "parameters": TypeAdapter(search_wikipedia).json_schema(),
        },
    }
]

client_openai = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
)

def get_completion(messages):
    response = client_openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=messages,
        tools=tools,
        tool_choice="required",
        temperature=0
    )
    return response

FUNCTION_MAP = {
    "search_wikipedia": search_wikipedia
}

messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant. Your job is to provide accurate information to users. When a user mentions someone or a specific topic, you must always use the search_wikipedia tool to find the most relevant article on Wikipedia and provide the link. Don't respond with generic information unless you retrieve it through the tool."
    }
]

while True:
    question = input("\nNhập câu hỏi của bạn (hoặc nhập 'exit' để thoát): ").strip()
    
    if question.lower() == "exit":
        print("Thoát chương trình.")
        break
    
 
    messages.append({"role": "user", "content": question})

    response = get_completion(messages)
    first_choice = response.choices[0]
    finish_reason = first_choice.finish_reason

    if finish_reason != "stop":
        tool_call = first_choice.message.tool_calls[0]

        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)
      
        messages.append(first_choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call_function.name,
            "content": json.dumps(result)
        })

        
        q = collection.query(query_texts=[question], n_results=3)
        CONTEXT = q["documents"][0]
       
        prompt = f"""
        Use the following CONTEXT to answer the QUESTION at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use an unbiased and journalistic tone.
        Please answer in **Vietnamese**.

        CONTEXT: {CONTEXT}

        QUESTION: {question}
        """
        
        # Gửi prompt cho OpenAI để lấy câu trả lời cuối cùng
        response = client_openai.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    # In kết quả trả lời cuối cùng từ OpenAI
    print(f"BOT: {first_choice.message.content}")
    messages.append(
        {"role": "assistant", "content": first_choice.message.content}
    )

import chromadb
from chromadb.utils import embedding_functions
from wikipediaapi import Wikipedia
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

COLLECTION_NAME = "celebrity"
client = chromadb.PersistentClient(path="./data")

embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Kiểm tra và tạo collection nếu chưa có
collections = client.list_collections()
if COLLECTION_NAME not in collections:
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
else:
    collection = client.get_collection(COLLECTION_NAME)

wiki = Wikipedia('TramWikipedia/0.0 (http://github.com/nthtram29)', 'en')

wiki_links = [
    "Sơn_Tùng_M-TP",
    "Jujutsu_Kaisen"
]

# Lấy và thêm tiểu sử vào cơ sở dữ liệu ChromaDB
for page_name in wiki_links:
    doc = wiki.page(page_name).text
    paragraphs = doc.split('\n\n')
    
    for index, paragraph in enumerate(paragraphs):
        collection.add(documents=[paragraph], ids=[f"{page_name}_{index}"])

# Chức năng để thực hiện truy vấn và lấy thông tin từ ChromaDB
def ask_question(query: str):
    q = collection.query(query_texts=[query], n_results=3)
    CONTEXT = q["documents"][0]

    prompt = f"""
    Use the following CONTEXT to answer the QUESTION at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use an unbiased and journalistic tone.
    Please answer in **Vietnamese**.

    CONTEXT: {CONTEXT}

    QUESTION: {query}
    """
    
    client_openai = OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=TOGETHER_API_KEY,
    )

    response = client_openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Nhập câu hỏi từ người dùng
while True:
    question = input("\nNhập câu hỏi tiểu sử (hoặc nhập 'exit' để thoát): ").strip()

    if question.lower() == "exit":
        print("Thoát chương trình.")
        break

    # Gọi hàm hỏi câu hỏi và in ra kết quả
    answer = ask_question(question)
    print(f"BOT: {answer}")

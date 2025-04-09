# Áp dụng Weaviate để tạo ra một RAG flow đơn giản, gợi ý sách hay, dựa theo query của người dùng.
# Thay vì hard code query, hãy lấy query từ người dùng input ở console, hoặc tạo app bằng Gradio.
import weaviate
import requests
import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def generate_with_gemini(prompt):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
        }
    }
    response = requests.post(GEMINI_API_URL, json=payload)
    if response.ok:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    print("Gemini API Error:", response.text)
    return "Error from Gemini API"

single_prompt = "Viết một đoạn giới thiệu ngắn gọn tiếng Việt về cuốn sách: {title}. Nội dung chính: {description}. Trích đoạn nhỏ: {intro}"

translate_prompt = """
Hãy dịch câu sau từ tiếng Việt sang tiếng Anh để sử dụng làm từ khoá tìm kiếm sách. 
Chỉ trả về một câu ngắn gọn, súc tích, giữ được nội dung của câu gốc và không giải thích thêm.

Câu cần dịch: "{query}"
"""

vector_db_client = weaviate.connect_to_local(
    host="localhost",
    port=8080
)

COLLECTION_NAME = "BookCollectionRAG"
books = vector_db_client.collections.get(COLLECTION_NAME)

def recommend_books(user_query):
    prompt_trans = translate_prompt.format(query=user_query)
    query_in_english = generate_with_gemini(prompt_trans)

    print("Query gốc:", user_query)
    print("Query dịch sang English:", query_in_english)

    response = books.query.near_text(
        query=query_in_english,
        limit=10,
    )

    if not response.objects:
        return "Không tìm thấy sách phù hợp."

    result_md = "## Cậu tham khảo thử những cuốn sách dưới đây nhé!\n\n"

    for idx, book in enumerate(response.objects, start=1):
        title = book.properties['title']
        description = book.properties.get('description', '')
        intro = book.properties.get('intro', '')
        author = book.properties.get('author', 'Không rõ')
        path = book.properties.get('path', '')

        url = f"https://www.commonlit.org{path}" if path else "#"

        prompt = single_prompt.format(title=title, description=description, intro=intro)
        generated_intro = generate_with_gemini(prompt)

        content = f"""
#### {idx}. [{title}]({url})

*Tác giả: {author}*

> {generated_intro}
#### [🔗 Xem chi tiết]({url})
---
----

"""
        result_md += content + "\n\n"
    print('Đã tìm xong!')
    return result_md

with gr.Blocks(theme="soft") as iface:
    gr.Markdown("## Book Recommender RAG App")
    gr.Markdown("Ứng dụng gợi ý sách dựa trên Weaviate + Gemini. Nhập chủ đề bạn muốn tìm sách và xem kết quả gợi ý.")

    with gr.Row():
        query = gr.Textbox(label="Nhập chủ đề bạn quan tâm (tiếng Anh hoặc Việt đều được)", placeholder="Ví dụ: truyện ngắn cảm động, self-help, kinh dị...")
   
    search_button = gr.Button("Tìm sách", variant="primary")
    
    status = gr.Markdown()
    result = gr.Markdown()

    def search_with_status(user_query):
        status_text = "___\n\n🔍 Tớ đang tìm sách, chờ tớ chút nhé..."
        yield status_text, ""

        result_md = recommend_books(user_query)

        status_text = ""
        yield status_text, result_md
    

    search_button.click(
        search_with_status,
        inputs=query,
        outputs=[status, result]
    )

iface.launch()

vector_db_client.close()

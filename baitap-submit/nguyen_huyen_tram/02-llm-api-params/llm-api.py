import requests
from bs4 import BeautifulSoup
from openai import OpenAI

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key="..."
)
messages = []

# Câu 1, 2, 3:

def extract_text_from_website(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Tuỳ chỉnh lấy nội dung theo từng báo cụ thể (ví dụ Tuổi Trẻ hoặc VNExpress)
        main_content = soup.find("div", id="main-detail")
        if main_content:
            paragraphs = main_content.find_all("p")
            text = "\n".join([p.get_text(strip=True) for p in paragraphs])
            return text.strip()
        else:
            return "⚠️ Không tìm thấy nội dung chính trong trang web."
    except Exception as e:
        return f"❌ Lỗi khi lấy nội dung trang web: {e}"

print("💬 Chat với bot (gõ 'exit' để thoát)")
print("🌐 Dán link báo vào để tóm tắt nội dung.")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Tạm biệt!")
            break

        if user_input.startswith("http://") or user_input.startswith("https://"):
            print("🔍 Đang lấy nội dung từ trang web...")
            article_text = extract_text_from_website(user_input)

            if article_text.startswith("⚠️") or article_text.startswith("❌"):
                print(article_text)
                continue

            prompt = f"Summarize the article below in Vietnamese. Make it short, clear, and easy to understand for a general audience:\n\n{article_text}"


            print("🤖 Bot (Tóm tắt): ", end="", flush=True)
            stream = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="meta-llama/Llama-3-70b-chat-hf",
                temperature=0.7,
                stream=True
            )

            full_reply = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_reply += content
            print()
        else:
            # Chat bình thường
            messages.append({"role": "user", "content": user_input})

            print("🤖 Bot: ", end="", flush=True)
            stream = client.chat.completions.create(
                messages=messages,
                model="meta-llama/Llama-3-70b-chat-hf",
                temperature=0.7,
                stream=True
            )

            full_reply = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    full_reply += content
            print()

            messages.append({"role": "assistant", "content": full_reply})

    except Exception as e:
        print(f"\n❌ Lỗi: {e}")

####

# Câu 4:
def split_text(text, max_chunk_size=2000):
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
        else:
            current_chunk += line + '\n'
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

input_path = input("📄 Nhập đường dẫn file cần dịch: ").strip()
output_path = input("💾 Nhập đường dẫn file xuất kết quả (ví dụ: translated_output.txt): ").strip()

if not os.path.exists(input_path):
    print("❌ File không tồn tại.")
    exit()

with open(input_path, 'r', encoding='utf-8') as f:
    original_text = f.read()

chunks = split_text(original_text)
translated_chunks = []

for idx, chunk in enumerate(chunks):
    print(f"🔄 Đang dịch đoạn {idx + 1}/{len(chunks)}...")

    prompt = f"Translate the following Vietnamese text into English with a formal and professional tone:\n\n{chunk}"

    try:
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0.7,
            stream=True
        )

        translated_text = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                translated_text += content
        print()
        translated_chunks.append(translated_text.strip())

    except Exception as e:
        print(f"❌ Lỗi khi dịch đoạn {idx + 1}: {e}")
        translated_chunks.append(f"[Lỗi dịch đoạn {idx + 1}]")

# Ghi kết quả vào file mới
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(translated_chunks))

print(f"\n✅ Đã hoàn tất! Kết quả được lưu vào: {output_path}")

####

# Câu 5:
print("💻 AI Code Assistant – Hỏi bài tập lập trình (Python/JavaScript)")
print("💬 Gõ câu hỏi (hoặc 'exit' để thoát):")

while True:
    try:
        user_input = input("📝 Câu hỏi: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Tạm biệt!")
            break

        prompt = f"""You are a helpful programming assistant.
Please write clean, well-documented code in either Python or JavaScript to solve the following problem:
{user_input}
Also include comments to explain the logic."""

        print("🤖 Bot đang viết code...\n")
        stream = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True
        )

        full_code = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                full_code += content
        print()

        save = input("💾 Bạn có muốn lưu code vào file không? (y/n): ").strip().lower()
        if save == "y":
            file_name = input("📂 Nhập tên file (vd: solution.py hoặc solution.js): ").strip()
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(full_code)
            print(f"✅ Đã lưu vào file: {file_name}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

###
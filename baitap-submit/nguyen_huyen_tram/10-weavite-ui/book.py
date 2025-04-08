# Viết code để tìm kiếm sách/query từ Weavite
import weaviate
import gradio as gr
import webbrowser
from weaviate.classes.query import Filter

# Connect to Weaviate local instance
client = weaviate.connect_to_local(host="localhost", port=8080)

print("DB is ready:", client.is_ready())

COLLECTION_NAME = "BookCollection"

book_url_map = {}

def search_books(title_intro_query, author_query, genre_query):
    book_collection = client.collections.get(COLLECTION_NAME)

    filters = []

    if title_intro_query:
        filters.append(
            Filter.any_of([
                Filter.by_property("title").like(f"*{title_intro_query}*"),
                Filter.by_property("intro").like(f"*{title_intro_query}*")
            ])
        )

    if author_query:
        filters.append(Filter.by_property("author").like(f"*{author_query}*"))

    if genre_query:
        filters.append(Filter.by_property("genre").like(f"*{genre_query}*"))

    combined_filter = Filter.all_of(filters) if filters else None

    response = book_collection.query.fetch_objects(
        filters=combined_filter,
        limit=10
    )

    results = []
    for book in response.objects:
        title = book.properties.get('title', 'Unknown Title')
        author = book.properties.get('author', 'Unknown Author')
        genre = book.properties.get('genre', 'Unknown Genre')
        path = book.properties.get('path', '')

        url = f"https://www.commonlit.org{path}"

        book_info = (
            f"📖 Title: {title}\n"
            f"✍️ Author: {author}\n"
            f"🏷️ Genre: {genre}\n\n"
            f"🔗 Click here to read!!"
        )

        results.append([book_info])
        book_url_map[book_info] = url

    return results

with gr.Blocks(title="Book Search App") as interface:
    gr.Markdown("## 🔍 Search for Books in the Vector Database")

    with gr.Row():
        title_intro_query = gr.Textbox(label="Search by Title or Introduction")
        author_query = gr.Textbox(label="Search by Author")
        genre_query = gr.Textbox(label="Search by Genre")

    search_btn = gr.Button("Search")

    book_list = gr.Dataframe(
        headers=["Book Information"],
        visible=False,
        interactive=False
    )

    def do_search(q1, q2, q3):
        results = search_books(q1, q2, q3)
        return gr.update(visible=True, value=results, interactive=False)

    search_btn.click(fn=do_search, inputs=[title_intro_query, author_query, genre_query], outputs=book_list)

    def open_url(evt: gr.SelectData):
        book_info = evt.value
        url = book_url_map.get(book_info)
        if url:
            webbrowser.open_new_tab(url)

    book_list.select(open_url)

interface.queue().launch()

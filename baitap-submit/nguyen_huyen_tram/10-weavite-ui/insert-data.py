# Viết code để insert dữ liệu vào Weavite
import weaviate
import gradio as gr
import pandas as pd
from weaviate.classes.config import Configure, Property, DataType, Tokenization

vector_db_client = weaviate.connect_to_local(
    host="localhost",
    port=8080
)

print("DB is ready: {}".format(vector_db_client.is_ready()))

# Cấu hình tên collection
COLLECTION_NAME = "BookCollection"

# Lệnh xóa collection cũ
def delete_collection():
    if vector_db_client.collections.exists(COLLECTION_NAME):
        vector_db_client.collections.delete(COLLECTION_NAME)
        print(f"Collection {COLLECTION_NAME} has been deleted.")


# Hàm tạo collection mới
def create_collection():
    # Tạo schema cho collection
    movie_collection = vector_db_client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
        properties = [
            Property(name="title", data_type=DataType.TEXT, vectorize_property_name=True, tokenization=Tokenization.LOWERCASE),
            Property(name="intro", data_type=DataType.TEXT, tokenization=Tokenization.WHITESPACE),
            Property(name="excerpt", data_type=DataType.TEXT, tokenization=Tokenization.WHITESPACE),
            Property(name="author", data_type=DataType.TEXT, tokenization=Tokenization.WHITESPACE),
            Property(name="genre", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.WHITESPACE),
            Property(name="description", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="grade", data_type=DataType.NUMBER, skip_vectorization=True),  # Changed to NUMBER
            Property(name="lexile", data_type=DataType.NUMBER, skip_vectorization=True),
            Property(name="path", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="is_prose", data_type=DataType.NUMBER, skip_vectorization=True),  # Changed to BOOLEAN
            Property(name="date", data_type=DataType.TEXT, skip_vectorization=True),
        ]

    )

    # Đọc dữ liệu từ file CSV
    data = pd.read_json('book.json')

    # Chuyển đổi dữ liệu để import và xử lý kiểu dữ liệu sai
    sent_to_vector_db = []
    for record in data.to_dict(orient='records'):
        # Sửa kiểu dữ liệu cho các trường
        if isinstance(record['is_prose'], float) and record['is_prose'] not in [0, 1]:
            record['is_prose'] = False  # Hoặc True tùy thuộc vào dữ liệu
        if isinstance(record['lexile'], float) and pd.isna(record['lexile']):
            record['lexile'] = 0.0  # Hoặc giá trị phù hợp với lexile
        if isinstance(record['lexile'], str) and record['lexile'] == '':
            record['lexile'] = 0.0  # Nếu là chuỗi rỗng, thay bằng số 0.0
        if isinstance(record['genre'], str):
            record['genre'] = [record['genre']]  # Đảm bảo genre là mảng
        if isinstance(record['date'], int):
            record['date'] = str(record['date'])  # Chuyển số thành chuỗi
        if record['date'] == '':
            record['date'] = 'unknown'  # Nếu date là chuỗi rỗng, thay bằng giá trị mặc định

        sent_to_vector_db.append(record)

    total_records = len(sent_to_vector_db)
    print(f"Inserting data to Vector DB. Total records: {total_records}")

    # Import dữ liệu vào DB theo batch
    with movie_collection.batch.dynamic() as batch:
        for data_row in sent_to_vector_db:
            print(f"Inserting: {data_row['title']}")
            batch.add_object(properties=data_row)

    print("Data saved to Vector DB")


# Kiểm tra và xóa collection cũ nếu có
delete_collection()

# Tạo collection mới
create_collection()

# Lấy collection theo tên
movie_collection = vector_db_client.collections.get(COLLECTION_NAME)

vector_db_client.close()

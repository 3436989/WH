import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# 加载CSV数据
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'campus_data.csv')
df = pd.read_csv(csv_path)

# 使用免费嵌入模型（本地运行）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"trust_remote_code": True}
)

# 准备文本和元数据（以回答内容为文本，元数据包含id, category, question, source）
texts = df['answer'].tolist()
metadatas = df[['id', 'category', 'question', 'source']].to_dict('records')

# 创建向量库并持久化
persist_dir = os.path.join(os.path.dirname(__file__), '..', 'vector_db')
vector_db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,          # 改为 metadatas
    persist_directory=persist_dir
)
# 新版本 Chroma 会自动持久化，无需显式调用 .persist()
# 但为了兼容，可以调用（如果存在）
try:
    vector_db.persist()
except AttributeError:
    pass

print(f"✅ 向量库构建完成，共存入 {len(texts)} 条记录")
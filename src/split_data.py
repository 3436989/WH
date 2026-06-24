from langchain_text_splitters import RecursiveCharacterTextSplitter

# 长文本切割工具函数
def split_long_text(long_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    chunks = text_splitter.split_text(long_text)
    return chunks

# 测试代码
if __name__ == "__main__":
    test_text = """校园奖学金评定要求：学年绩点3.0以上，无挂科，体测达标，无处分。一等奖3.8绩点，二等奖3.5绩点。"""
    result = split_long_text(test_text)
    print("文本切分结果：")
    print(result)
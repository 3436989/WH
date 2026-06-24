import streamlit as st
import os
import re
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa

load_dotenv()

# ------------------- 缓存资源 -------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True}
    )

@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()
    return Chroma(persist_directory="./vector_db", embedding_function=embeddings)

embeddings = load_embeddings()
vector_db = load_vector_db()

APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 密钥缺失：请在项目根目录 .env 文件中配置 SPARK_APIPASSWORD")
    st.stop()

# ------------------- RAG & Agent 函数（保持不变） -------------------
def rag_retrieve_answer(question):
    docs = vector_db.similarity_search(question, k=2)
    context = "\n\n".join([d.page_content for d in docs])
    prompt_text = RAG_PROMPT.format(context=context, question=question)
    
    url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {APIPASSWORD}"}
    payload = {
        "model": "spark-x",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.2
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            raw_ans = resp.json()["choices"][0]["message"]["content"]
            return f"**📚 校园知识库解答**：\n\n{raw_ans}"
        else:
            return f"❌ 接口调用失败（{resp.status_code}）"
    except Exception as e:
        return f"⚠️ 网络异常，请稍后再试：{str(e)}"

def agent_answer(question):
    if re.search(r'第几周|现在周数|校历周|目前第几周|当前是几周|本周是第几周', question):
        return f"**📅 校历查询**：\n\n{get_current_week()}"
    
    if re.search(r'绩点|GPA|均分|算绩点|计算平均分|算成绩', question):
        nums = re.findall(r'\d+', question)
        if nums:
            return f"**🧮 绩点计算结果**：\n\n{calculate_gpa(','.join(nums))}"
        else:
            return "💡 请直接输入各科分数，用英文逗号隔开，例如：86,95,88"
    
    return rag_retrieve_answer(question)

# ------------------- 美化后的UI -------------------
st.set_page_config(page_title="安徽交通职业技术学院 - 校园百事通", page_icon="🏫", layout="wide")

# 顶部大标题栏
st.markdown("""
    <div style="background: linear-gradient(135deg, #003087, #0050b3); padding: 25px; border-radius: 10px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin: 0; font-size: 2.2em;">安徽交通职业技术学院</h1>
        <p style="color: #a8d4ff; margin: 8px 0 0 0; font-size: 1.1em;">校园生活百事通助手</p>
    </div>
""", unsafe_allow_html=True)

st.title("🏠 校园生活百事通")

# 左侧边栏
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/003087/ffffff?text=安徽交院", use_container_width=True)
    st.markdown("### 功能导航")
    st.markdown("""
    • 规章制度查询  
    • 请假报修流程  
    • 奖学金政策  
    • 一卡通与宿舍  
    • 图书馆服务  
    • 校历与绩点
    """)
    st.markdown("---")
    st.info("💡 **提示**：可直接输入如“怎么请病假？”、“现在第几周？”、“算绩点 85,92,78”")

# 主界面聊天区
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入你的问题，例如：怎么请病假？现在第几周？算绩点..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("正在检索校园知识库并思考..."):
            answer = agent_answer(prompt)
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
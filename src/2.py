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

# ------------------- 资源加载 -------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh", model_kwargs={"trust_remote_code": True})

@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()
    return Chroma(persist_directory="./vector_db", embedding_function=embeddings)

embeddings = load_embeddings()
vector_db = load_vector_db()

APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 密钥缺失：请在 .env 文件中配置 SPARK_APIPASSWORD")
    st.stop()

# RAG & Agent 函数（保持功能不变）
def rag_retrieve_answer(question):
    docs = vector_db.similarity_search(question, k=2)
    context = "\n\n".join([d.page_content for d in docs])
    prompt_text = RAG_PROMPT.format(context=context, question=question)
    
    url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {APIPASSWORD}"}
    payload = {"model": "spark-x", "messages": [{"role": "user", "content": prompt_text}], "temperature": 0.2}
    
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

# ------------------- 现代化美观UI -------------------
st.set_page_config(page_title="安徽交通职业技术学院AI能力中心", page_icon="🏫", layout="wide")

# 现代渐变顶部大标题
st.markdown("""
    <div style="background: linear-gradient(135deg, #003087 0%, #0050b3 50%, #1e88e5 100%); 
                padding: 40px 20px; border-radius: 0 0 20px 20px; text-align: center; 
                margin-bottom: 10px; box-shadow: 0 10px 30px rgba(0,80,179,0.3);">
        <h1 style="color: white; margin: 0; font-size: 2.8em; font-weight: bold;">
            安徽交通职业技术学院 AI能力中心
        </h1>
        <p style="color: #a8d4ff; margin: 15px 0 0 0; font-size: 1.35em;">
            智启无限可能，创领多元未来，赋能教育变革与突破！
        </p>
    </div>
""", unsafe_allow_html=True)

# 功能亮点卡片
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📋 校园规则", "40+", "已收录")
with col2:
    st.metric("🛠️ 智能工具", "3", "校历・绩点・问答")
with col3:
    st.metric("👥 服务对象", "全校学生", "实时在线")

st.title("🏠 校园生活百事通")

# 主聊天区域提示
st.markdown("**智能问答系统** | 支持请假、奖学金、宿舍、一卡通、图书馆等校园事务咨询")

# 左侧边栏
with st.sidebar:
    st.image("https://via.placeholder.com/220x100/003087/ffffff?text=ACVTC", use_container_width=True)
    st.markdown("### 📌 功能导航")
    st.markdown("""
    • 规章制度智能查询  
    • 请假报修完整流程  
    • 奖学金政策解读  
    • 一卡通与宿舍管理  
    • 图书馆借阅规则  
    • 校历周数与绩点计算
    """)
    st.markdown("---")
    st.success("💡 **推荐提问**：\n- 怎么请病假？\n- 现在是第几周？\n- 算绩点 85,92,78")

# 聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "👋 欢迎使用**校园生活百事通**！\n\n我是你的AI校园助手，有任何校园问题都可以直接问我～"
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入你的校园问题，例如：怎么请病假？现在第几周？算绩点..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索校园知识库并思考..."):
            answer = agent_answer(prompt)
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
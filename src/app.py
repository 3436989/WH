import streamlit as st
import os
import re
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa
import base64
import io
from datetime import datetime

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

# ------------------- 语音功能函数 -------------------
def text_to_speech(text, lang='zh'):
    """文字转语音并返回音频HTML"""
    try:
        from gtts import gTTS
        
        # 清理文本
        clean_text = re.sub(r'[\*\_\#\`\>]', '', text)
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode()
        audio_tag = f'''
            <audio controls style="width: 100%; margin-top: 5px; border-radius: 20px;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                您的浏览器不支持音频播放
            </audio>
        '''
        return audio_tag
    except ImportError:
        return '<p style="color: #999; font-size: 0.9em;">💡 安装gTTS实现语音播报: pip install gTTS</p>'
    except Exception as e:
        return '<p style="color: #999; font-size: 0.9em;">🔊 语音播报暂不可用</p>'

# ------------------- RAG & Agent 函数 -------------------
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

# ------------------- 统一的处理函数 -------------------
def process_user_input(user_input):
    """统一处理用户输入（文字或语音）"""
    if not user_input or not user_input.strip():
        return
    
    # 添加到消息历史
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 获取回答
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索校园知识库并思考..."):
            answer = agent_answer(user_input)
        st.markdown(answer)
        
        # 语音播报
        if st.session_state.get("voice_output_enabled", True):
            audio_html = text_to_speech(answer)
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)
                st.success("🔊 语音播报已开始")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ------------------- 语音输入组件（使用 st.components.v1.html） -------------------
def voice_input_component():
    """使用 st.components.v1.html 实现的语音输入组件"""
    
    voice_html = """
    <div style="display: flex; align-items: center; margin: 10px 0 15px 0; padding: 10px; background: #f5f5f5; border-radius: 10px;">
        <button id="voiceBtn" style="
            background: linear-gradient(135deg, #1a73e8, #0d47a1);
            color: white;
            border: none;
            padding: 10px 28px;
            border-radius: 50px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            margin-right: 10px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(26, 115, 232, 0.3);
        " onclick="toggleRecording()">
            🎤 点击说话
        </button>
        <span id="voiceStatus" style="font-size: 14px; color: #666;">点击按钮，说出你的问题</span>
    </div>
    
    <style>
    .recording {
        background: linear-gradient(135deg, #d32f2f, #b71c1c) !important;
        animation: pulse 1.2s infinite;
        box-shadow: 0 4px 15px rgba(211, 47, 47, 0.3) !important;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.06); }
        100% { transform: scale(1); }
    }
    </style>
    
    <script>
    let recognition = null;
    let isRecording = false;
    
    function toggleRecording() {
        const btn = document.getElementById('voiceBtn');
        const status = document.getElementById('voiceStatus');
        
        // 检查浏览器支持
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            status.textContent = '❌ 请使用 Chrome 或 Edge 浏览器';
            status.style.color = '#d32f2f';
            return;
        }
        
        // 如果正在录音，停止
        if (isRecording && recognition) {
            recognition.stop();
            isRecording = false;
            btn.classList.remove('recording');
            btn.textContent = '🎤 点击说话';
            status.textContent = '⏹️ 已停止录音';
            status.style.color = '#666';
            return;
        }
        
        // 创建识别实例
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'zh-CN';
        recognition.continuous = false;
        recognition.interimResults = true;
        
        // 开始录音
        recognition.start();
        isRecording = true;
        btn.classList.add('recording');
        btn.textContent = '⏹️ 停止录音';
        status.textContent = '🎙️ 正在录音，请说话...';
        status.style.color = '#0d47a1';
        
        // 识别结果处理
        recognition.onresult = function(event) {
            let finalText = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    finalText += event.results[i][0].transcript;
                }
            }
            if (finalText) {
                status.textContent = '✅ 已识别：' + finalText;
                status.style.color = '#2e7d32';
                btn.textContent = '🎤 点击说话';
                btn.classList.remove('recording');
                isRecording = false;
                
                // 通过URL参数传递识别结果
                submitViaURL(finalText);
            }
        };
        
        // 错误处理
        recognition.onerror = function(event) {
            let errorMsg = '';
            switch(event.error) {
                case 'not-allowed':
                    errorMsg = '请允许麦克风权限（点击地址栏右侧的麦克风图标）';
                    break;
                case 'no-speech':
                    errorMsg = '未检测到语音，请重新尝试';
                    break;
                case 'audio-capture':
                    errorMsg = '麦克风访问失败，请检查麦克风设置';
                    break;
                case 'network':
                    errorMsg = '网络连接问题';
                    break;
                default:
                    errorMsg = '错误: ' + event.error;
            }
            status.textContent = '❌ ' + errorMsg;
            status.style.color = '#d32f2f';
            btn.textContent = '🎤 点击说话';
            btn.classList.remove('recording');
            isRecording = false;
        };
        
        // 录音结束处理
        recognition.onend = function() {
            if (isRecording) {
                isRecording = false;
                btn.classList.remove('recording');
                btn.textContent = '🎤 点击说话';
                if (status.textContent.includes('正在录音')) {
                    status.textContent = '⏹️ 录音已结束';
                    status.style.color = '#666';
                }
            }
        };
        
        // 通过URL参数提交
        function submitViaURL(text) {
            try {
                // 获取当前URL
                const currentUrl = new URL(window.parent.location.href);
                // 添加参数
                currentUrl.searchParams.set('voice_text', encodeURIComponent(text));
                // 跳转
                window.parent.location.href = currentUrl.toString();
                console.log('✅ 通过URL提交:', text);
            } catch (error) {
                console.error('❌ URL提交失败:', error);
                status.textContent = '❌ 提交失败，请手动输入';
                status.style.color = '#d32f2f';
            }
        }
    </script>
    """
    
    return st.components.v1.html(voice_html, height=80)

# ------------------- 处理语音输入（通过URL参数） -------------------
def handle_voice_input():
    """处理通过URL参数传递的语音输入"""
    # 获取URL参数
    query_params = st.query_params
    
    if "voice_text" in query_params:
        voice_text = query_params["voice_text"]
        # 解码
        try:
            voice_text = voice_text.encode('utf-8').decode('unicode_escape')
        except:
            pass
        
        # 清除URL参数
        st.query_params.clear()
        
        if voice_text and voice_text.strip():
            # 使用统一的处理函数
            process_user_input(voice_text)
            st.rerun()

# ------------------- 美化后的UI -------------------
st.set_page_config(
    page_title="安徽交通职业技术学院 - 校园百事通", 
    page_icon="🏫", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 全局样式 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 聊天消息样式 */
    .stChatMessage {
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
    }
    
    /* 输入框样式 */
    .stChatInput > div {
        border-radius: 25px !important;
        border: 2px solid #e0e0e0 !important;
        transition: border-color 0.3s ease !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.1) !important;
    }
    
    /* 按钮样式优化 */
    .stButton > button {
        border-radius: 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# 顶部大标题栏 - 带校徽
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <div style="
            width: 70px;
            height: 70px;
            background: linear-gradient(135deg, #ffffff, #f0f4ff);
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 3px solid rgba(255,255,255,0.5);
        ">
            <span style="font-size: 36px;">🏛️</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d47a1, #1565c0, #1a73e8);
        padding: 20px 30px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(13, 71, 161, 0.3);
        margin: 10px 0;
        position: relative;
        overflow: hidden;
    ">
        <div style="position: absolute; top: -50%; right: -20%; width: 200px; height: 200px; background: rgba(255,255,255,0.05); border-radius: 50%;"></div>
        <div style="position: absolute; bottom: -40%; left: -10%; width: 150px; height: 150px; background: rgba(255,255,255,0.03); border-radius: 50%;"></div>
        <h1 style="color: white; margin: 0; font-size: 2.2em; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.2); letter-spacing: 2px;">
            🎓 安徽交通职业技术学院
        </h1>
        <p style="color: #90caf9; margin: 8px 0 0 0; font-size: 1.1em; font-weight: 300; letter-spacing: 4px;">
            校园生活百事通助手
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    current_time = datetime.now().strftime("%H:%M")
    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        color: #1a73e8;
        font-size: 0.9em;
        font-weight: 500;
    ">
        <div style="
            background: rgba(255,255,255,0.9);
            padding: 8px 16px;
            border-radius: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        ">
            🕐 {current_time}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 左侧边栏
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <div style="
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #0d47a1, #1a73e8);
            border-radius: 50%;
            display: inline-flex;
            justify-content: center;
            align-items: center;
            box-shadow: 0 4px 20px rgba(13, 71, 161, 0.3);
            margin-bottom: 10px;
        ">
            <span style="font-size: 40px;">🏛️</span>
        </div>
        <h3 style="color: #0d47a1; margin: 0; font-weight: 600;">校园百事通</h3>
        <p style="color: #666; font-size: 0.85em; margin: 5px 0 0 0;">智能问答助手</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="padding: 5px 0;">
        <p style="font-weight: 600; color: #0d47a1; font-size: 1.05em; margin-bottom: 15px;">📋 功能导航</p>
    </div>
    """, unsafe_allow_html=True)
    
    nav_items = [
        ("📚", "规章制度查询"),
        ("📝", "请假报修流程"),
        ("💰", "奖学金政策"),
        ("💳", "一卡通与宿舍"),
        ("📖", "图书馆服务"),
        ("📅", "校历与绩点")
    ]
    
    for icon, text in nav_items:
        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 8px 12px;
            margin: 4px 0;
            border-radius: 8px;
            transition: all 0.2s ease;
            cursor: default;
        ">
            <span style="font-size: 1.1em; margin-right: 10px;">{icon}</span>
            <span style="color: #444; font-size: 0.95em;">{text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.info("💡 **提示**：可直接输入如\"怎么请病假？\"、\"现在第几周？\"、\"算绩点 85,92,78\"")
    
    st.markdown("""
    <div style="margin: 20px 0 10px 0;">
        <p style="font-weight: 600; color: #0d47a1; font-size: 1.05em;">🎙️ 语音助手</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 语音输入组件
    voice_input_component()
    st.caption("💡 点击「🎤 点击说话」按钮，直接说出你的问题！")
    st.caption("识别后会自动提交并播报回答")
    
    st.markdown("### 🔊 语音播报设置")
    voice_output_enabled = st.checkbox("启用语音播报", value=True, key="voice_output_enabled")
    st.caption("开启后，助手回复将自动朗读")
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75em; color: #999; text-align: center; padding: 10px 0;">
        ⚠️ 语音功能需要Chrome或Edge浏览器<br>
        🏫 安徽交通职业技术学院
    </div>
    """, unsafe_allow_html=True)

# 主界面聊天区
if "messages" not in st.session_state:
    st.session_state.messages = []

# 处理语音输入（通过URL参数）
handle_voice_input()

# 显示聊天历史
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and st.session_state.get("voice_output_enabled", True):
            audio_html = text_to_speech(msg["content"])
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)

# 聊天输入
prompt = st.chat_input("💬 输入你的问题，或点击左侧🎤语音输入...")

if prompt:
    # 使用统一的处理函数
    process_user_input(prompt)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.85em; padding: 10px 0;">
    💡 支持文字输入和语音交互 | 语音识别使用浏览器Web Speech API | 数据来源于校园知识库
</div>
""", unsafe_allow_html=True)

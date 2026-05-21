import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from config import INDEX_FILE, LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from rag_chain import answer_question


st.set_page_config(page_title="计算机专业文档问答", page_icon=":computer:", layout="wide")

st.title("计算机专业文档问答系统")
st.caption(f"Ollama: {OLLAMA_BASE_URL} | 生成模型: {LLM_MODEL}")

if not INDEX_FILE.exists():
    st.warning("知识库尚未构建，请先在终端运行：python src/build_kb.py")
    st.stop()

with st.sidebar:
    st.header("检索参数")
    top_k = st.slider("Top-K 片段数", min_value=1, max_value=8, value=TOP_K)
    st.write("知识库索引：")
    st.code(str(INDEX_FILE), language="text")

question = st.text_area(
    "输入问题",
    placeholder="例如：操作系统为什么需要虚拟内存？",
    height=110,
)

if st.button("开始问答", type="primary", disabled=not question.strip()):
    with st.spinner("正在检索知识库并调用本地模型..."):
        result = answer_question(question.strip(), top_k=top_k)

    st.subheader("回答")
    st.write(result["answer"])

    st.subheader("引用来源")
    for index, source in enumerate(result["sources"], start=1):
        page = f" · 第 {source['page']} 页" if source.get("page") else ""
        with st.expander(
            f"片段 {index}: {source['source']}{page} · 相似度 {source['score']:.4f}",
            expanded=index == 1,
        ):
            st.write(source["content"])

    st.caption(
        f"检索耗时 {result['retrieve_time']:.2f}s | "
        f"生成耗时 {result['generate_time']:.2f}s | "
        f"总耗时 {result['total_time']:.2f}s"
    )

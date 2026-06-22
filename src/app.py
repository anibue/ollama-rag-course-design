import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))

from config import ANSWER_MODES, INDEX_FILE, LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from rag_chain import answer_question, answer_question_adaptive, check_faithfulness, format_context


st.set_page_config(page_title="计算机专业文档问答", page_icon=":computer:", layout="wide")
st.title("计算机专业文档问答系统")
st.caption(f"Ollama: {OLLAMA_BASE_URL} | 生成模型: {LLM_MODEL}")

if not INDEX_FILE.exists():
    st.warning("知识库尚未构建，请先在终端运行：python src/build_kb.py")
    st.stop()

# 对话只存在当前页面会话里，不写入磁盘，答辩演示时清空也方便。
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, result}]

# 侧边栏只放会影响结果的参数，避免页面看起来像调试面板。
with st.sidebar:
    st.header("检索参数")
    top_k = st.slider("Top-K 片段数", min_value=1, max_value=8, value=TOP_K)
    retrieval_mode = st.selectbox(
        "检索策略",
        ANSWER_MODES,
        index=ANSWER_MODES.index("adaptive") if "adaptive" in ANSWER_MODES else 1,
        help="hybrid=向量+TF-IDF+实体匹配，adaptive=自动扩大二轮检索，vector=纯向量",
    )
    prompt_variant = st.selectbox(
        "Prompt 策略",
        ["stepwise", "baseline", "reflective"],
        help="stepwise=分步推理(推荐)，baseline=直接回答，reflective=生成后自反思",
    )
    enable_faithfulness = st.checkbox(
        "开启忠实度自检",
        value=False,
        help="生成后调用 LLM 检查答案是否有参考资料依据（增加约 1 次生成耗时）",
    )
    st.divider()
    if st.button("清空对话历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.write("知识库索引：")
    st.code(str(INDEX_FILE), language="text")

# ---- Chat history ----------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- Input -----------------------------------------------------------------
question = st.chat_input("请输入问题，例如：操作系统为什么需要虚拟内存？")

if question and question.strip():
    question = question.strip()
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并调用本地模型..."):
            try:
                if retrieval_mode == "adaptive":
                    result = answer_question_adaptive(question, top_k=top_k)
                else:
                    result = answer_question(
                        question,
                        top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        prompt_variant=prompt_variant,
                    )
            except Exception as exc:
                st.error(f"调用失败：{exc}")
                st.stop()

        st.write(result["answer"])

        # 忠实度自检比较慢，所以让用户手动勾选。
        if enable_faithfulness:
            with st.spinner("正在进行忠实度自检..."):
                context_str = format_context(
                    [{"id": s["id"], "content": s["content"], "source": s["source"],
                      "page": s.get("page"), "score": s["score"]} for s in result["sources"]]
                )
                faith = check_faithfulness(result["answer"], context_str)
            score = faith.get("faithfulness_score")
            label = f"{score*10:.0f}/10" if score is not None else "N/A"
            st.info(f"忠实度评分：{label}")
            with st.expander("查看忠实度详情"):
                st.text(faith.get("check_result", ""))

        # 展开来源片段，方便检查模型答案到底参考了哪几段材料。
        with st.expander(
            f"检索来源 ({len(result['sources'])} 片段) — "
            f"检索 {result['retrieve_time']:.2f}s | 生成 {result['generate_time']:.2f}s | "
            f"总计 {result['total_time']:.2f}s"
            + (f" | 检索轮数: {result.get('retrieval_rounds', 1)}" if retrieval_mode == "adaptive" else "")
        ):
            for i, src in enumerate(result["sources"], start=1):
                page = f" · 第 {src['page']} 页" if src.get("page") else ""
                st.markdown(
                    f"**片段 {i}: {src['source']}{page}**  \n"
                    f"综合分: `{src['score']:.4f}` · "
                    f"向量: `{src['embedding_score']:.4f}` · "
                    f"词汇: `{src['lexical_score']:.4f}` · "
                    f"实体: `{src['entity_score']:.4f}`"
                )
                st.caption(src["content"][:300] + ("..." if len(src["content"]) > 300 else ""))

        if retrieval_mode == "adaptive" and result.get("sufficiency_judgment"):
            st.caption(f"充分性判断：{result['sufficiency_judgment']}")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

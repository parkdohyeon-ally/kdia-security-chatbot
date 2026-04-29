"""
디롱이(SecureGuide) - Streamlit 웹 앱 (라이트 테마 + KDIA 로고)
"""
import re
import os
import time
import base64
from pathlib import Path
from datetime import date

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── 페이지 설정 ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="한국디스플레이산업협회 · 디스플레이산업 보안가이드",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 로고 base64 변환 ──────────────────────────────────────────────────
def get_image_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

LOGO_B64 = get_image_base64("kdia_logo.png")

def logo_img_tag(size: int = 26) -> str:
    """아바타용 KDIA 로고 img 태그. 파일 없으면 fallback 텍스트."""
    if LOGO_B64:
        return f'<img src="data:image/png;base64,{LOGO_B64}" style="width:{size}px;height:{size}px;object-fit:contain;">'
    return "🛡️"

# ── 스타일 ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #f5f6fa;
    color: #1e2340;
    font-family: 'Noto Sans KR', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e6f0;
}

/* 헤더 */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 8px 0 24px 0;
    border-bottom: 2px solid #e2e6f0;
    margin-bottom: 24px;
}
.app-header .logo-wrap {
    background: #ffffff;
    border: 1px solid #e2e6f0;
    border-radius: 10px;
    padding: 8px 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
.app-header h1 {
    font-size: 1.6rem; font-weight: 700;
    color: #1e2340; margin: 0; letter-spacing: -0.5px;
}
.app-header .sub {
    font-size: 0.78rem; color: #8891b0;
    margin-top: 2px; font-weight: 300;
}

/* 사용자 말풍선 */
.msg-user { display: flex; justify-content: flex-end; margin: 12px 0; }
.msg-user .bubble {
    background: #1e40af; color: #ffffff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px; max-width: 72%;
    font-size: 0.93rem; line-height: 1.6;
    box-shadow: 0 2px 8px rgba(30,64,175,0.18);
}

/* 봇 말풍선 */
.msg-bot {
    display: flex; justify-content: flex-start;
    margin: 12px 0; gap: 10px; align-items: flex-start;
}
.msg-bot .avatar {
    width: 38px; height: 38px;
    background: #ffffff;
    border: 1px solid #e2e6f0;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 2px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    padding: 4px;
}
.msg-bot .bubble {
    background: #ffffff; border: 1px solid #e2e6f0;
    color: #1e2340; border-radius: 4px 18px 18px 18px;
    padding: 14px 18px; max-width: 82%;
    font-size: 0.93rem; line-height: 1.7;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.msg-bot .bubble b { color: #1d4ed8; }

/* 메타 태그 */
.meta-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.meta-tag {
    background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; font-weight: 600;
}
.meta-tag.type {
    background: #f0fdf4; border-color: #bbf7d0; color: #15803d;
}

/* 출처 */
.source-section { margin-top: 10px; border-top: 1px solid #e2e6f0; padding-top: 8px; }
.source-toggle { font-size: 0.75rem; color: #8891b0; margin-bottom: 4px; }
.source-item {
    background: #f8f9fd; border: 1px solid #e2e6f0; border-radius: 6px;
    padding: 7px 12px; margin: 3px 0;
    font-size: 0.74rem; color: #5a6480; font-family: 'JetBrains Mono', monospace;
}

/* 가이드 카드 */
.guide-card {
    background: #f8f9fd; border: 1px solid #e2e6f0;
    border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;
}
.guide-card .gen {
    font-size: 0.7rem; font-weight: 700; color: #2563eb;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 3px;
}
.guide-card .title { font-size: 0.83rem; color: #1e2340; font-weight: 600; margin-bottom: 2px; }
.guide-card .year { font-size: 0.72rem; color: #8891b0; }
.guide-card.current { background: #eff6ff; border-color: #93c5fd; }
.guide-card.current .gen { color: #059669; }

/* 사용량 바 */
.limit-bar {
    background: #e2e6f0; border-radius: 20px;
    height: 7px; margin: 6px 0 2px 0; overflow: hidden;
}
.limit-fill {
    height: 100%; border-radius: 20px;
    background: linear-gradient(90deg, #2563eb, #60a5fa);
    transition: width 0.3s ease;
}
.limit-fill.warn { background: linear-gradient(90deg, #d97706, #fbbf24); }
.limit-fill.full { background: linear-gradient(90deg, #dc2626, #f87171); }

/* 버튼 */
.stButton > button {
    background: #f8f9fd !important; border: 1px solid #e2e6f0 !important;
    color: #3d4a6b !important; border-radius: 8px !important;
    font-size: 0.8rem !important; padding: 6px 12px !important;
    text-align: left !important; width: 100% !important;
    transition: all 0.18s !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}
.stButton > button:hover {
    border-color: #93c5fd !important; color: #1d4ed8 !important;
    background: #eff6ff !important;
}

/* 입력창 */
[data-testid="stChatInput"] textarea {
    background: #ffffff !important; border: 1.5px solid #e2e6f0 !important;
    color: #1e2340 !important; border-radius: 12px !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* 알림 박스 */
.rate-warn {
    background: #fffbeb; border: 1px solid #fcd34d;
    border-radius: 8px; padding: 10px 14px;
    color: #92400e; font-size: 0.85rem; margin: 8px 0;
}
.err-box {
    background: #fef2f2; border: 1px solid #fca5a5;
    border-radius: 8px; padding: 10px 14px;
    color: #991b1b; font-size: 0.85rem; margin: 8px 0;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f5f6fa; }
::-webkit-scrollbar-thumb { background: #d1d5e8; border-radius: 3px; }
hr { border-color: #e2e6f0; }
</style>
""", unsafe_allow_html=True)

# ── 상수 ─────────────────────────────────────────────────────────────
DAILY_LIMIT = 30
MIN_INTERVAL = 4

QUERY_TYPE_LABELS = {
    "A": "단일 주제", "B": "기수 지정",
    "C": "기수 비교", "D": "절차/방법", "E": "가이드 외",
}

SAMPLE_QUESTIONS = [
    "수출승인 절차를 단계별로 알려줘",
    "해외사업장 보호구역 분류 방법은?",
    "퇴직자 경업금지 조항 어떻게 넣어?",
    "유료자문업체 대응 방법은?",
    "재택근무 보안관리 방안은?",
    "1기, 2기, 3기 가이드 차이점은?",
    "국가핵심기술 수출 시 필요한 서류는?",
    "장비 시운전 시 보안 위험은?",
]


# ── 세션 초기화 ───────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [], "question_count": 0,
        "last_date": date.today(), "last_request": 0.0,
        "chain": None, "pending_question": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.last_date != date.today():
        st.session_state.question_count = 0
        st.session_state.last_date = date.today()
# app.py 상단 load_chain() 함수 위에 추가
from pathlib import Path

def ensure_vectorstore():
    """vectorstore가 없으면 자동으로 생성"""
    from config import VECTORSTORE_DIR, PDF_DIR
    vs_path = Path(VECTORSTORE_DIR)
    
    if not vs_path.exists() or not any(vs_path.iterdir()):
        st.info("🔧 처음 실행입니다. 가이드를 읽어들이는 중... (3~5분 소요)")
        from src.pdf_loader import load_all_guides
        from src.vectorstore import build_vectorstore
        chunks = load_all_guides()
        if chunks:
            build_vectorstore(chunks)
            st.success("✅ 준비 완료!")
            st.rerun()
        else:
            st.error("❌ PDF 파일을 찾을 수 없습니다. data/pdfs/ 폴더를 확인하세요.")
            st.stop()

@st.cache_resource(show_spinner=False)
def load_chain():
    from src.chain import SecurityGuideChain
    return SecurityGuideChain()


# ── 렌더링 헬퍼 ──────────────────────────────────────────────────────
def render_meta_tags(result: dict) -> str:
    tags = []
    qtype = result.get("query_type", "A")
    tags.append(f'<span class="meta-tag type">{QUERY_TYPE_LABELS.get(qtype, qtype)}</span>')
    for v in result.get("specified_versions", []):
        tags.append(f'<span class="meta-tag">{v}</span>')

    for prefix, keys in [
        ("1기", ["business_type", "process_gen1"]),
        ("2기", ["lifecycle_stage", "lifecycle_item"]),
        ("3기", ["overseas_domain", "procedure_type"]),
    ]:
        parts = [result.get(k, "미지정") for k in keys if result.get(k, "미지정") != "미지정"]
        if parts:
            tags.append(f'<span class="meta-tag">{prefix} · {" / ".join(parts)}</span>')

    if result.get("risk_level", "미지정") != "미지정":
        tags.append(f'<span class="meta-tag">등급:{result["risk_level"]}</span>')

    return f'<div class="meta-row">{"".join(tags)}</div>' if tags else ""


def render_sources(docs: list) -> str:
    if not docs:
        return ""
    items = []
    for doc in docs:
        v = doc.metadata.get("version", "?")
        p = doc.metadata.get("page", "?")
        if v == "1기":
            proc = doc.metadata.get("process", "?")
            ctype = doc.metadata.get("content_type", "?")
            label = f"1기 | {proc} | {ctype} | p.{p}"
        elif v == "2기":
            stage = doc.metadata.get("lifecycle_stage", "?")
            item = doc.metadata.get("lifecycle_item", "?")
            label = f"2기 | {stage} · {item} | p.{p}"
        elif v == "3기":
            domain = doc.metadata.get("overseas_domain", "N/A")
            risk_id = doc.metadata.get("risk_id", "N/A")
            risk_level = doc.metadata.get("risk_level", "N/A")
            procedure_type = doc.metadata.get("procedure_type", "N/A")
            chapter = doc.metadata.get("gen3_chapter", "?")
            if domain != "N/A":
                risk_str = f" [{risk_id}/{risk_level}]" if risk_id != "N/A" else ""
                label = f"3기 | Ⅴ장 {domain}{risk_str} | p.{p}"
            elif procedure_type != "N/A":
                label = f"3기 | Ⅲ장 {procedure_type} | p.{p}"
            else:
                label = f"3기 | {chapter} | p.{p}"
        else:
            label = f"{v} | p.{p}"
        items.append(f'<div class="source-item">📄 {label}</div>')

    return (
        '<div class="source-section">'
        f'<div class="source-toggle">📚 참조 문서 {len(docs)}개</div>'
        + "".join(items) + "</div>"
    )


def render_messages():
    # 에러 메시지 표시
    if "last_error" in st.session_state:
        err = st.session_state["last_error"]
        if "한도" in err or "⏳" in err:
            st.markdown(f'<div class="rate-warn">{err}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="err-box">{err}</div>', unsafe_allow_html=True)

    avatar = f'<div class="avatar">{logo_img_tag(26)}</div>'
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user"><div class="bubble">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            result = msg.get("result", {})
            clean = re.sub(r'</?(?:div|span|p|br)[^>]*>', '', msg["content"])
            # 페이지 자리표시자 제거
            clean = clean.replace("p.XX", "").replace("PART X", "").replace("p.X", "")
            answer = clean.replace("\n", "<br>")
            st.markdown(
                f'''<div class="msg-bot">
                    {avatar}
                    <div class="bubble">
                        {render_meta_tags(result)}
                        {answer}
                        {render_sources(result.get("source_documents", []))}
                    </div>
                </div>''',
                unsafe_allow_html=True,
            )


def render_sidebar():
    with st.sidebar:
        # 사이드바 상단 로고
        if LOGO_B64:
            st.markdown(
                f'<div style="text-align:center;padding:12px 0 4px 0;">'
                f'<img src="data:image/png;base64,{LOGO_B64}" style="height:48px;width:auto;object-fit:contain;"></div>',
                unsafe_allow_html=True,
            )
        st.markdown("### 📚 가이드 시리즈")

# 1기
        st.markdown("""
        <div class="guide-card">
            <div class="gen">1기 · 2019.08</div>
            <div class="title">디스플레이산업 실무 보안가이드</div>
            <div class="year">장비 / 부품·소재 기업용</div>
        </div>
        """, unsafe_allow_html=True)

        with open("data/pdfs/1기/[1기] 디스플레이산업 실무보안가이드_FN.pdf", "rb") as f:
            st.download_button(
            label="📥 1기 가이드 다운로드",
            data=f,
            file_name="1기_디스플레이산업_실무보안가이드.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_1"
            )

# 2기
        st.markdown("""
        <div class="guide-card">
            <div class="gen">2기 · 2022.03</div>
            <div class="title">핵심인력 보안가이드</div>
            <div class="year">채용 / 재직 / 퇴사 전주기</div>
        </div>
        """, unsafe_allow_html=True)

        with open("data/pdfs/2기/[2기] 디스플레이산업 핵심인력보안가이드_FN.pdf", "rb") as f:
            st.download_button(
            label="📥 2기 가이드 다운로드",
            data=f,
            file_name="2기_디스플레이산업_핵심인력보안가이드.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_2"
            )

# 3기
        st.markdown("""
        <div class="guide-card">
            <div class="gen">3기 · 2024.09</div>
            <div class="title">수출 보안 가이드</div>
            <div class="year">해외사업장 · 수출절차</div>
        </div>
        """, unsafe_allow_html=True)

        with open("data/pdfs/3기/[3기] 디스플레이산업 수출보안가이드_FN.pdf", "rb") as f:
            st.download_button(
            label="📥 3기 가이드 다운로드",
            data=f,
            file_name="3기_디스플레이산업_수출보안가이드.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_3"
            )
        st.markdown("---")

        count = st.session_state.get("question_count", 0)
        pct = int(count / DAILY_LIMIT * 100)
        fill_class = "full" if pct >= 100 else "warn" if pct >= 70 else ""
        st.markdown("### 📊 오늘 사용량")
        st.markdown(
            f'<div class="limit-bar"><div class="limit-fill {fill_class}" style="width:{min(pct,100)}%"></div></div>'
            f'<div style="font-size:0.75rem;color:#8891b0;">{count} / {DAILY_LIMIT}회</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### 💡 추천 질문")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sq_{q}"):
                st.session_state.pending_question = q

        st.markdown("---")
        if st.button("🗑️ 대화 초기화"):
            st.session_state.messages = []
            st.rerun()

        st.markdown(
            '<div style="font-size:0.7rem;color:#aab0c8;margin-top:16px;">'
            'Powered by Groq · Llama 3.3 70B<br>'
            'KDIA 디스플레이산업 보안가이드</div>',
            unsafe_allow_html=True,
        )


def process_question(question: str):
    if st.session_state.question_count >= DAILY_LIMIT:
        st.markdown('<div class="rate-warn">⚠️ 오늘 질문 한도를 모두 사용했습니다. 내일 다시 시도해주세요.</div>', unsafe_allow_html=True)
        return

    elapsed = time.time() - st.session_state.last_request
    if elapsed < MIN_INTERVAL:
        wait = int(MIN_INTERVAL - elapsed) + 1
        st.markdown(f'<div class="rate-warn">⏱️ {wait}초 후 다시 질문해주세요.</div>', unsafe_allow_html=True)
        return

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.question_count += 1
    st.session_state.last_request = time.time()

    # 에러 메시지를 세션에 저장해서 rerun 후에도 표시
    error_msg = None

    with st.spinner("🔍 가이드를 검색하고 있습니다..."):
        try:
            result = st.session_state.chain.invoke(question)
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "result": result,
            })
        except Exception as e:
            error_msg = f"❌ 실제 에러: {type(e).__name__}: {str(e)}"
            st.session_state.messages.pop()

    # 에러 메시지를 세션에 저장
    if error_msg:
        st.session_state["last_error"] = error_msg
    else:
        st.session_state.pop("last_error", None)

    st.rerun()

def main():
    init_session()
    ensure_vectorstore()
    render_sidebar()

    # 헤더 — KDIA 로고 카드
    if LOGO_B64:
        logo_html = (
            f'<div class="logo-wrap">'
            f'<img src="data:image/png;base64,{LOGO_B64}" style="height:46px;width:auto;object-fit:contain;">'
            f'</div>'
        )
    else:
        logo_html = '<div style="font-size:2.4rem;">🛡️</div>'

    st.markdown(f"""
    <div class="app-header">
        {logo_html}
        <div>
            <h1>한국디스플레이산업협회 · SecureGuide</h1>
            <div class="sub">디스플레이산업 보안가이드 1기 / 2기 / 3기 통합 RAG 챗봇</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.chain is None:
        with st.spinner("🔧 챗봇 초기화 중... (처음 실행 시 30초~1분 소요됩니다)"):
            try:
                st.session_state.chain = load_chain()
            except Exception as e:
                st.markdown(
                    f'<div class="err-box">❌ 초기화 실패: {e}<br>'
                    'GROQ_API_KEY와 벡터스토어(data/vectorstore/)를 확인하세요.</div>',
                    unsafe_allow_html=True,
                )
                return

    # 웰컴 메시지
    if not st.session_state.messages:
        avatar = f'<div class="avatar">{logo_img_tag(26)}</div>'
        st.markdown(f"""
        <div class="msg-bot">
            {avatar}
            <div class="bubble">
                안녕하세요! 저는 <b>한국디스플레이산업협회</b>입니다. 😊<br><br>
                디스플레이산업 보안가이드 <b>1기 · 2기 · 3기</b>를 바탕으로 질문에 답변드립니다.<br><br>
                • <b>1기</b>: 장비/부품·소재 기업 실무 보안<br>
                • <b>2기</b>: 핵심인력 채용·재직·퇴사 보안<br>
                • <b>3기</b>: 해외수출·사업장 보안<br><br>
                왼쪽 추천 질문을 눌러보거나, 직접 질문해주세요!
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_messages()

    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        process_question(q)

    if user_input := st.chat_input("디스플레이산업 보안에 대해 질문해주세요..."):
        process_question(user_input)


if __name__ == "__main__":
    main()

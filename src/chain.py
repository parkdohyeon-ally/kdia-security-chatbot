"""
RAG 체인 (1기/2기/3기 통합)
"""
import re
from typing import Dict, List, Any
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from config import LLM_MODEL, LLM_TEMPERATURE, GROQ_API_KEY
from src.prompts import SYSTEM_PROMPT, HUMAN_PROMPT
from src.query_classifier import classify_query, QueryClassification
from src.retriever import retrieve, format_context
from src.vectorstore import load_vectorstore

QUERY_TYPE_LABELS = {
    "A": "유형 A - 단일 주제 질문",
    "B": "유형 B - 특정 기수 지정 질문",
    "C": "유형 C - 버전 비교 질문",
    "D": "유형 D - 절차/실행 방법 질문",
    "E": "유형 E - 가이드 외 질문",
}

class SecurityGuideChain:
    def __init__(self):
        print("🔧 챗봇 초기화 중...")
        self.vectorstore = load_vectorstore()
        self.llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=1200,
            api_key=GROQ_API_KEY,
        )
        print("✅ 초기화 완료\n")

    def invoke(self, question: str) -> Dict[str, Any]:
        classification: QueryClassification = classify_query(question)
        query_type_label = QUERY_TYPE_LABELS[classification.type]
        documents: List[Document] = retrieve(self.vectorstore, question, classification)

        # 3기 Ⅲ장 절차 질문이면 법령 청크 강제 추가
        # — 문서 메타데이터가 잘못 태깅된 경우도 커버하기 위해 분류기 결과를 우선 사용
        has_gen3_proc = (
            classification.gen3_chapter == "Ⅲ_보호제도절차"
            or classification.procedure_type != "미지정"
            or any(d.metadata.get("gen3_chapter") == "Ⅲ_보호제도절차" for d in documents)
        )
        _PROC_LAW_QUERIES = {
            "수출승인":   "산업기술보호법 제11조 국가핵심기술 수출 승인 연구개발비",
            "수출신고":   "산업기술보호법 제11조 국가핵심기술 수출 신고",
            "해외인수합병": "산업기술보호법 제11조의2 해외인수합병 승인 신고",
            "사전검토":   "산업기술보호법 제11조 사전검토",
            "기술판정":   "산업기술보호법 제9조 국가핵심기술 판정",
            "침해신고":   "산업기술보호법 제14조 침해행위 신고",
        }
        if has_gen3_proc:
            proc = classification.procedure_type
            search_q = _PROC_LAW_QUERIES.get(
                proc, "산업기술보호법 제11조 국가핵심기술 수출"
            )
            law_results = self.vectorstore.similarity_search(search_q, k=50)
            seen_page_contents = {d.page_content for d in documents}
            law_added = 0
            for doc in law_results:
                if doc.metadata.get("version") == "법령" and doc.page_content not in seen_page_contents:
                    documents.append(doc)
                    seen_page_contents.add(doc.page_content)
                    law_added += 1
                    if law_added >= 3:
                        break

        context = format_context(documents)
        system_message = SystemMessage(
            content=SYSTEM_PROMPT.format(
                context=context,
                query_type=query_type_label,
            )
        )
        human_message = HumanMessage(content=HUMAN_PROMPT.format(question=question))
        try:
            response = self.llm.invoke([system_message, human_message])
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["rate", "429", "limit", "quota", "exceeded"]):
                raise RuntimeError("RATE_LIMIT")
            raise

        answer = response.content

        # LLM이 [관련 법령] 섹션을 직접 쓴 경우 제거 (app.py에서 별도 렌더링)
        answer = re.sub(
            r'\*{0,2}\[관련 법령\]\*{0,2}[^\n]*\n?', '', answer
        ).rstrip()

        return {
            "answer": answer,
            "query_type": classification.type,
            "query_type_label": query_type_label,
            "specified_versions": classification.specified_versions,
            "business_type": classification.business_type,
            "process_gen1": classification.process_gen1,
            "lifecycle_stage": classification.lifecycle_stage,
            "lifecycle_item": classification.lifecycle_item,
            "gen3_chapter": classification.gen3_chapter,
            "overseas_domain": classification.overseas_domain,
            "risk_level": classification.risk_level,
            "measure_type": classification.measure_type,
            "procedure_type": classification.procedure_type,
            "source_documents": documents,
        }

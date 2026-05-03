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
            max_tokens=2000,
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
        if has_gen3_proc:
            proc = classification.procedure_type
            search_q = (
                f"{proc} 산업기술보호법 국가핵심기술 수출 {question}"
                if proc != "미지정"
                else f"{question} 산업기술보호법 제11조 국가핵심기술 수출"
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

        # LLM이 [관련 법령] 섹션을 쓴 경우 제거 (원문으로 대체할 것)
        answer = re.sub(
            r'\*{0,2}\[관련 법령\]\*{0,2}[^\n]*\n?', '', answer
        ).rstrip()

        # 검색된 법령 청크를 답변 끝에 무조건 추가
        law_docs = [d for d in documents if d.metadata.get("version") == "법령"]
        if law_docs:
            law_parts = []
            for doc in law_docs[:3]:
                law_name = doc.metadata.get("law_name", "")
                law_article = doc.metadata.get("law_article", "")
                clean_content = re.sub(
                    r'\[법령명:[^\]]+\]\s*', '', doc.page_content
                ).strip()
                # <개정>, <신설> 등이 HTML 태그로 파싱되는 것 방지
                clean_content = clean_content.replace('<', '&lt;').replace('>', '&gt;')
                if clean_content:
                    law_parts.append(f"「{law_name}」 {law_article}\n{clean_content}")
            if law_parts:
                answer += "\n\n**[관련 법령]** ⚖️\n\n" + "\n\n---\n\n".join(law_parts)

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

"""
RAG 체인 (1기/2기/3기 통합)
"""
import os
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
            max_tokens=800,
            api_key=GROQ_API_KEY,
        )
        print("✅ 초기화 완료\n")

    def invoke(self, question: str) -> Dict[str, Any]:
        classification: QueryClassification = classify_query(question)
        query_type_label = QUERY_TYPE_LABELS[classification.type]
        documents: List[Document] = retrieve(self.vectorstore, question, classification)
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
        return {
            "answer": response.content,
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

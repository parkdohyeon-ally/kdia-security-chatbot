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
            max_tokens=1500,
            api_key=GROQ_API_KEY,
        )
        print("✅ 초기화 완료\n")

    def invoke(self, question: str) -> Dict[str, Any]:
        classification: QueryClassification = classify_query(question)
        query_type_label = QUERY_TYPE_LABELS[classification.type]
        documents: List[Document] = retrieve(self.vectorstore, question, classification)

        # 3기 Ⅲ장 절차 질문이면 법령 청크 강제 추가
        has_gen3_proc = any(
            d.metadata.get("gen3_chapter") == "Ⅲ_보호제도절차" for d in documents
        )
        if has_gen3_proc:
            law_results = self.vectorstore.similarity_search(
                f"{question} 산업기술보호법 제11조 국가핵심기술 수출", k=10
            )
            existing_ids = {id(d) for d in documents}
            law_added = 0
            for doc in law_results:
                if doc.metadata.get("version") == "법령" and id(doc) not in existing_ids:
                    documents.append(doc)
                    existing_ids.add(id(doc))
                    law_added += 1
                    if law_added >= 2:
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

        # 법령 원문 텍스트 생성
        law_docs = [d for d in documents if d.metadata.get("version") == "법령"]
        if law_docs:
            law_text = "**[관련 법령]** ⚖️"
            added = False
            for doc in law_docs:
                law_name = doc.metadata.get("law_name", "")
                law_article = doc.metadata.get("law_article", "")
                article_num = law_article.split("(")[0] if law_article else ""
                if article_num and article_num in answer:
                    clean_content = re.sub(
                        r'\[법령명:[^\]]+\]\s*', '', doc.page_content
                    ).strip()
                    law_text += f"\n\n「{law_name}」 {law_article}\n{clean_content}"
                    added = True

            if added:
                # LLM이 만든 [관련 법령] 섹션을 실제 원문으로 교체
                if "[관련 법령]" in answer:
                    answer = re.sub(
                        r'\*{0,2}\[관련 법령\]\*{0,2}.*?(?=\*{0,2}\[출처\]|\Z)',
                        law_text + "\n\n",
                        answer,
                        flags=re.DOTALL,
                    )
                else:
                    if "[출처]" in answer:
                        answer = answer.replace("[출처]", law_text + "\n\n**[출처]**")
                    else:
                        answer += "\n\n" + law_text

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

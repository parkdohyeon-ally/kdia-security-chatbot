"""
벡터 DB(ChromaDB) 관리 모듈
- 신규 구축 (build_vectorstore)
- 기존 로드 (load_vectorstore)
"""
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
)

COLLECTION_NAME = "security_guides"


def get_embeddings() -> HuggingFaceEmbeddings:
    """임베딩 모델 인스턴스 생성 (로컬 무료 실행)"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """청크 리스트로부터 벡터 DB를 신규 구축합니다."""
    print(f"\n🔧 벡터 DB 구축 시작 (저장 경로: {VECTORSTORE_DIR})")

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
    )

    print(f"✅ 벡터 DB 구축 완료 ({len(chunks)}개 청크 임베딩)")
    return vectorstore


def load_vectorstore() -> Chroma:
    """기존 구축된 벡터 DB를 로드합니다."""
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    return vectorstore
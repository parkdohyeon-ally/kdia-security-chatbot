"""
벡터 DB(FAISS) 관리 모듈
"""
from typing import List
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import VECTORSTORE_DIR, EMBEDDING_MODEL

FAISS_INDEX_PATH = str(VECTORSTORE_DIR)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: List[Document]) -> FAISS:
    print(f"\n🔧 벡터 DB 구축 시작 (저장 경로: {VECTORSTORE_DIR})")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"✅ 벡터 DB 구축 완료 ({len(chunks)}개 청크 임베딩)")
    return vectorstore


def load_vectorstore() -> FAISS:
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore

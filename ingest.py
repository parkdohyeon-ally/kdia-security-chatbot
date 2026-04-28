"""
PDF 적재 스크립트 (1회 실행)

사용법:
    python ingest.py

사전 준비:
    data/pdfs/1기/ 에 1기 가이드 PDF 배치
    data/pdfs/2기/ 에 2기 가이드 PDF 배치
    data/pdfs/3기/ 에 3기 가이드 PDF 배치
"""
from src.pdf_loader import load_all_guides
from src.vectorstore import build_vectorstore


def main():
    print("=" * 70)
    print("📥 보안가이드 PDF 적재 시작")
    print("=" * 70)
    
    # 1. PDF 로딩 + 청킹
    chunks = load_all_guides()
    
    if not chunks:
        print("❌ 적재할 청크가 없습니다. data/pdfs/ 폴더를 확인하세요.")
        return
    
    # 2. 벡터 DB 구축
    build_vectorstore(chunks)
    
    print("\n" + "=" * 70)
    print("✅ 모든 작업 완료. 이제 'python main.py'로 챗봇을 실행하세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()

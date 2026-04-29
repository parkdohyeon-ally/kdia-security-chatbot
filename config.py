"""
환경 설정 중앙 관리 모듈 (디스플레이산업 보안가이드 1기/2기/3기 통합 RAG)

세 기수는 분류 체계가 모두 다릅니다:
- 1기: 기업유형(장비/부품·소재) × 업무PROCESS × RISK/COUNTERMEASURE
- 2기: 인력생애주기(채용/재직/퇴사) × 보안관리항목 × 보안사고/우수사례
- 3기: 6개 챕터 + Ⅴ장 해외사업장 6개 영역 × 위험등급(상/중/하) × 대응방안(필수/선택)

따라서 메타데이터 스키마를 세 가지 구조를 모두 담도록 설계했습니다.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === 경로 설정 ===
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

# === API 키 ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === 모델 설정 ===
LLM_MODEL = "llama-3.3-70b-versatile"

# HuggingFace 한국어 임베딩 모델 (무료, 로컬 실행)
EMBEDDING_MODEL ="jhgan/ko-sroberta-multitask"
LLM_TEMPERATURE = 0.1

# === 청킹 설정 ===
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# === 검색 설정 ===
DEFAULT_TOP_K = 5
COMPARISON_TOP_K_PER_VERSION = 4

# === 디스플레이산업 보안가이드 메타정보 (실제 PDF 기준) ===
GUIDE_METADATA = {
    "1기": {
        "full_title": "2019 디스플레이산업 실무 보안가이드 (장비·부품·소재 기업용)",
        "publisher": "한국디스플레이산업협회 (KDIA), 디스플레이산업보안협의회",
        "issued_date": "2019-08",
        "effective_period": "2019-08 ~",
        "scope": "디스플레이 장비/부품/소재 공급 기업의 산업기술 유출방지 실무 가이드",
        "structure_axis": "기업유형 (장비공급기업 / 부품·소재공급기업)",
        "content_pattern": "PROCESS → 보안위험(RISK) → 보안대책(COUNTERMEASURE)",
        "is_current": False,
    },
    "2기": {
        "full_title": "2022 디스플레이산업 핵심인력 보안가이드",
        "publisher": "한국디스플레이산업협회 + 국가정보원 산업기밀보호센터 + 중앙대학교",
        "issued_date": "2022-03",
        "effective_period": "2022-03 ~",
        "scope": "디스플레이 핵심기술을 취급하는 핵심인력의 채용/재직/퇴사 전주기 보안관리",
        "structure_axis": "인력생애주기 (채용 / 재직 / 퇴사)",
        "content_pattern": "보안관리항목 본문 → 보안사고 사례 또는 보안우수 사례",
        "is_current": False,
    },
    "3기": {
        "full_title": "2024 디스플레이산업 수출 보안 가이드",
        "publisher": "한국디스플레이산업협회 + 중앙대학교 산업보안학과 + 국가정보원 산업기밀보호센터",
        "issued_date": "2024-09",
        "effective_period": "2024-09 ~",
        "scope": "디스플레이 산업기술의 해외 수출·이전·M&A 및 해외사업장 보안관리",
        "structure_axis": "해외사업장 보안관리 6개 영역 (기술정보/기술보호지원/외부위험/시설/인력/업무환경)",
        "content_pattern": "보안위험사례(상/중/하) → 보안관리방안(필수/선택)",
        "is_current": False,  # 3기가 현재 최신
    },
}

VERSIONS = list(GUIDE_METADATA.keys())
CURRENT_VERSION = "3기"  # 3기가 현재 최신

# === 1기 도메인 분류 (기업유형 × 업무PROCESS) ===
BUSINESS_TYPES_GEN1 = ["장비공급기업", "부품·소재공급기업", "공통"]

PROCESS_TYPES_GEN1 = {
    "장비공급기업": ["수주활동", "설계및제작", "설치및시운전", "준공과사후관리"],
    "부품·소재공급기업": ["수주활동", "개발및평가", "제조와생산", "사후관리"],
}

# === 2기 도메인 분류 (인력생애주기 × 보안관리항목) ===
LIFECYCLE_STAGES_GEN2 = ["채용", "재직", "퇴사"]

LIFECYCLE_ITEMS_GEN2 = {
    "채용": [
        "채용검증",
        "보안서약",
    ],
    "재직": [
        "핵심인력대상선정",
        "산업보안담당자지정",
        "핵심인력산업보안교육",
        "산업보안문화조성",
        "보안활동면담관리",
        "외부접촉보안관리",
        "자문중개업체보안대응",
        "재택근무보안관리",
        "해외사업장파견보안관리",
        "기술유출징후탐지와대응",
        "고용계약갱신",
    ],
    "퇴사": [
        "권한조정과회수관리",
        "보안서약과퇴직관리",
    ],
}

# === 3기 도메인 분류 (해외사업장 보안관리 6개 영역) ===
# Ⅴ장 "해외사업장 보안관리 방안"이 3기의 핵심 콘텐츠
OVERSEAS_DOMAINS_GEN3 = [
    "기술정보관리",
    "기술보호지원",
    "외부위험대응",
    "시설보안관리",
    "인력보안관리",
    "업무환경변화대응",
]

# 3기 위험사례 ID 체계 (도메인-번호) - 총 20개
GEN3_RISK_IDS = {
    "기술정보관리": ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6"],
    "기술보호지원": ["2-1", "2-2", "2-3", "2-4"],
    "외부위험대응": ["3-1", "3-2", "3-3", "3-4"],
    "시설보안관리": ["4-1", "4-2"],
    "인력보안관리": ["5-1", "5-2", "5-3"],
    "업무환경변화대응": ["6-1"],
}

# 3기 위험등급
RISK_LEVELS_GEN3 = ["상", "중", "하"]

# 3기 대응방안 유형
MEASURE_TYPES_GEN3 = ["필수", "선택"]

# 3기 챕터 카테고리
GEN3_CHAPTER_CATEGORIES = {
    "Ⅰ": "유출보호동향",
    "Ⅱ": "보호제도소개",      # 국가핵심기술/국가첨단전략기술 정의
    "Ⅲ": "보호제도절차",      # 수출승인/신고/사전검토, 해외인수합병 절차
    "Ⅳ": "정부지원제도",
    "Ⅴ": "해외사업장보안",    # ★ 핵심 (위험사례×대응방안)
    "Ⅵ": "부록",              # 산업기술보호지침, 서류양식
}

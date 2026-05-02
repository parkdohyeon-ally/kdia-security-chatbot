"""
PDF 로딩 및 메타데이터 태깅 모듈 (디스플레이산업 보안가이드 1기/2기/3기 + 법령 통합)

기수마다 구조가 다르므로 version-aware 파싱을 합니다:

[1기] 구조: PART X → PROCESS → RISK/COUNTERMEASURE
[2기] 구조: 단계(채용/재직/퇴사) → 보안관리항목 → 본문 + 사례
[3기] 구조: 챕터(Ⅰ~Ⅵ) → 해외사업장 6개 영역 → 위험사례(상/중/하) → 대응방안(필수/선택)
[법령] 구조: 법령명 → 조 → 항 → 호 (txt 파일로 관리)
"""
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP, GUIDE_METADATA, VERSIONS


# ===========================================================================
# 1기 가이드 파싱 패턴
# ===========================================================================

GEN1_PART_PATTERNS = [
    (re.compile(r"PART\s*1", re.IGNORECASE), "PART 1"),
    (re.compile(r"PART\s*2", re.IGNORECASE), "PART 2"),
    (re.compile(r"PART\s*3", re.IGNORECASE), "PART 3"),
    (re.compile(r"APPENDIX|별\s*첨", re.IGNORECASE), "APPENDIX"),
    (re.compile(r"PROLOGUE|인사말"), "PROLOGUE"),
]

GEN1_BUSINESS_BY_PART = {
    "PART 2": "장비공급기업",
    "PART 3": "부품·소재공급기업",
    "PART 1": "공통",
    "PROLOGUE": "공통",
    "APPENDIX": "공통",
}

GEN1_PROCESS_PATTERNS = [
    (re.compile(r"수주활동\s*\(\s*PROCESS\s*\)"), "수주활동"),
    (re.compile(r"설계\s*및\s*제작\s*\(\s*PROCESS\s*\)"), "설계및제작"),
    (re.compile(r"설치\s*및\s*시운전\s*\(\s*PROCESS\s*\)"), "설치및시운전"),
    (re.compile(r"준공과\s*사후관리\s*\(\s*PROCESS\s*\)"), "준공과사후관리"),
    (re.compile(r"개발\s*및\s*평가\s*\(\s*PROCESS\s*\)"), "개발및평가"),
    (re.compile(r"제조와\s*생산\s*\(\s*PROCESS\s*\)"), "제조와생산"),
    (re.compile(r"사후관리\s*\(\s*PROCESS\s*\)"), "사후관리"),
]

GEN1_RISK_PATTERN = re.compile(r"보안\s*위험\s*\(\s*RISK\s*\)")
GEN1_COUNTER_PATTERN = re.compile(r"보안\s*대책\s*\(\s*COUNTERMEASURE\s*\)")


# ===========================================================================
# 2기 가이드 파싱 패턴
# ===========================================================================

GEN2_STAGE_PATTERNS = [
    (re.compile(r"채용\s*[①②③④⑤⑥⑦⑧⑨⑩⑪]"), "채용"),
    (re.compile(r"재직\s*[①②③④⑤⑥⑦⑧⑨⑩⑪]"), "재직"),
    (re.compile(r"퇴사\s*[①②③④⑤⑥⑦⑧⑨⑩⑪]"), "퇴사"),
]

GEN2_ITEM_PATTERNS = [
    (re.compile(r"①\s*채용\s*검증"), "채용검증"),
    (re.compile(r"②\s*보안\s*서약"), "보안서약"),
    (re.compile(r"①\s*핵심인력\s*대상선정"), "핵심인력대상선정"),
    (re.compile(r"②\s*산업보안\s*담당자지정"), "산업보안담당자지정"),
    (re.compile(r"③\s*핵심인력\s*산업보안교육"), "핵심인력산업보안교육"),
    (re.compile(r"④\s*산업보안\s*문화조성"), "산업보안문화조성"),
    (re.compile(r"⑤\s*보안활동\s*면담관리"), "보안활동면담관리"),
    (re.compile(r"⑥\s*외부접촉\s*보안관리"), "외부접촉보안관리"),
    (re.compile(r"⑦\s*자문중개업체\s*보안대응"), "자문중개업체보안대응"),
    (re.compile(r"⑧\s*재택근무\s*보안관리"), "재택근무보안관리"),
    (re.compile(r"⑨\s*해외사업장파견\s*보안관리"), "해외사업장파견보안관리"),
    (re.compile(r"⑩\s*기술유출\s*징후탐지와\s*대응"), "기술유출징후탐지와대응"),
    (re.compile(r"⑪\s*고용계약\s*갱신"), "고용계약갱신"),
    (re.compile(r"①\s*권한조정과\s*회수관리"), "권한조정과회수관리"),
    (re.compile(r"②\s*보안서약과\s*퇴직관리"), "보안서약과퇴직관리"),
]

GEN2_INCIDENT_PATTERN = re.compile(r"보안사고\s*사례")
GEN2_BEST_PRACTICE_PATTERN = re.compile(r"보안우수\s*사례")
APPENDIX_PATTERN = re.compile(r"\[별첨\s*\d+\]|별\s*첨")


# ===========================================================================
# 3기 가이드 파싱 패턴
# ===========================================================================

GEN3_CHAPTER_PATTERNS = [
    (re.compile(r"Ⅰ\s*[\.\s]*산업기술\s*유출\s*및\s*보호\s*동향"), "Ⅰ_유출보호동향"),
    (re.compile(r"Ⅱ\s*[\.\s]*산업기술\s*보호제도\s*소개"), "Ⅱ_보호제도소개"),
    (re.compile(r"Ⅲ\s*[\.\s]*산업기술\s*보호제도\s*절차"), "Ⅲ_보호제도절차"),
    (re.compile(r"Ⅳ\s*[\.\s]*산업기술보호\s*정부지원"), "Ⅳ_정부지원제도"),
    (re.compile(r"Ⅴ\s*[\.\s]*해외사업장\s*보안관리"), "Ⅴ_해외사업장보안"),
    (re.compile(r"부\s*록|산업기술보호지침|별\s*[첨지표]"), "Ⅵ_부록"),
]

GEN3_DOMAIN_PATTERNS = [
    (re.compile(r"해외사업장\s*기술정보\s*관리"), "기술정보관리"),
    (re.compile(r"해외사업장\s*기술보호\s*지원"), "기술보호지원"),
    (re.compile(r"해외사업장\s*외부위험\s*대응"), "외부위험대응"),
    (re.compile(r"해외사업장\s*시설보안\s*관리"), "시설보안관리"),
    (re.compile(r"해외사업장\s*인력보안\s*관리"), "인력보안관리"),
    (re.compile(r"해외사업장\s*업무환경\s*변화\s*대응"), "업무환경변화대응"),
]

GEN3_RISK_ID_PATTERN = re.compile(r"([1-6])\s*[-–]\s*([1-9])\s")

GEN3_RISK_LEVEL_PATTERNS = [
    (re.compile(r"\[\s*보안위험사례\s*\].*?\b상\b"), "상"),
    (re.compile(r"\[\s*보안위험사례\s*\].*?\b중\b"), "중"),
    (re.compile(r"\[\s*보안위험사례\s*\].*?\b하\b"), "하"),
]

GEN3_RISK_CASE_PATTERN = re.compile(r"\[\s*보안위험사례\s*\]")
GEN3_COUNTERMEASURE_PATTERN = re.compile(r"\[\s*보안관리방안\s*\]")
GEN3_REQUIRED_PATTERN = re.compile(r"^\s*필수\s*$|\n\s*필수\s*\n")
GEN3_OPTIONAL_PATTERN = re.compile(r"^\s*선택\s*$|\n\s*선택\s*\n")

GEN3_PROCEDURE_PATTERNS = [
    (re.compile(r"수출\s*승인"), "수출승인"),
    (re.compile(r"수출\s*신고"), "수출신고"),
    (re.compile(r"해외인수\s*[·∙]?\s*합병|해외\s*인수\s*[·∙]?\s*합병"), "해외인수합병"),
    (re.compile(r"사전\s*검토"), "사전검토"),
    (re.compile(r"보호대상\s*기술\s*여부\s*판정|국가핵심기술\s*판정"), "기술판정"),
    (re.compile(r"산업기술\s*침해신고"), "침해신고"),
]

# ===========================================================================
# 법령 파싱 패턴
# ===========================================================================

LAW_NAME_PATTERNS = [
    (re.compile(r"산업기술의\s*유출방지\s*및\s*보호에\s*관한\s*법률|산업기술보호법"), "산업기술보호법"),
    (re.compile(r"국가첨단전략산업\s*경쟁력\s*강화\s*및\s*보호에\s*관한\s*특별조치법|국가첨단전략산업법"), "국가첨단전략산업법"),
    (re.compile(r"부정경쟁방지\s*및\s*영업비밀보호에\s*관한\s*법률|부정경쟁방지법"), "부정경쟁방지법"),
    (re.compile(r"근로기준법"), "근로기준법"),
    (re.compile(r"대외무역법"), "대외무역법"),
    (re.compile(r"방위사업법"), "방위사업법"),
]

LAW_ARTICLE_PATTERN = re.compile(r"제(\d+조(?:의\d+)?)\s*\(([^)]+)\)")


# ===========================================================================
# 1기 메타데이터 추출
# ===========================================================================

def detect_gen1_part(text: str) -> Optional[str]:
    for pattern, label in GEN1_PART_PATTERNS:
        if pattern.search(text):
            return label
    return None


def detect_gen1_process(text: str) -> Optional[str]:
    for pattern, label in GEN1_PROCESS_PATTERNS:
        if pattern.search(text):
            return label
    return None


def detect_gen1_content_type(text: str) -> str:
    has_risk = bool(GEN1_RISK_PATTERN.search(text))
    has_counter = bool(GEN1_COUNTER_PATTERN.search(text))
    has_appendix = bool(APPENDIX_PATTERN.search(text))
    if has_appendix:
        return "별첨"
    if has_risk and has_counter:
        return "위험과대책"
    if has_risk:
        return "보안위험(RISK)"
    if has_counter:
        return "보안대책(COUNTERMEASURE)"
    return "일반설명"


def build_gen1_page_metadata(
    page_text: str,
    current_part: Optional[str],
    current_process: Optional[str],
) -> Dict[str, str]:
    new_part = detect_gen1_part(page_text) or current_part
    new_process = detect_gen1_process(page_text) or current_process
    business_type = GEN1_BUSINESS_BY_PART.get(new_part or "", "공통")
    content_type = detect_gen1_content_type(page_text)
    return {
        "structure_type": "gen1",
        "part": new_part or "미분류",
        "business_type": business_type,
        "process": new_process or "N/A",
        "content_type": content_type,
        "lifecycle_stage": "N/A",
        "lifecycle_item": "N/A",
        "gen3_chapter": "N/A",
        "overseas_domain": "N/A",
        "risk_id": "N/A",
        "risk_level": "N/A",
        "measure_type": "N/A",
        "procedure_type": "N/A",
        "law_name": "N/A",
        "law_article": "N/A",
    }


# ===========================================================================
# 2기 메타데이터 추출
# ===========================================================================

def detect_gen2_stage_and_item(text: str) -> Tuple[Optional[str], Optional[str]]:
    stage = None
    for pattern, label in GEN2_STAGE_PATTERNS:
        if pattern.search(text):
            stage = label
            break
    item = None
    for pattern, label in GEN2_ITEM_PATTERNS:
        if pattern.search(text):
            item = label
            break
    return stage, item


def detect_gen2_content_type(text: str) -> str:
    has_incident = bool(GEN2_INCIDENT_PATTERN.search(text))
    has_best = bool(GEN2_BEST_PRACTICE_PATTERN.search(text))
    has_appendix = bool(APPENDIX_PATTERN.search(text))
    if has_appendix:
        return "별첨"
    if has_incident and has_best:
        return "사고및우수사례"
    if has_incident:
        return "보안사고사례"
    if has_best:
        return "보안우수사례"
    return "본문"


def build_gen2_page_metadata(
    page_text: str,
    current_stage: Optional[str],
    current_item: Optional[str],
) -> Dict[str, str]:
    new_stage_detected, new_item_detected = detect_gen2_stage_and_item(page_text)
    new_stage = new_stage_detected or current_stage
    new_item = new_item_detected or current_item
    content_type = detect_gen2_content_type(page_text)
    return {
        "structure_type": "gen2",
        "part": "N/A",
        "business_type": "전산업공통",
        "process": "N/A",
        "content_type": content_type,
        "lifecycle_stage": new_stage or "미분류",
        "lifecycle_item": new_item or "미분류",
        "gen3_chapter": "N/A",
        "overseas_domain": "N/A",
        "risk_id": "N/A",
        "risk_level": "N/A",
        "measure_type": "N/A",
        "procedure_type": "N/A",
        "law_name": "N/A",
        "law_article": "N/A",
    }


# ===========================================================================
# 3기 메타데이터 추출
# ===========================================================================

def detect_gen3_chapter(text: str) -> Optional[str]:
    for pattern, label in GEN3_CHAPTER_PATTERNS:
        if pattern.search(text):
            return label
    return None


def detect_gen3_domain(text: str) -> Optional[str]:
    for pattern, label in GEN3_DOMAIN_PATTERNS:
        if pattern.search(text):
            return label
    return None


def detect_gen3_risk_id_and_level(text: str) -> Tuple[Optional[str], Optional[str]]:
    risk_level = None
    for pattern, label in GEN3_RISK_LEVEL_PATTERNS:
        if pattern.search(text):
            risk_level = label
            break
    if not risk_level:
        return None, None
    risk_id = None
    match = GEN3_RISK_ID_PATTERN.search(text)
    if match:
        risk_id = f"{match.group(1)}-{match.group(2)}"
    return risk_id, risk_level


def detect_gen3_measure_type(text: str) -> str:
    has_required = bool(GEN3_REQUIRED_PATTERN.search(text))
    has_optional = bool(GEN3_OPTIONAL_PATTERN.search(text))
    if has_required and has_optional:
        return "필수와선택"
    if has_required:
        return "필수"
    if has_optional:
        return "선택"
    return "N/A"


def detect_gen3_procedure_type(text: str) -> Optional[str]:
    for pattern, label in GEN3_PROCEDURE_PATTERNS:
        if pattern.search(text):
            return label
    return None


def detect_gen3_content_type(text: str, chapter: Optional[str]) -> str:
    has_risk_case = bool(GEN3_RISK_CASE_PATTERN.search(text))
    has_countermeasure = bool(GEN3_COUNTERMEASURE_PATTERN.search(text))
    has_appendix = bool(APPENDIX_PATTERN.search(text))
    if chapter == "Ⅵ_부록" or has_appendix:
        return "별첨"
    if has_risk_case and has_countermeasure:
        return "위험사례와대응방안"
    if has_risk_case:
        return "보안위험사례"
    if has_countermeasure:
        return "보안관리방안"
    if chapter == "Ⅲ_보호제도절차":
        return "보호제도절차"
    if chapter == "Ⅱ_보호제도소개":
        return "제도소개"
    if chapter == "Ⅳ_정부지원제도":
        return "정부지원제도"
    if chapter == "Ⅰ_유출보호동향":
        return "동향분석"
    return "일반설명"


def build_gen3_page_metadata(
    page_text: str,
    current_chapter: Optional[str],
    current_domain: Optional[str],
    current_risk_id: Optional[str],
    current_risk_level: Optional[str],
) -> Dict[str, str]:
    new_chapter = detect_gen3_chapter(page_text) or current_chapter
    if new_chapter == "Ⅴ_해외사업장보안":
        new_domain = detect_gen3_domain(page_text) or current_domain
    else:
        new_domain = "N/A"
    if new_chapter == "Ⅴ_해외사업장보안":
        detected_risk_id, detected_risk_level = detect_gen3_risk_id_and_level(page_text)
        new_risk_id = detected_risk_id or current_risk_id
        new_risk_level = detected_risk_level or current_risk_level
    else:
        new_risk_id = "N/A"
        new_risk_level = "N/A"
    content_type = detect_gen3_content_type(page_text, new_chapter)
    measure_type = detect_gen3_measure_type(page_text)
    procedure_type = detect_gen3_procedure_type(page_text) if new_chapter == "Ⅲ_보호제도절차" else None
    return {
        "structure_type": "gen3",
        "part": "N/A",
        "business_type": "전산업공통",
        "process": "N/A",
        "content_type": content_type,
        "lifecycle_stage": "N/A",
        "lifecycle_item": "N/A",
        "gen3_chapter": new_chapter or "미분류",
        "overseas_domain": new_domain or "N/A",
        "risk_id": new_risk_id or "N/A",
        "risk_level": new_risk_level or "N/A",
        "measure_type": measure_type,
        "procedure_type": procedure_type or "N/A",
        "law_name": "N/A",
        "law_article": "N/A",
    }


# ===========================================================================
# 법령 메타데이터 추출
# ===========================================================================

def detect_law_name(text: str) -> Optional[str]:
    # [법령명: XXX] 태그 우선 감지
    tag_match = re.search(r"\[법령명:\s*([^\]]+)\]", text)
    if tag_match:
        return tag_match.group(1).strip()
    # 태그 없으면 키워드로 감지
    for pattern, label in LAW_NAME_PATTERNS:
        if pattern.search(text):
            return label
    return None


def detect_law_article(text: str) -> Optional[str]:
    # [법령명: XXX] 태그 바로 뒤의 조항 감지
    tag_match = re.search(r"\[법령명:[^\]]+\]\s*(제\d+조(?:의\d+)?)\s*\(([^)]+)\)", text)
    if tag_match:
        return f"{tag_match.group(1)}({tag_match.group(2)})"
    # 일반 조항 패턴
    match = LAW_ARTICLE_PATTERN.search(text)
    if match:
        return f"제{match.group(1)}({match.group(2)})"
    return None


def build_law_chunk_metadata(
    chunk_text: str,
    current_law_name: Optional[str],
    current_article: Optional[str],
) -> Dict[str, str]:
    new_law_name = detect_law_name(chunk_text) or current_law_name
    new_article = detect_law_article(chunk_text) or current_article
    return {
        "structure_type": "law",
        "part": "N/A",
        "business_type": "전산업공통",
        "process": "N/A",
        "content_type": "법령조항",
        "lifecycle_stage": "N/A",
        "lifecycle_item": "N/A",
        "gen3_chapter": "N/A",
        "overseas_domain": "N/A",
        "risk_id": "N/A",
        "risk_level": "N/A",
        "measure_type": "N/A",
        "procedure_type": "N/A",
        "law_name": new_law_name or "미분류",
        "law_article": new_article or "N/A",
    }


# ===========================================================================
# 법령 txt 파일 로딩
# ===========================================================================

def load_law_txt_files() -> List[Document]:
    """법령 txt 파일을 읽어서 Document 리스트로 반환"""
    law_dir = PDF_DIR / "법령"
    if not law_dir.exists():
        print(f"⚠️  {law_dir} 폴더가 없습니다.")
        return []

    txt_files = list(law_dir.glob("*.txt"))
    if not txt_files:
        print(f"⚠️  {law_dir}에 txt 파일이 없습니다.")
        return []

    meta = GUIDE_METADATA["법령"]
    documents = []
    current_law_name = None
    current_article = None

    for txt_path in txt_files:
        print(f"  📄 법령 파일 로딩 중: {txt_path.name}")
        text = txt_path.read_text(encoding="utf-8")

        # 구분선(===, ---) 기준으로 섹션 분리
        sections = re.split(r"(?=\[법령명:)", text)

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            # 빈 구분자나 헤더만 있는 섹션 건너뜀
            if len(section) < 30:
                continue

            # 법령명 및 조항 감지
            detected_law = detect_law_name(section)
            if detected_law:
                current_law_name = detected_law
            detected_article = detect_law_article(section)
            if detected_article:
                current_article = detected_article

            meta_extracted = build_law_chunk_metadata(
                section, current_law_name, current_article
            )

            doc = Document(
                page_content=section,
                metadata={
                    "version": "법령",
                    "source_file": txt_path.name,
                    "publisher": meta["publisher"],
                    "issued_date": meta["issued_date"],
                    "is_current": meta["is_current"],
                    "page": i + 1,
                    **meta_extracted,
                },
            )
            documents.append(doc)

    return documents


def split_law_documents(documents: List[Document]) -> List[Document]:
    """법령 청킹 — 조항 단위로 분리"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100,
        separators=[
            "\n\n", "\n①", "\n②", "\n③", "\n④", "\n⑤",
            "\n⑥", "\n⑦", "\n⑧", "\n⑨", "\n⑩",
            "\n1.", "\n2.", "\n3.", "\n4.", "\n5.",
            "\n", ".", " ", ""
        ],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # 청크별 법령명/조항 독립적으로 재감지
    for chunk in chunks:
        text = chunk.page_content
        detected_law = detect_law_name(text)
        detected_article = detect_law_article(text)
        if detected_law:
            chunk.metadata["law_name"] = detected_law
        if detected_article:
            chunk.metadata["law_article"] = detected_article

    return chunks


# ===========================================================================
# 통합 PDF 로딩
# ===========================================================================

def load_pdfs_for_version(version: str) -> List[Document]:
    """기수별로 다른 파싱 로직 적용 (법령은 별도 함수 사용)"""
    if version == "법령":
        return load_law_txt_files()

    version_dir = PDF_DIR / version
    if not version_dir.exists():
        print(f"⚠️  {version_dir} 폴더가 없습니다.")
        return []

    pdf_files = list(version_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  {version_dir}에 PDF 파일이 없습니다.")
        return []

    meta = GUIDE_METADATA[version]
    documents = []

    for pdf_path in pdf_files:
        print(f"  📄 로딩 중: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        gen1_part: Optional[str] = None
        gen1_process: Optional[str] = None
        gen2_stage: Optional[str] = None
        gen2_item: Optional[str] = None
        gen3_chapter: Optional[str] = None
        gen3_domain: Optional[str] = None
        gen3_risk_id: Optional[str] = None
        gen3_risk_level: Optional[str] = None

        for page in pages:
            page_text = page.page_content

            if version == "1기":
                meta_extracted = build_gen1_page_metadata(
                    page_text, gen1_part, gen1_process
                )
                gen1_part = meta_extracted["part"]
                gen1_process = meta_extracted["process"]
            elif version == "2기":
                meta_extracted = build_gen2_page_metadata(
                    page_text, gen2_stage, gen2_item
                )
                gen2_stage = meta_extracted["lifecycle_stage"]
                gen2_item = meta_extracted["lifecycle_item"]
            elif version == "3기":
                meta_extracted = build_gen3_page_metadata(
                    page_text, gen3_chapter, gen3_domain, gen3_risk_id, gen3_risk_level,
                )
                gen3_chapter = meta_extracted["gen3_chapter"]
                gen3_domain = meta_extracted["overseas_domain"]
                gen3_risk_id = meta_extracted["risk_id"]
                gen3_risk_level = meta_extracted["risk_level"]
            else:
                meta_extracted = {
                    "structure_type": "unknown",
                    "part": "N/A", "business_type": "N/A", "process": "N/A",
                    "content_type": "N/A", "lifecycle_stage": "N/A", "lifecycle_item": "N/A",
                    "gen3_chapter": "N/A", "overseas_domain": "N/A", "risk_id": "N/A",
                    "risk_level": "N/A", "measure_type": "N/A", "procedure_type": "N/A",
                    "law_name": "N/A", "law_article": "N/A",
                }

            page.metadata.update({
                "version": version,
                "source_file": pdf_path.name,
                "publisher": meta["publisher"],
                "issued_date": meta["issued_date"],
                "is_current": meta["is_current"],
                "page": page.metadata.get("page", 0) + 1,
                **meta_extracted,
            })
            documents.append(page)

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        version = chunk.metadata.get("version", "")
        if version == "1기":
            recomputed = detect_gen1_content_type(chunk.page_content)
            if recomputed != "일반설명":
                chunk.metadata["content_type"] = recomputed
        elif version == "2기":
            recomputed = detect_gen2_content_type(chunk.page_content)
            if recomputed != "본문":
                chunk.metadata["content_type"] = recomputed
        elif version == "3기":
            chapter = chunk.metadata.get("gen3_chapter", "N/A")
            recomputed = detect_gen3_content_type(chunk.page_content, chapter)
            if recomputed != "일반설명":
                chunk.metadata["content_type"] = recomputed
            recomputed_measure = detect_gen3_measure_type(chunk.page_content)
            if recomputed_measure != "N/A":
                chunk.metadata["measure_type"] = recomputed_measure

    return chunks


def load_all_guides() -> List[Document]:
    all_chunks = []

    for version in VERSIONS:
        print(f"\n📚 [{version}] 로딩 시작")
        docs = load_pdfs_for_version(version)
        if not docs:
            continue

        if version == "법령":
            chunks = split_law_documents(docs)
        else:
            chunks = split_documents(docs)

        print(f"  ✅ {len(docs)}개 섹션 → {len(chunks)}청크 생성")

        # 통계 출력
        if version == "법령":
            law_stats = {}
            for c in chunks:
                law = c.metadata.get("law_name", "N/A")
                law_stats[law] = law_stats.get(law, 0) + 1
            print(f"  📊 법령별 청크: {law_stats}")
        elif version == "1기":
            stats = {}
            for c in chunks:
                p = c.metadata.get("process", "N/A")
                stats[p] = stats.get(p, 0) + 1
            print(f"  📊 PROCESS별 청크: {stats}")
        elif version == "2기":
            stats = {}
            for c in chunks:
                item = c.metadata.get("lifecycle_item", "N/A")
                stats[item] = stats.get(item, 0) + 1
            print(f"  📊 보안관리항목별 청크: {stats}")
        elif version == "3기":
            chapter_stats = {}
            for c in chunks:
                ch = c.metadata.get("gen3_chapter", "N/A")
                chapter_stats[ch] = chapter_stats.get(ch, 0) + 1
            print(f"  📊 챕터별 청크: {chapter_stats}")

        all_chunks.extend(chunks)

    print(f"\n🎯 전체 청크 수: {len(all_chunks)}")
    return all_chunks

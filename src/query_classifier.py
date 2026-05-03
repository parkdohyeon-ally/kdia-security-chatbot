"""
질문 유형 분류 모듈 (1기/2기/3기 통합)
"""
import json
import re
from typing import List, Literal
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config import LLM_MODEL, GROQ_API_KEY
from src.prompts import CLASSIFIER_PROMPT


QueryType = Literal["A", "B", "C", "D", "E"]


class QueryClassification(BaseModel):
    """질문 분류 결과 (1기/2기/3기 메타데이터 모두 포함)"""
    type: QueryType = Field(description="질문 유형")
    specified_versions: List[str] = Field(default_factory=list)

    # 1기 영역
    business_type: str = Field(default="미지정")
    process_gen1: str = Field(default="미지정")

    # 2기 영역
    lifecycle_stage: str = Field(default="미지정")
    lifecycle_item: str = Field(default="미지정")

    # 3기 영역
    gen3_chapter: str = Field(default="미지정")
    overseas_domain: str = Field(default="미지정")
    risk_level: str = Field(default="미지정")
    measure_type: str = Field(default="미지정")
    procedure_type: str = Field(default="미지정")


# === 정규식 보조 ===

VERSION_PATTERN = re.compile(r"([1-3])\s*기")

# 1기 키워드
GEN1_BUSINESS_KEYWORDS = {
    "장비공급기업": ["장비", "임프린트", "증착", "LDP", "장비배치", "FAT", "시운전", "납품검수"],
    "부품·소재공급기업": ["부품", "소재", "유리기판", "편광판", "패널", "시료", "MSDS", "Recipe", "배합", "MTA"],
}

GEN1_PROCESS_KEYWORDS = {
    "수주활동": ["수주", "고객 요구사항", "제안서", "견적", "계약체결", "발주"],
    "설계및제작": ["설계", "제작", "생산계획", "도급", "협력업체"],
    "설치및시운전": ["설치", "시운전", "현장설치"],
    "준공과사후관리": ["준공", "FAT", "납품검수"],
    "개발및평가": ["시료", "샘플", "비밀유지계약"],
    "제조와생산": ["생산현장", "제조", "공장"],
    "사후관리": ["사후관리", "추가 정보 요청"],
}

# 2기 키워드
GEN2_STAGE_KEYWORDS = {
    "채용": ["채용", "신입", "경력직", "입사", "면접", "고용계약서"],
    "재직": ["재직", "근무", "면담", "징후탐지"],
    "퇴사": ["퇴사", "퇴직", "이직"],
}

GEN2_ITEM_KEYWORDS = {
    "채용검증": ["채용 검증", "평판도 조회", "이전 직장"],
    "보안서약": ["보안서약", "비밀유지의무"],
    "핵심인력대상선정": ["핵심인력 선정", "대상선정", "충성도"],
    "산업보안담당자지정": ["산업보안담당자", "산업보안담당관", "보안위원회"],
    "핵심인력산업보안교육": ["산업보안교육", "법정교육", "보안 교육"],
    "산업보안문화조성": ["문화조성", "캠페인", "보안 표어", "포스터", "보안 홍보"],
    "보안활동면담관리": ["면담", "보안의식 수준"],
    "외부접촉보안관리": ["외부접촉", "사전신고", "사후보고", "협력사 만남"],
    "자문중개업체보안대응": ["자문중개업체", "Expert Network", "유료자문"],
    "재택근무보안관리": ["재택근무", "사외접속", "원격근무"],
    "해외사업장파견보안관리": ["해외파견", "주재원 파견"],
    "기술유출징후탐지와대응": ["기술유출 징후", "이상행위", "탐지"],
    "고용계약갱신": ["고용계약 갱신", "보안수당", "포상"],
    "권한조정과회수관리": ["권한조정", "자산회수", "출입증 회수"],
    "보안서약과퇴직관리": ["퇴직 보안서약", "퇴직관리", "전직금지약정", "경업금지"],
}

# 3기 키워드
GEN3_CHAPTER_KEYWORDS = {
    "Ⅰ_유출보호동향": ["기술유출 동향", "유출 통계", "유출 적발", "유출 추세"],
    "Ⅱ_보호제도소개": ["국가핵심기술 정의", "국가첨단전략기술 정의", "보호제도", "지정 절차", "지정 현황"],
    "Ⅲ_보호제도절차": ["수출승인", "수출 승인", "수출신고", "수출 신고", "사전검토", "해외인수합병", "해외 인수", "M&A 승인", "M&A 신고", "기술 판정", "침해신고"],
    "Ⅳ_정부지원제도": ["정부지원", "보안교육 지원", "산업보안 컨설팅", "기술지킴서비스", "보안닥터"],
    "Ⅴ_해외사업장보안": ["해외사업장", "해외 진출", "현지법인", "주재원", "현지 채용"],
    "Ⅵ_부록": ["산업기술보호지침", "별첨", "별지", "서식", "양식"],
}

GEN3_DOMAIN_KEYWORDS = {
    "기술정보관리": ["기술정보 관리", "기술 탈취", "역공학", "Reverse Engineering", "DRM", "기술자료 요구", "현지 정부", "전산망 분리", "데이터 전송"],
    "기술보호지원": ["보안 전담 조직", "보안담당자 겸직", "공정 레시피", "공정 Recipe", "기술이전 계약", "서브 라이선스", "보안관리체계", "기술이전 계약 부실"],
    "외부위험대응": ["현지 법", "법적 분쟁", "현장 실사", "세금 계산", "폐기물", "기술이전 계약 전"],
    "시설보안관리": ["여러 회사 입주", "건물 입주", "임대 시설", "CCTV", "출입통제", "보호구역", "통제구역", "제한구역", "사원증", "신원확인 장비"],
    "인력보안관리": ["현지 파견", "주재원 퇴직", "직원 이동", "협력업체 직원", "A/S 인원", "비밀유지서약서", "전직금지", "경업금지"],
    "업무환경변화대응": ["재택근무", "원격 접속", "VPN", "VDI", "2-Factor 인증", "계정 관리"],
}

GEN3_RISK_LEVEL_KEYWORDS = {
    "상": ["가장 위험", "최우선", "심각한 위험", "최고 위험"],
    "중": ["중간 위험"],
    "하": ["낮은 위험", "경미한 위험"],
}

GEN3_MEASURE_TYPE_KEYWORDS = {
    "필수": ["필수 조치", "반드시", "필수적", "의무 사항"],
    "선택": ["선택 조치", "권장", "추가로"],
}

GEN3_PROCEDURE_KEYWORDS = {
    "수출승인": ["수출승인", "수출 승인", "R&D 자금 지원", "정부지원 받은", "승인 신청", "승인 대상"],
    "수출신고": ["수출신고", "수출 신고", "독자 개발", "R&D 자금 지원받지 않은", "신고 대상", "신고 수리", "포괄신고", "수출신고서", "제21조", "제22조", "제23조"],
    "해외인수합병": ["해외인수합병", "해외 인수", "해외 M&A", "외국인투자", "합작투자"],
    "사전검토": ["사전검토", "사전 검토"],
    "기술판정": ["기술 판정", "국가핵심기술 판정", "기술 여부 판정"],
    "침해신고": ["산업기술 침해", "침해신고", "침해 신고"],
}


def extract_versions_by_regex(question: str) -> List[str]:
    matches = VERSION_PATTERN.findall(question)
    return sorted(set(f"{m}기" for m in matches))


def _detect_best(question: str, kw_dict: dict):
    scores = {k: sum(1 for kw in kws if kw in question) for k, kws in kw_dict.items()}
    scores = {k: v for k, v in scores.items() if v > 0}
    return max(scores, key=scores.get) if scores else None


def detect_gen1_business_type(question: str):
    return _detect_best(question, GEN1_BUSINESS_KEYWORDS)

def detect_gen1_process(question: str):
    return _detect_best(question, GEN1_PROCESS_KEYWORDS)

def detect_gen2_stage(question: str):
    return _detect_best(question, GEN2_STAGE_KEYWORDS)

def detect_gen2_item(question: str):
    return _detect_best(question, GEN2_ITEM_KEYWORDS)

def detect_gen3_chapter(question: str):
    return _detect_best(question, GEN3_CHAPTER_KEYWORDS)

def detect_gen3_domain(question: str):
    return _detect_best(question, GEN3_DOMAIN_KEYWORDS)

def detect_gen3_risk_level(question: str):
    return _detect_best(question, GEN3_RISK_LEVEL_KEYWORDS)

def detect_gen3_measure_type(question: str):
    return _detect_best(question, GEN3_MEASURE_TYPE_KEYWORDS)

def detect_gen3_procedure(question: str):
    return _detect_best(question, GEN3_PROCEDURE_KEYWORDS)


def classify_query(question: str) -> QueryClassification:
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
    )

    prompt = CLASSIFIER_PROMPT.format(question=question)
    response = llm.invoke([HumanMessage(content=prompt)])

    content = response.content.strip()
    json_match = re.search(r"\{.*\}", content, re.DOTALL)

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            classification = QueryClassification(**parsed)
        except (json.JSONDecodeError, ValueError):
            classification = QueryClassification(type="A")
    else:
        classification = QueryClassification(type="A")

    # 정규식 보정
    regex_versions = extract_versions_by_regex(question)
    if regex_versions and not classification.specified_versions:
        classification.specified_versions = regex_versions
        if len(regex_versions) >= 2 and classification.type == "A":
            classification.type = "C"

    # 1기 보정
    if classification.business_type == "미지정":
        bt = detect_gen1_business_type(question)
        if bt:
            classification.business_type = bt
    if classification.process_gen1 == "미지정":
        proc = detect_gen1_process(question)
        if proc:
            classification.process_gen1 = proc

    # 2기 보정
    if classification.lifecycle_stage == "미지정":
        stage = detect_gen2_stage(question)
        if stage:
            classification.lifecycle_stage = stage
    if classification.lifecycle_item == "미지정":
        item = detect_gen2_item(question)
        if item:
            classification.lifecycle_item = item

    # 3기 보정
    if classification.gen3_chapter == "미지정":
        ch = detect_gen3_chapter(question)
        if ch:
            classification.gen3_chapter = ch
    if classification.overseas_domain == "미지정":
        dom = detect_gen3_domain(question)
        if dom:
            classification.overseas_domain = dom
    if classification.risk_level == "미지정":
        lv = detect_gen3_risk_level(question)
        if lv:
            classification.risk_level = lv
    if classification.measure_type == "미지정":
        mt = detect_gen3_measure_type(question)
        if mt:
            classification.measure_type = mt
    if classification.procedure_type == "미지정":
        proc = detect_gen3_procedure(question)
        if proc:
            classification.procedure_type = proc

    return classification


"""
검색 모듈 (1기/2기/3기 통합)

핵심 전략:
1. 질문에서 추출한 메타데이터로 필터링
   - 1기 관련 단서가 있으면 1기 위주
   - 2기 관련 단서가 있으면 2기 위주
   - 3기 관련 단서가 있으면 3기 위주
   - 신호 없으면 양쪽 검색
2. 짝 보강 (기수별로 다름)
   - 1기: RISK 검색 시 같은 PROCESS의 COUNTERMEASURE도 함께
   - 2기: 본문 검색 시 같은 항목의 보안사고/우수사례도 함께
   - 3기: 위험사례 검색 시 같은 risk_id의 대응방안(필수/선택)도 함께
3. 비교 질문은 기수별 독립 검색
"""
from typing import List, Dict, Set, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import DEFAULT_TOP_K, COMPARISON_TOP_K_PER_VERSION, VERSIONS
from src.query_classifier import QueryClassification


def _has_gen1_signal(c: QueryClassification) -> bool:
    return (c.business_type != "미지정" and c.business_type != "공통") or c.process_gen1 != "미지정"


def _has_gen2_signal(c: QueryClassification) -> bool:
    return c.lifecycle_stage != "미지정" or c.lifecycle_item != "미지정"


def _has_gen3_signal(c: QueryClassification) -> bool:
    return (
        c.gen3_chapter != "미지정"
        or c.overseas_domain != "미지정"
        or c.procedure_type != "미지정"
        or c.measure_type != "미지정"
    )


def _build_filter(classification: QueryClassification) -> Optional[Dict]:
    """분류 결과에서 메타데이터 필터 생성"""
    conditions = []
    
    gen1_signal = _has_gen1_signal(classification)
    gen2_signal = _has_gen2_signal(classification)
    gen3_signal = _has_gen3_signal(classification)
    
    # 신호가 정확히 한 기수에만 있을 때만 해당 기수로 필터링
    # 여러 기수 신호가 동시에 있으면 필터 없이 검색 (LLM이 정리)
    
    # 1기 신호만 있을 때
    if gen1_signal and not gen2_signal and not gen3_signal:
        conditions.append({"version": "1기"})
        if classification.business_type not in ("미지정", "공통"):
            conditions.append({
                "business_type": {"$in": [classification.business_type, "공통"]}
            })
        if classification.process_gen1 != "미지정":
            conditions.append({"process": classification.process_gen1})
    
    # 2기 신호만 있을 때
    elif gen2_signal and not gen1_signal and not gen3_signal:
        conditions.append({"version": "2기"})
        if classification.lifecycle_stage != "미지정":
            conditions.append({"lifecycle_stage": classification.lifecycle_stage})
        if classification.lifecycle_item != "미지정":
            conditions.append({"lifecycle_item": classification.lifecycle_item})
    
    # 3기 신호만 있을 때
    elif gen3_signal and not gen1_signal and not gen2_signal:
        conditions.append({"version": "3기"})
        if classification.gen3_chapter != "미지정":
            conditions.append({"gen3_chapter": classification.gen3_chapter})
        if classification.overseas_domain != "미지정":
            conditions.append({"overseas_domain": classification.overseas_domain})
        if classification.procedure_type != "미지정":
            conditions.append({"procedure_type": classification.procedure_type})
        # measure_type은 "필수와선택" 케이스 때문에 $in 사용
        if classification.measure_type != "미지정":
            conditions.append({
                "measure_type": {"$in": [classification.measure_type, "필수와선택"]}
            })
    
    # 둘 이상 신호가 있거나 신호 없으면 필터 없이 검색
    
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _enrich_gen1_pairs(
    vectorstore: Chroma, results: List[Document]
) -> List[Document]:
    """1기 RISK ↔ COUNTERMEASURE 짝 보강"""
    enriched = list(results)
    seen_keys: Set[str] = set()
    
    for doc in results:
        if doc.metadata.get("version") != "1기":
            continue
        key = (
            doc.metadata.get("version"),
            doc.metadata.get("process"),
            doc.metadata.get("content_type"),
        )
        seen_keys.add(str(key))
    
    for doc in results:
        if doc.metadata.get("version") != "1기":
            continue
        process = doc.metadata.get("process", "N/A")
        ctype = doc.metadata.get("content_type", "")
        if process == "N/A":
            continue
        
        if "RISK" in ctype or "위험" in ctype:
            target = "보안대책(COUNTERMEASURE)"
        elif "COUNTERMEASURE" in ctype or "대책" in ctype:
            target = "보안위험(RISK)"
        else:
            continue
        
        pair_key = str(("1기", process, target))
        if pair_key in seen_keys:
            continue
        
        try:
            pair = vectorstore.similarity_search(
                process, k=2,
                filter={"$and": [
                    {"version": "1기"},
                    {"process": process},
                    {"content_type": target},
                ]},
            )
            for p in pair:
                k = str((
                    p.metadata.get("version"),
                    p.metadata.get("process"),
                    p.metadata.get("content_type"),
                ))
                if k not in seen_keys:
                    enriched.append(p)
                    seen_keys.add(k)
        except Exception:
            continue
    
    return enriched


def _enrich_gen2_examples(
    vectorstore: Chroma, results: List[Document]
) -> List[Document]:
    """2기 본문 ↔ 사례 짝 보강"""
    enriched = list(results)
    seen_keys: Set[str] = set()
    
    for doc in results:
        if doc.metadata.get("version") != "2기":
            continue
        key = (
            doc.metadata.get("version"),
            doc.metadata.get("lifecycle_item"),
            doc.metadata.get("content_type"),
        )
        seen_keys.add(str(key))
    
    for doc in results:
        if doc.metadata.get("version") != "2기":
            continue
        item = doc.metadata.get("lifecycle_item", "미분류")
        ctype = doc.metadata.get("content_type", "")
        if item in ("미분류", "N/A"):
            continue
        
        if ctype == "본문":
            targets = ["보안사고사례", "보안우수사례"]
        elif ctype in ("보안사고사례", "보안우수사례"):
            targets = ["본문"]
        else:
            continue
        
        for target in targets:
            pair_key = str(("2기", item, target))
            if pair_key in seen_keys:
                continue
            try:
                pair = vectorstore.similarity_search(
                    item, k=2,
                    filter={"$and": [
                        {"version": "2기"},
                        {"lifecycle_item": item},
                        {"content_type": target},
                    ]},
                )
                for p in pair:
                    k = str((
                        p.metadata.get("version"),
                        p.metadata.get("lifecycle_item"),
                        p.metadata.get("content_type"),
                    ))
                    if k not in seen_keys:
                        enriched.append(p)
                        seen_keys.add(k)
            except Exception:
                continue
    
    return enriched


def _enrich_gen3_pairs(
    vectorstore: Chroma, results: List[Document]
) -> List[Document]:
    """
    3기 위험사례 ↔ 대응방안 짝 보강
    
    Ⅴ장은 [보안위험사례] + [보안관리방안] 한 쌍이지만, 청크가 분리되었을 수 있음.
    risk_id가 같은 청크끼리 묶어서 보강한다.
    또한 Ⅲ장은 절차의 단계별 청크가 분리되어 있을 수 있어 procedure_type 단위로 보강.
    """
    enriched = list(results)
    seen_keys: Set[str] = set()
    
    # 이미 검색된 청크의 키 등록
    for doc in results:
        if doc.metadata.get("version") != "3기":
            continue
        key = (
            doc.metadata.get("version"),
            doc.metadata.get("risk_id"),
            doc.metadata.get("procedure_type"),
            doc.metadata.get("content_type"),
        )
        seen_keys.add(str(key))
    
    for doc in results:
        if doc.metadata.get("version") != "3기":
            continue
        
        risk_id = doc.metadata.get("risk_id", "N/A")
        procedure_type = doc.metadata.get("procedure_type", "N/A")
        ctype = doc.metadata.get("content_type", "")
        
        # 케이스 1: Ⅴ장 위험사례 ↔ 대응방안 짝 보강
        if risk_id != "N/A":
            if ctype in ("보안위험사례", "위험사례와대응방안"):
                target_types = ["보안관리방안", "위험사례와대응방안"]
            elif ctype == "보안관리방안":
                target_types = ["보안위험사례", "위험사례와대응방안"]
            else:
                continue
            
            for target in target_types:
                pair_key = str(("3기", risk_id, "N/A", target))
                if pair_key in seen_keys:
                    continue
                try:
                    pair = vectorstore.similarity_search(
                        f"위험사례 {risk_id}", k=2,
                        filter={"$and": [
                            {"version": "3기"},
                            {"risk_id": risk_id},
                            {"content_type": target},
                        ]},
                    )
                    for p in pair:
                        k = str((
                            p.metadata.get("version"),
                            p.metadata.get("risk_id"),
                            p.metadata.get("procedure_type"),
                            p.metadata.get("content_type"),
                        ))
                        if k not in seen_keys:
                            enriched.append(p)
                            seen_keys.add(k)
                except Exception:
                    continue
        
        # 케이스 2: Ⅲ장 절차 보강
        elif procedure_type != "N/A":
            pair_key_proc = str(("3기", "N/A", procedure_type, "보호제도절차"))
            if pair_key_proc in seen_keys:
                continue
            try:
                pair = vectorstore.similarity_search(
                    procedure_type, k=2,
                    filter={"$and": [
                        {"version": "3기"},
                        {"procedure_type": procedure_type},
                    ]},
                )
                for p in pair:
                    k = str((
                        p.metadata.get("version"),
                        p.metadata.get("risk_id"),
                        p.metadata.get("procedure_type"),
                        p.metadata.get("content_type"),
                    ))
                    if k not in seen_keys:
                        enriched.append(p)
                        seen_keys.add(k)
            except Exception:
                continue
    
    return enriched


def _enrich_all(vectorstore: Chroma, results: List[Document]) -> List[Document]:
    """1기/2기/3기 모두 짝 보강"""
    results = _enrich_gen1_pairs(vectorstore, results)
    results = _enrich_gen2_examples(vectorstore, results)
    results = _enrich_gen3_pairs(vectorstore, results)
    return results


# === 유형별 검색 ===

def retrieve_for_single_topic(
    vectorstore: Chroma, question: str, classification: QueryClassification,
) -> List[Document]:
    """유형 A, D"""
    filter_dict = _build_filter(classification)
    
    if filter_dict:
        results = vectorstore.similarity_search(question, k=DEFAULT_TOP_K, filter=filter_dict)
        if len(results) < 3:
            extra = vectorstore.similarity_search(question, k=DEFAULT_TOP_K)
            seen = {id(d) for d in results}
            for d in extra:
                if id(d) not in seen:
                    results.append(d)
    else:
        results = vectorstore.similarity_search(question, k=DEFAULT_TOP_K)
    
    return _enrich_all(vectorstore, results)


def retrieve_for_specific_version(
    vectorstore: Chroma, question: str, classification: QueryClassification,
) -> List[Document]:
    """유형 B"""
    versions = classification.specified_versions
    base_filter = _build_filter(classification)
    
    version_cond = (
        {"version": {"$in": versions}} if len(versions) > 1
        else {"version": versions[0]}
    )
    
    if base_filter:
        if "$and" in base_filter:
            other_conds = [c for c in base_filter["$and"] if "version" not in c]
            full_filter = {"$and": other_conds + [version_cond]} if other_conds else version_cond
        elif "version" in base_filter:
            full_filter = version_cond
        else:
            full_filter = {"$and": [base_filter, version_cond]}
    else:
        full_filter = version_cond
    
    results = vectorstore.similarity_search(question, k=DEFAULT_TOP_K, filter=full_filter)
    return _enrich_all(vectorstore, results)


def retrieve_for_comparison(
    vectorstore: Chroma, question: str, classification: QueryClassification,
) -> List[Document]:
    """유형 C: 기수별 독립 검색"""
    target_versions = classification.specified_versions or VERSIONS
    
    all_results = []
    for version in target_versions:
        try:
            v_results = vectorstore.similarity_search(
                question, k=COMPARISON_TOP_K_PER_VERSION,
                filter={"version": version},
            )
            all_results.extend(v_results)
        except Exception:
            continue
    
    return _enrich_all(vectorstore, all_results)


def retrieve(
    vectorstore: Chroma, question: str, classification: QueryClassification,
) -> List[Document]:
    if classification.type == "E":
        return []
    if classification.type == "B" and classification.specified_versions:
        return retrieve_for_specific_version(vectorstore, question, classification)
    if classification.type == "C":
        return retrieve_for_comparison(vectorstore, question, classification)
    return retrieve_for_single_topic(vectorstore, question, classification)


def format_context(documents: List[Document]) -> str:
    """1기/2기/3기 메타데이터를 구분해서 표시"""
    if not documents:
        return "(검색된 관련 문서 없음)"
    
    formatted = []
    for i, doc in enumerate(documents, 1):
        version = doc.metadata.get("version", "N/A")
        page = doc.metadata.get("page", "N/A")
        source_file = doc.metadata.get("source_file", "N/A")
        content_type = doc.metadata.get("content_type", "N/A")
        
        # 기수별로 다른 메타 표시
        if version == "1기":
            part = doc.metadata.get("part", "N/A")
            business_type = doc.metadata.get("business_type", "N/A")
            process = doc.metadata.get("process", "N/A")
            location_info = (
                f"📖 [1기 가이드] {part} | 대상: {business_type} | "
                f"단계: {process} | 유형: {content_type} | p.{page}"
            )
        elif version == "2기":
            stage = doc.metadata.get("lifecycle_stage", "N/A")
            item = doc.metadata.get("lifecycle_item", "N/A")
            location_info = (
                f"📖 [2기 가이드] 단계: {stage} | 항목: {item} | "
                f"유형: {content_type} | p.{page}"
            )
        elif version == "3기":
            chapter = doc.metadata.get("gen3_chapter", "N/A")
            domain = doc.metadata.get("overseas_domain", "N/A")
            risk_id = doc.metadata.get("risk_id", "N/A")
            risk_level = doc.metadata.get("risk_level", "N/A")
            measure_type = doc.metadata.get("measure_type", "N/A")
            procedure_type = doc.metadata.get("procedure_type", "N/A")
            
            if chapter == "Ⅴ_해외사업장보안" and domain != "N/A":
                level_str = f" [{risk_level}]" if risk_level != "N/A" else ""
                risk_str = f" | 위험번호: {risk_id}{level_str}" if risk_id != "N/A" else ""
                measure_str = f" | 대응: {measure_type}" if measure_type != "N/A" else ""
                location_info = (
                    f"📖 [3기 가이드] Ⅴ장 해외사업장 | 영역: {domain}"
                    f"{risk_str}{measure_str} | 유형: {content_type} | p.{page}"
                )
            elif chapter == "Ⅲ_보호제도절차" and procedure_type != "N/A":
                location_info = (
                    f"📖 [3기 가이드] Ⅲ장 보호제도절차 | 절차: {procedure_type} | "
                    f"유형: {content_type} | p.{page}"
                )
            else:
                location_info = (
                    f"📖 [3기 가이드] {chapter} | 유형: {content_type} | p.{page}"
                )
        else:
            location_info = f"📖 [{version}] p.{page} | 유형: {content_type}"
        
        chunk_text = (
            f"--- [청크 {i}] ---\n"
            f"{location_info}\n"
            f"파일: {source_file}\n"
            f"내용:\n{doc.page_content}\n"
        )
        formatted.append(chunk_text)
    
    return "\n".join(formatted)

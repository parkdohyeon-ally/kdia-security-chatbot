"""
검색 모듈 (1기/2기/3기 통합)

핵심 전략:
1. 질문에서 추출한 메타데이터로 필터링
2. 짝 보강 (기수별로 다름)
3. 비교 질문은 기수별 독립 검색
"""
from typing import List, Dict, Set, Optional
from langchain_community.vectorstores import FAISS
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


def _meta_filter(docs: List[Document], filter_dict: Dict) -> List[Document]:
    """FAISS는 메타데이터 필터를 직접 지원하지 않으므로 검색 후 Python으로 필터링"""
    if not filter_dict:
        return docs

    def match(doc: Document) -> bool:
        meta = doc.metadata
        for key, val in filter_dict.items():
            if key == "$and":
                if not all(match_single(doc, c) for c in val):
                    return False
            else:
                if not match_single(doc, {key: val}):
                    return False
        return True

    def match_single(doc: Document, cond: Dict) -> bool:
        meta = doc.metadata
        for key, val in cond.items():
            if key.startswith("$"):
                continue
            meta_val = meta.get(key)
            if isinstance(val, dict):
                if "$in" in val:
                    if meta_val not in val["$in"]:
                        return False
            else:
                if meta_val != val:
                    return False
        return True

    return [d for d in docs if match(d)]


def _build_filter(classification: QueryClassification) -> Optional[Dict]:
    conditions = []

    gen1_signal = _has_gen1_signal(classification)
    gen2_signal = _has_gen2_signal(classification)
    gen3_signal = _has_gen3_signal(classification)

    if gen1_signal and not gen2_signal and not gen3_signal:
        conditions.append({"version": "1기"})
        if classification.business_type not in ("미지정", "공통"):
            conditions.append({
                "business_type": {"$in": [classification.business_type, "공통"]}
            })
        if classification.process_gen1 != "미지정":
            conditions.append({"process": classification.process_gen1})

    elif gen2_signal and not gen1_signal and not gen3_signal:
        conditions.append({"version": "2기"})
        if classification.lifecycle_stage != "미지정":
            conditions.append({"lifecycle_stage": classification.lifecycle_stage})
        if classification.lifecycle_item != "미지정":
            conditions.append({"lifecycle_item": classification.lifecycle_item})

    elif gen3_signal and not gen1_signal and not gen2_signal:
        conditions.append({"version": "3기"})
        if classification.gen3_chapter != "미지정":
            conditions.append({"gen3_chapter": classification.gen3_chapter})
        if classification.overseas_domain != "미지정":
            conditions.append({"overseas_domain": classification.overseas_domain})
        if classification.procedure_type != "미지정":
            conditions.append({"procedure_type": classification.procedure_type})
        if classification.measure_type != "미지정":
            conditions.append({
                "measure_type": {"$in": [classification.measure_type, "필수와선택"]}
            })

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _enrich_gen1_pairs(
    vectorstore: FAISS, results: List[Document]
) -> List[Document]:
    enriched = list(results)
    seen_keys: Set[str] = set()

    for doc in results:
        if doc.metadata.get("version") != "1기":
            continue
        key = (doc.metadata.get("version"), doc.metadata.get("process"), doc.metadata.get("content_type"))
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
            candidates = vectorstore.similarity_search(process, k=10)
            pair = _meta_filter(candidates, {"$and": [
                {"version": "1기"}, {"process": process}, {"content_type": target}
            ]})[:2]
            for p in pair:
                k = str((p.metadata.get("version"), p.metadata.get("process"), p.metadata.get("content_type")))
                if k not in seen_keys:
                    enriched.append(p)
                    seen_keys.add(k)
        except Exception:
            continue

    return enriched


def _enrich_gen2_examples(
    vectorstore: FAISS, results: List[Document]
) -> List[Document]:
    enriched = list(results)
    seen_keys: Set[str] = set()

    for doc in results:
        if doc.metadata.get("version") != "2기":
            continue
        key = (doc.metadata.get("version"), doc.metadata.get("lifecycle_item"), doc.metadata.get("content_type"))
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
                candidates = vectorstore.similarity_search(item, k=10)
                pair = _meta_filter(candidates, {"$and": [
                    {"version": "2기"}, {"lifecycle_item": item}, {"content_type": target}
                ]})[:2]
                for p in pair:
                    k = str((p.metadata.get("version"), p.metadata.get("lifecycle_item"), p.metadata.get("content_type")))
                    if k not in seen_keys:
                        enriched.append(p)
                        seen_keys.add(k)
            except Exception:
                continue

    return enriched


def _enrich_gen3_pairs(
    vectorstore: FAISS, results: List[Document]
) -> List[Document]:
    enriched = list(results)
    seen_keys: Set[str] = set()

    for doc in results:
        if doc.metadata.get("version") != "3기":
            continue
        key = (doc.metadata.get("version"), doc.metadata.get("risk_id"), doc.metadata.get("procedure_type"), doc.metadata.get("content_type"))
        seen_keys.add(str(key))

    for doc in results:
        if doc.metadata.get("version") != "3기":
            continue

        risk_id = doc.metadata.get("risk_id", "N/A")
        procedure_type = doc.metadata.get("procedure_type", "N/A")
        ctype = doc.metadata.get("content_type", "")

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
                    candidates = vectorstore.similarity_search(f"위험사례 {risk_id}", k=10)
                    pair = _meta_filter(candidates, {"$and": [
                        {"version": "3기"}, {"risk_id": risk_id}, {"content_type": target}
                    ]})[:2]
                    for p in pair:
                        k = str((p.metadata.get("version"), p.metadata.get("risk_id"), p.metadata.get("procedure_type"), p.metadata.get("content_type")))
                        if k not in seen_keys:
                            enriched.append(p)
                            seen_keys.add(k)
                except Exception:
                    continue

        elif procedure_type != "N/A":
            pair_key_proc = str(("3기", "N/A", procedure_type, "보호제도절차"))
            if pair_key_proc in seen_keys:
                continue
            try:
                candidates = vectorstore.similarity_search(procedure_type, k=10)
                pair = _meta_filter(candidates, {"$and": [
                    {"version": "3기"}, {"procedure_type": procedure_type}
                ]})[:2]
                for p in pair:
                    k = str((p.metadata.get("version"), p.metadata.get("risk_id"), p.metadata.get("procedure_type"), p.metadata.get("content_type")))
                    if k not in seen_keys:
                        enriched.append(p)
                        seen_keys.add(k)
            except Exception:
                continue

    return enriched


def _enrich_appendix(
    vectorstore: FAISS, results: List[Document]
) -> List[Document]:
    enriched = list(results)
    seen_contents = {d.page_content for d in results}

    has_gen3_procedure = any(
        d.metadata.get("gen3_chapter") == "Ⅲ_보호제도절차"
        for d in results
    )

    if has_gen3_procedure:
        # procedure_type 체크 없이 바로 법령 검색
        procedure_keywords = {
            "수출승인": "산업기술보호법 제11조 국가핵심기술 수출 승인 연구개발비",
            "수출신고": "산업기술보호법 제11조 국가핵심기술 수출 신고",
            "해외인수합병": "산업기술보호법 제11조의2 해외인수합병 승인 신고",
            "사전검토": "산업기술보호법 제11조 사전검토 신청",
            "기술판정": "산업기술보호법 제9조 국가핵심기술 지정 판정",
            "침해신고": "산업기술보호법 제14조 침해행위 신고",
        }

        # Ⅲ장 청크들에서 procedure_type 수집
        procedures = set()
        for doc in results:
            if doc.metadata.get("gen3_chapter") == "Ⅲ_보호제도절차":
                p = doc.metadata.get("procedure_type", "N/A")
                if p != "N/A":
                    procedures.add(p)

        # procedure_type이 없으면 기본 수출승인으로 검색
        if not procedures:
            procedures = {"수출승인"}

        for procedure in procedures:
            try:
                search_keyword = procedure_keywords.get(
                    procedure, f"{procedure} 법률 조항 산업기술보호법"
                )
                candidates = vectorstore.similarity_search(search_keyword, k=50)

                # 법령 청크 우선
                law_docs = _meta_filter(candidates, {
                    "$and": [
                        {"version": "법령"},
                        {"content_type": "법령조항"},
                    ]
                })[:3]
                for a in law_docs:
                    if a.page_content not in seen_contents:
                        enriched.append(a)
                        seen_contents.add(a.page_content)

                # 3기 별첨도 추가
                appendix_docs = _meta_filter(candidates, {
                    "$and": [
                        {"version": "3기"},
                        {"content_type": "별첨"},
                    ]
                })[:1]
                for a in appendix_docs:
                    if a.page_content not in seen_contents:
                        enriched.append(a)
                        seen_contents.add(a.page_content)

            except Exception:
                continue

    # 2기 별첨 보강
    has_gen2 = any(d.metadata.get("version") == "2기" for d in results)
    if has_gen2:
        for doc in results:
            if doc.metadata.get("version") != "2기":
                continue
            item = doc.metadata.get("lifecycle_item", "N/A")
            if item == "N/A":
                continue
            try:
                candidates = vectorstore.similarity_search(
                    f"{item} 법률 조항 근거", k=6
                )
                appendix_docs = _meta_filter(candidates, {
                    "$and": [
                        {"version": "2기"},
                        {"content_type": "별첨"},
                    ]
                })[:1]
                for a in appendix_docs:
                    if a.page_content not in seen_contents:
                        enriched.append(a)
                        seen_contents.add(a.page_content)
            except Exception:
                continue

    return enriched


def _enrich_all(vectorstore: FAISS, results: List[Document]) -> List[Document]:
    results = _enrich_gen1_pairs(vectorstore, results)
    results = _enrich_gen2_examples(vectorstore, results)
    results = _enrich_gen3_pairs(vectorstore, results)
    results = _enrich_appendix(vectorstore, results)
    return results


def retrieve_for_single_topic(
    vectorstore: FAISS, question: str, classification: QueryClassification,
) -> List[Document]:
    filter_dict = _build_filter(classification)
    all_results = vectorstore.similarity_search(question, k=DEFAULT_TOP_K * 3)

    if filter_dict:
        results = _meta_filter(all_results, filter_dict)[:DEFAULT_TOP_K]
        if len(results) < 3:
            seen = {id(d) for d in results}
            for d in all_results:
                if id(d) not in seen:
                    results.append(d)
                if len(results) >= DEFAULT_TOP_K:
                    break
    else:
        results = all_results[:DEFAULT_TOP_K]

    return _enrich_all(vectorstore, results)


def retrieve_for_specific_version(
    vectorstore: FAISS, question: str, classification: QueryClassification,
) -> List[Document]:
    versions = classification.specified_versions
    all_results = vectorstore.similarity_search(question, k=DEFAULT_TOP_K * 3)
    results = _meta_filter(all_results, {"version": {"$in": versions}})[:DEFAULT_TOP_K]
    return _enrich_all(vectorstore, results)


def retrieve_for_comparison(
    vectorstore: FAISS, question: str, classification: QueryClassification,
) -> List[Document]:
    target_versions = classification.specified_versions or VERSIONS
    all_results = vectorstore.similarity_search(question, k=COMPARISON_TOP_K_PER_VERSION * len(target_versions) * 2)

    all_docs = []
    for version in target_versions:
        v_results = _meta_filter(all_results, {"version": version})[:COMPARISON_TOP_K_PER_VERSION]
        all_docs.extend(v_results)

    return _enrich_all(vectorstore, all_docs)


def retrieve(
    vectorstore: FAISS, question: str, classification: QueryClassification,
) -> List[Document]:
    if classification.type == "E":
        return []
    if classification.type == "B" and classification.specified_versions:
        return retrieve_for_specific_version(vectorstore, question, classification)
    if classification.type == "C":
        return retrieve_for_comparison(vectorstore, question, classification)
    return retrieve_for_single_topic(vectorstore, question, classification)


def format_context(documents: List[Document]) -> str:
    if not documents:
        return "(검색된 관련 문서 없음)"

    formatted = []
    for i, doc in enumerate(documents, 1):
        version = doc.metadata.get("version", "N/A")
        page = doc.metadata.get("page", "N/A")
        content_type = doc.metadata.get("content_type", "N/A")

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
        elif version == "법령":
            law_name = doc.metadata.get("law_name", "N/A")
            law_article = doc.metadata.get("law_article", "N/A")
            location_info = (
                f"📖 [관련 법령] {law_name} | {law_article} | p.{page}"
            )
        else:
            location_info = f"📖 [{version}] p.{page} | 유형: {content_type}"

        chunk_text = (
            f"[{i}] {location_info}\n"
            f"{doc.page_content}\n"
        )
        formatted.append(chunk_text)

    return "\n".join(formatted)

"""
챗봇 진입점 (1기/2기/3기 통합)
"""
from src.chain import SecurityGuideChain


def print_response(result: dict):
    print("\n" + "=" * 70)
    print(f"🏷️  질문 유형: {result['query_type_label']}")
    
    info_parts = []
    if result["specified_versions"]:
        info_parts.append(f"기수: {', '.join(result['specified_versions'])}")
    
    # 1기 영역 정보
    gen1_parts = []
    if result["business_type"] != "미지정":
        gen1_parts.append(result["business_type"])
    if result["process_gen1"] != "미지정":
        gen1_parts.append(result["process_gen1"])
    if gen1_parts:
        info_parts.append(f"[1기] {' / '.join(gen1_parts)}")
    
    # 2기 영역 정보
    gen2_parts = []
    if result["lifecycle_stage"] != "미지정":
        gen2_parts.append(result["lifecycle_stage"])
    if result["lifecycle_item"] != "미지정":
        gen2_parts.append(result["lifecycle_item"])
    if gen2_parts:
        info_parts.append(f"[2기] {' / '.join(gen2_parts)}")
    
    # 3기 영역 정보
    gen3_parts = []
    if result["gen3_chapter"] != "미지정":
        gen3_parts.append(result["gen3_chapter"])
    if result["overseas_domain"] != "미지정":
        gen3_parts.append(result["overseas_domain"])
    if result["procedure_type"] != "미지정":
        gen3_parts.append(f"절차:{result['procedure_type']}")
    if result["risk_level"] != "미지정":
        gen3_parts.append(f"등급:{result['risk_level']}")
    if result["measure_type"] != "미지정":
        gen3_parts.append(f"대응:{result['measure_type']}")
    if gen3_parts:
        info_parts.append(f"[3기] {' / '.join(gen3_parts)}")
    
    if info_parts:
        print(f"📌 식별된 정보: {' | '.join(info_parts)}")
    print("=" * 70)
    print(f"\n💬 답변:\n{result['answer']}\n")
    
    if result["source_documents"]:
        print("-" * 70)
        print(f"📚 참조 청크 ({len(result['source_documents'])}개):")
        for i, doc in enumerate(result["source_documents"], 1):
            v = doc.metadata.get("version", "?")
            ctype = doc.metadata.get("content_type", "?")
            p = doc.metadata.get("page", "?")
            
            if v == "1기":
                part = doc.metadata.get("part", "?")
                proc = doc.metadata.get("process", "?")
                print(f"  {i}. {v} | {part} | {proc} | {ctype} | p.{p}")
            elif v == "2기":
                stage = doc.metadata.get("lifecycle_stage", "?")
                item = doc.metadata.get("lifecycle_item", "?")
                print(f"  {i}. {v} | {stage} | {item} | {ctype} | p.{p}")
            elif v == "3기":
                chapter = doc.metadata.get("gen3_chapter", "?")
                domain = doc.metadata.get("overseas_domain", "?")
                risk_id = doc.metadata.get("risk_id", "?")
                risk_level = doc.metadata.get("risk_level", "?")
                measure_type = doc.metadata.get("measure_type", "?")
                procedure_type = doc.metadata.get("procedure_type", "?")
                
                if chapter == "Ⅴ_해외사업장보안" and domain != "N/A":
                    risk_str = f" | {risk_id}[{risk_level}]" if risk_id != "N/A" else ""
                    measure_str = f" | {measure_type}" if measure_type != "N/A" else ""
                    print(f"  {i}. {v} | Ⅴ장 {domain}{risk_str}{measure_str} | {ctype} | p.{p}")
                elif chapter == "Ⅲ_보호제도절차" and procedure_type != "N/A":
                    print(f"  {i}. {v} | Ⅲ장 {procedure_type} | {ctype} | p.{p}")
                else:
                    print(f"  {i}. {v} | {chapter} | {ctype} | p.{p}")
            else:
                print(f"  {i}. {v} | {ctype} | p.{p}")
    print("=" * 70 + "\n")


def run_chatbot():
    print("\n" + "=" * 70)
    print("🛡️  디롱이(SecureGuide) - 디스플레이산업 보안가이드 통합 챗봇")
    print("=" * 70)
    print("• 1기 (2019): 디스플레이산업 실무 보안가이드 (장비/부품·소재 기업용)")
    print("• 2기 (2022): 디스플레이산업 핵심인력 보안가이드")
    print("• 3기 (2024): 디스플레이산업 수출 보안 가이드 — 현행")
    print("종료: 'exit' 또는 'quit'\n")
    
    chain = SecurityGuideChain()
    
    while True:
        try:
            question = input("❓ 질문: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "종료"):
                print("👋 챗봇을 종료합니다.")
                break
            
            result = chain.invoke(question)
            print_response(result)
            
        except KeyboardInterrupt:
            print("\n👋 챗봇을 종료합니다.")
            break
        except Exception as e:
            print(f"\n⚠️  오류 발생: {e}\n")

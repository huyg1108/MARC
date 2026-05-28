def build_zero_shot_prompt(claim):
    return (
        "You are a multimodal misinformation analyst. "
        "Assess the provided image and textual claim and assign one of four categories:\n\n"
        "1. Textual Veracity Distortion: The claim is false, irrespective of the image.\n"
        "2. Visual Veracity Distortion: The image is manipulated or synthetic, while the claim is true.\n"
        "3. Cross-Modal Consistency Distortion: Both modalities are individually true but mutually inconsistent.\n"
        "4. Real Data: Both modalities are true and mutually consistent.\n\n"
        "Evaluate textual veracity, visual authenticity, and cross-modal consistency.\n"
        "Provide a concise, structured rationale and conclude with a single category.\n"
        f"Claim: {claim}\n"
    )


def build_physical_prompt(claim):
    return (
        "You are a forensic image analyst focused on detecting synthetic or manipulated media.\n"
        "Inspect the image for physical, biological, contextual, and digital inconsistencies.\n"
        "Adopt a conservative stance: flag any structurally implausible evidence.\n\n"
        "Evaluate four dimensions:\n"
        "1. Physical and environmental consistency:\n"
        "- Lighting and shadows alignment.\n"
        "- Reflection correctness.\n"
        "- Gravity and interaction realism.\n\n"
        "2. Biological and anatomical plausibility (humans):\n"
        "- Skin texture realism and boundary integrity.\n"
        "- Proportions and limb structure.\n"
        "- Eyes, teeth, and hands (pupils, dentition, finger count).\n\n"
        "3. Contextual and logical coherence:\n"
        "- Anachronisms relative to the setting.\n"
        "- Text, logos, and background semantics.\n\n"
        "4. Digital artifacts:\n"
        "- Over-sharpening, halos, pixelation, or synthetic gloss.\n\n"
        f"Claim context: \"{claim}\"\n"
        "Check for contradictions with real-world constraints relevant to the claim.\n\n"
        "Output format:\n"
        "Style: [Real Photo / Digital Art / Hyper-Realistic AI / Staged]\n"
        "Logic_Check: [PASS / FAIL - Impossible Object / FAIL - Context Mismatch]\n"
        "Anomaly_Flag: [HIGH / LOW / NONE]\n"
        "Key_Observations: [Bullet points focusing on concrete visual cues.]"
    )


def build_semantic_prompt(forensic_report=""):
    return (
        "You are a forensic image analyst. Determine whether the image is a real photograph or AI-generated.\n"
        "Follow a strict four-step reasoning process:\n\n"
        "Step 1: Subject identification (internal knowledge).\n"
        "- Identify the main subject (e.g., a specific celebrity, athlete, or generic person).\n"
        "- Identify the setting/context (e.g., NBA game, movie scene).\n"
        "- If the subject is a famous public figure, recall known appearance, uniforms, and tattoos.\n\n"
        "Step 2: World-knowledge discrepancy test.\n"
        "- Inspect text/logos on clothing or signage.\n"
        "- Transcribe observed text exactly.\n"
        "- Decide whether observed text matches real-world facts (Yes/No).\n"
        "- Estimate OCR confidence for the observed text: [HIGH / MEDIUM / LOW].\n\n"
        "Step 3: Forensic artifact review.\n"
        f"- Automated scanner flags: \"{forensic_report}\"\n"
        "- Check flagged regions (often text or skin boundaries) for painted textures or nonsensical geometry.\n\n"
        "Step 4: Final conclusion.\n"
        "- If the subject is famous but the text/logo is gibberish, classify as AI-generated (99% confidence).\n"
        "- If the scanner flags the text area and the text is misspelled, classify as AI-generated.\n\n"
        "Output format:\n"
        "Subject_ID: [Name/Archetype]\n"
        "Expected_Details: [What SHOULD be on the jersey/background]\n"
        "Observed_Details: [What is ACTUALLY there]\n"
        "OCR_Confidence: [HIGH / MEDIUM / LOW]\n"
        "Knowledge_Gap: [MATCH / FATAL MISMATCH]\n"
        "Forensic_Confirmation: [Did scanner flags align with the mismatch?]\n"
        "Final_Verdict: [REAL PHOTO / AI-GENERATED]"
    )


def build_conflict_resolution_prompt(claim, semantic_report, physical_report):
    return (
        "You are a senior forensic reviewer. Resolve conflicts between a semantic report and a physical-forensics report.\n"
        "Your goal is to issue a final, conservative verdict using a consistent decision policy.\n\n"
        "Inputs:\n"
        f"Claim: \"{claim}\"\n"
        f"Semantic report: {semantic_report}\n"
        f"Physical report: {physical_report}\n\n"
        "Decision policy:\n"
        "1. Treat text/logo mismatches as strong evidence only when OCR_Confidence is HIGH and the mismatch is clear and specific (e.g., wrong team/brand or obvious misspelling).\n"
        "2. If OCR_Confidence is MEDIUM or LOW and the mismatch is a missing character/spacing/single-letter ambiguity, do not conclude AI-generated without corroborating visual anomalies.\n"
        "3. If semantic report flags anomalies only in background elements (e.g., flags, banners) without text mismatch and physical report finds LOW anomalies, downgrade confidence in AI-generated.\n"
        "4. If physical report flags HIGH anomalies in anatomy, geometry, or physics, treat this as strong evidence for AI-generated or manipulation.\n"
        "5. If both reports are LOW-anomaly and no concrete mismatch is identified, favor REAL PHOTO.\n"
        "6. If evidence is mixed or weak, output UNCERTAIN and explain what additional evidence would resolve the case.\n\n"
        "Output format:\n"
        "Conflict_Type: [Text_Mismatch / Background_Only / Physical_Anomaly / None / Mixed]\n"
        "Resolution_Rationale: [Concise justification in 1-3 sentences]\n"
        "Final_Verdict: [REAL PHOTO / AI-GENERATED / UNCERTAIN]"
    )


def build_strategy_prompt(claim):
    return (
        "Analyze the image-claim pair and assign a strategy class:\n"
        "1. News Event: A verifiable event involving named entities, locations, organizations, dates, or specific actions.\n"
        "2. Generic Scene: Descriptive content without a verifiable event.\n\n"
        "Decision criteria:\n"
        "- If the claim names a person, place, organization, date, or event action, select News Event.\n"
        "- Otherwise, select Generic Scene.\n\n"
        "Action specification:\n"
        "- News Event: propose three search queries for external evidence.\n"
        "- Generic Scene: output 'ACTION: Internal Analysis'.\n\n"
        f"Claim: \"{claim}\"\n\n"
        "Output format:\n"
        "Category: [News Event OR Generic Scene]\n"
        "Action: [Queries List OR 'Internal Analysis']"
    )


def build_evidence_filtering_prompt(claim, raw_text_chunk):
    raise NotImplementedError(
        "build_evidence_filtering_prompt is missing in main_draft.py. Port the prompt text here."
    )


def build_image_evidence_filtering_prompt(claim, raw_text_chunk):
    raise NotImplementedError(
        "build_image_evidence_filtering_prompt is missing in main_draft.py. Port the prompt text here."
    )


def build_evidence_consolidation_prompt(claim, raw_evidence_text):
    raise NotImplementedError(
        "build_evidence_consolidation_prompt is missing in main_draft.py. Port the prompt text here."
    )


def build_image_context_consolidation_prompt(raw_image_evidence_text):
    raise NotImplementedError(
        "build_image_context_consolidation_prompt is missing in main_draft.py. Port the prompt text here."
    )


def build_news_resolution_prompt(claim, conflict_report, evidence_text, image_evidence_text):
    raise NotImplementedError(
        "build_news_resolution_prompt is missing in main_draft.py. Port the prompt text here."
    )


def build_generic_analysis_prompt(claim, prior_prediction, conflict_report):
    raise NotImplementedError(
        "build_generic_analysis_prompt is missing in main_draft.py. Port the prompt text here."
    )


def build_final_pred(final_reasoning):
    raise NotImplementedError(
        "build_final_pred is missing in main_draft.py. Port the prompt text here."
    )

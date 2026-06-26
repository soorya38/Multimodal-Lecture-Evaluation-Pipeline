# Multimodal-Lecture-Evaluation-Pipeline
Multimodal Lecture Evaluation Pipeline is a multimodal lecture assessment engine that evaluates educational videos using speech recognition, OCR, visual understanding, and LLM-based reasoning. It extracts spoken content, on-screen text, handwritten notes, and diagrams, consolidates them into a unified knowledge representation, and generates objective scores for technical accuracy, grammar quality, and language usage.

Designed for academic institutions, training platforms, and quality assurance workflows, the system produces explainable, rubric-driven evaluations from raw lecture recordings with minimal human intervention.

Core capabilities

Speech-to-text for multilingual lectures
Typed text and handwriting extraction
Diagram and visual content understanding
English/Tamil/Tanglish language analysis
Grammar and communication quality scoring
Technical correctness evaluation using RAG and LLMs
Weighted rubric-based scoring and report generation

Input: Lecture video
Output: Technical score, grammar score, language-mix analysis, and detailed evaluation report.

```mermaid
sequenceDiagram
    participant Client
    participant Handler as Evaluation Handler<br/>(handler.py)
    participant Usecase as Evaluation Usecase<br/>(usecase.py)
    participant Media as Media Subsystem<br/>(media/usecase.py)
    participant Storage as MinIO Storage
    participant LLM as Gemini Evaluators<br/>(evaluate.py)

    Client->>Handler: POST /api/v1/evaluate<br/>(video file, person_name, subject, timing)
    activate Handler
    
    Handler->>Usecase: run_full_pipeline(file, ...)
    activate Usecase

    Note over Usecase,Media: Step 1: Split video/audio
    Usecase->>Media: split_and_store(file)
    activate Media
    Media->>Storage: Save video & audio streams
    Media-->>Usecase: split_result (upload_id)
    deactivate Media

    Note over Usecase,Media: Step 2: Extract Frames & Transcribe (Concurrent)
    par Frame Extraction
        Usecase->>Media: extract_frames_and_store(upload_id)
        activate Media
        Media->>Storage: Save extracted frames
        Media-->>Usecase: 
        deactivate Media
    and Transcription
        Usecase->>Media: transcribe_and_store(upload_id)
        activate Media
        Media->>Storage: Save audio transcript
        Media-->>Usecase: 
        deactivate Media
    end

    Note over Usecase,Media: Step 3: OCR on frames
    Usecase->>Media: extract_text_and_store(upload_id)
    activate Media
    Media->>Storage: Save OCR results
    Media-->>Usecase: 
    deactivate Media

    Note over Usecase,Media: Step 4: Consolidate
    Usecase->>Media: consolidate_and_store(upload_id)
    activate Media
    Media->>Storage: Read transcripts & OCR data
    Media->>Storage: Save consolidated JSON
    Media-->>Usecase: consolidate_result
    deactivate Media

    Note over Usecase,Storage: Step 5: Download data for evaluation
    Usecase->>Storage: _download_consolidated(object_key)
    Storage-->>Usecase: consolidated_data (JSON)

    Note over Usecase,LLM: Step 6: LLM Evaluations (Concurrent)
    par Technical Score
        Usecase->>LLM: evaluate_technical(data, subject)
        LLM-->>Usecase: technical_score
    and Grammatical Score
        Usecase->>LLM: evaluate_grammar(data)
        LLM-->>Usecase: grammatical_score
    and Language Mix
        Usecase->>LLM: evaluate_language_mix(data)
        LLM-->>Usecase: language_mix (%)
    end

    Usecase-->>Handler: EvaluateResponse(scores, percentages)
    deactivate Usecase
    
    Handler-->>Client: HTTP 200 OK<br/>(EvaluateResponse)
    deactivate Handler
```
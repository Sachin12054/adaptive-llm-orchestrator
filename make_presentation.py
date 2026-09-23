import os
import sys

def create_presentation():
    try:
        import pptx
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
        from pptx.enum.shapes import MSO_SHAPE
    except ImportError:
        print("python-pptx not available yet")
        return

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette: Dark Navy & Cyan Accent
    BG_DARK = RGBColor(11, 15, 25)
    NAVY_CARD = RGBColor(20, 27, 45)
    CYAN_ACCENT = RGBColor(6, 182, 212)
    TEXT_LIGHT = RGBColor(248, 250, 252)
    TEXT_MUTED = RGBColor(148, 163, 184)
    GREEN_SUCCESS = RGBColor(16, 185, 129)
    WHITE = RGBColor(255, 255, 255)

    def add_blank_slide():
        blank_slide_layout = prs.slide_layouts[6]
        slide = prs.slides.add_slide(blank_slide_layout)
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = BG_DARK
        return slide

    def add_header(slide, title_text, category_text="ADAPTIVE LLM ORCHESTRATOR"):
        # Category / Subtitle Badge
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = CYAN_ACCENT

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_LIGHT

    def create_card(slide, left, top, width, height, title, items, badge_text=None, border_color=CYAN_ACCENT):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = NAVY_CARD
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.5)

        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.3)
        tf.margin_top = Inches(0.3)
        tf.margin_right = Inches(0.3)
        tf.margin_bottom = Inches(0.3)

        if badge_text:
            p_badge = tf.paragraphs[0]
            p_badge.text = f"[{badge_text}]"
            p_badge.font.size = Pt(10)
            p_badge.font.bold = True
            p_badge.font.color.rgb = GREEN_SUCCESS
            p_head = tf.add_paragraph()
        else:
            p_head = tf.paragraphs[0]

        p_head.text = title
        p_head.font.size = Pt(16)
        p_head.font.bold = True
        p_head.font.color.rgb = CYAN_ACCENT
        p_head.space_after = Pt(10)

        for item in items:
            p_item = tf.add_paragraph()
            p_item.text = f"• {item}"
            p_item.font.size = Pt(13)
            p_item.font.color.rgb = TEXT_LIGHT
            p_item.space_after = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    s1 = add_blank_slide()
    # Main Card
    shape1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.2), Inches(11.333), Inches(5.1))
    shape1.fill.solid()
    shape1.fill.fore_color.rgb = NAVY_CARD
    shape1.line.color.rgb = CYAN_ACCENT
    shape1.line.width = Pt(2)

    tf1 = shape1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(0.5)
    tf1.margin_top = Inches(0.5)

    p = tf1.paragraphs[0]
    p.text = "FINAL YEAR B.TECH PROJECT PRESENTATION"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CYAN_ACCENT
    p.space_after = Pt(15)

    p2 = tf1.add_paragraph()
    p2.text = "Adaptive LLM Orchestrator: Cost-Aware and Resource-Sensitive Dynamic Model Selection via Contextual Bandit Telemetry"
    p2.font.size = Pt(24)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_LIGHT
    p2.space_after = Pt(20)

    p3 = tf1.add_paragraph()
    p3.text = "Department of Computer Science & Engineering | Amrita Vishwa Vidyapeetham\nAcademic Year: 2025 - 2026 | Domain: Artificial Intelligence & Cloud Architecture"
    p3.font.size = Pt(13)
    p3.font.color.rgb = TEXT_MUTED
    p3.space_after = Pt(25)

    p4 = tf1.add_paragraph()
    p4.text = "TEAM MEMBERS (GROUP OF 4):\n• Member 1: AI Lead (BGE-M3 1024D Embeddings & Baseline Adaptive Policy)\n• Member 2: RL Specialist (12D State Vector Encoder & Shadow Contextual Bandit Policy)\n• Member 3: Backend Engineer (FastAPI Architecture, Telemetry & Provider Adapters)\n• Member 4: Frontend & Evaluation Lead (React Dashboard UI, SSE Reader & 28 Pytest Suite)"
    p4.font.size = Pt(12)
    p4.font.color.rgb = TEXT_LIGHT

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement
    # -------------------------------------------------------------
    s2 = add_blank_slide()
    add_header(s2, "Problem Statement & Operational Inefficiencies")
    create_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Existing Single-Model Inefficiencies", 
                ["Monolithic 70B cloud LLMs assigned to simple factual queries cause severe token cost waste ($0.00059/1k tokens).",
                 "Lightweight 3B/4B models cause execution failure on multi-step calculus or system design tasks.",
                 "High Latency Penalties: Simple queries incur unnecessary network round-trips and queueing delays.",
                 "Manual or Static Switches: Lack real-time task difficulty adaptation."])

    create_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Edge Hardware & Telemetry Limitations", 
                ["Hardware Agnosticism: Routers ignore host GPU free VRAM, causing out-of-memory (OOM) crashes.",
                 "Discarded Telemetry: Historical execution outcomes are discarded rather than logged for RL learning.",
                 "Context Window Bias: Primitive switches select 1M token context models for 10-word prompts.",
                 "Core Challenge: High API costs for simple tasks vs model failure on complex queries."])

    # -------------------------------------------------------------
    # SLIDE 3: Motivation
    # -------------------------------------------------------------
    s3 = add_blank_slide()
    add_header(s3, "Project Motivation & Real-World Relevance")
    create_card(s3, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Industry & Research Drive", 
                ["Proliferation of specialized LLMs (coding, reasoning, translation, lightweight QA).",
                 "Cost-Quality Pareto Frontier: Need sub-400ms free local execution for simple queries.",
                 "Cloud API Expenditure: Enterprises require up to 70% cost reduction on high-volume workloads.",
                 "Zero-Risk RL Exploration: Collecting experience replay records without live production routing risk."])

    create_card(s3, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Why AI & Telemetry are Required", 
                ["Rule-based regex patterns fail on dynamic semantic prompt complexity.",
                 "1024D PyTorch dense vector embeddings accurately capture intent across 9 prototypes.",
                 "Real-time host VRAM monitoring protects edge GPU nodes from memory thrashing.",
                 "Shadow Contextual Bandit learning continuously evaluates online policy convergence."])

    # -------------------------------------------------------------
    # SLIDE 4: Objectives
    # -------------------------------------------------------------
    s4 = add_blank_slide()
    add_header(s4, "Project Objectives & Implementation Scope")
    create_card(s4, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Primary Objective", 
                ["Develop an adaptive multi-LLM orchestration platform enforcing a single production decision policy.",
                 "Eliminate single-model over-provisioning and host GPU VRAM crashes.",
                 "Provide a transparent 5-factor scoring engine for model selection.",
                 "Achieve 100% routing invariant compliance across all backend endpoints."],
                badge_text="COMPLETED & VERIFIED")

    create_card(s4, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Secondary Objectives", 
                ["Generate 1024D dense vectors via PyTorch BAAI/bge-m3.",
                 "Classify intent across 9 prototypes and score 5 complexity factors.",
                 "Inspect CPU %, RAM %, and free GPU VRAM prior to model scoring.",
                 "Operate a Contextual Bandit policy in shadow mode (production_override: false).",
                 "Stream real-time stage telemetry to a React UI via Server-Sent Events (SSE)."],
                badge_text="COMPLETED & VERIFIED")

    # -------------------------------------------------------------
    # SLIDE 5: Existing System
    # -------------------------------------------------------------
    s5 = add_blank_slide()
    add_header(s5, "Existing System Workflow & Limitations")
    create_card(s5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Conventional Architecture Workflow", 
                ["User prompt sent directly to a fixed single model (e.g. GPT-4 or LLaMA 70B).",
                 "Or processed through a simple fallback switch that only triggers when primary API fails.",
                 "No prompt complexity scoring prior to model selection.",
                 "No inspection of host hardware VRAM or system load."])

    create_card(s5, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Major Limitations & Drawbacks", 
                ["Financial Waste: Simple queries incur full 70B cloud API token costs ($0.00059/1k tokens).",
                 "Task Failures: Lightweight models hallucinate on calculus proofs or code generation.",
                 "GPU VRAM Crashes: Local Ollama models overwhelm host memory without pre-checks.",
                 "No Feedback Loop: Execution rewards are discarded rather than logged for RL."])

    # -------------------------------------------------------------
    # SLIDE 6: Proposed System
    # -------------------------------------------------------------
    s6 = add_blank_slide()
    add_header(s6, "Proposed Adaptive LLM Orchestrator System")
    create_card(s6, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Core Proposed Solution", 
                ["Input Prompt ──> PyTorch BGE-M3 1024D Dense Semantic Vector.",
                 "Intent & Complexity Analysis ──> 9 Prototypes & 5 Structural Dimensions.",
                 "Hardware VRAM Check ──> Real-time CPU, RAM, and GPU free VRAM inspection.",
                 "BaselineAdaptivePolicy ──> Single Production Routing Authority (5-Factor Score).",
                 "Model Execution ──> Dispatched to Ollama or Cloud API Providers."])

    create_card(s6, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Key Innovations & Advantages", 
                ["Up to 70% Token Cost Reduction for low-complexity query benchmarks.",
                 "Zero Local VRAM Out-of-Memory Crashes via host resource pre-checks.",
                 "60% Quality Protection Weight (Capability + Complexity) prevents weak model selection.",
                 "Shadow Mode RL Evaluation (production_override: false) for safe experience logging.",
                 "Real-time SSE Event Streaming to Vite/React Telemetry Dashboard."])

    # -------------------------------------------------------------
    # SLIDE 7: System Architecture
    # -------------------------------------------------------------
    s7 = add_blank_slide()
    add_header(s7, "System Architecture & Component Integration")
    create_card(s7, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.1), 
                "Frontend & API Tier", 
                ["React 18 Single Page App",
                 "Vite 5.4 Glassmorphism UI",
                 "SSE Telemetry Event Consumer",
                 "FastAPI 0.115 ASGI Backend",
                 "Uvicorn Async Web Server",
                 "Pydantic v2 Data Schemas"])

    create_card(s7, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.1), 
                "AI Processing & Decision Engine", 
                ["BAAI/bge-m3 PyTorch 1024D Service",
                 "9-Intent Cosine Classifier",
                 "5-Factor Complexity Service",
                 "Hardware Telemetry (psutil/nvml)",
                 "BaselineAdaptivePolicy (5-Factor)",
                 "Neutral Multi-Tier Tie-Breaker"])

    create_card(s7, Inches(8.9), Inches(1.6), Inches(3.6), Inches(5.1), 
                "Providers & Shadow RL", 
                ["Ollama Local Adapter (Gemma, DeepSeek)",
                 "Online API Manager (Gemini, Groq, etc)",
                 "RewardSignal (5-Component Scalar)",
                 "ExperienceBuffer (JSONL Replay)",
                 "RLContextualBandit (12D Vector)",
                 "Shadow Mode (override=false)"])

    # -------------------------------------------------------------
    # SLIDE 8: Methodology
    # -------------------------------------------------------------
    s8 = add_blank_slide()
    add_header(s8, "Complete 14-Stage Execution Methodology")
    create_card(s8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Stages 1 to 7: Vectoring & Decision", 
                ["1. Input Validation: Text sanitization & length check.",
                 "2. Dense Embedding: PyTorch BGE-M3 generates 1024D vector.",
                 "3. Intent Matching: Cosine similarity against 9 prototypes.",
                 "4. Complexity Analysis: 5 structural complexity dimensions.",
                 "5. Hardware Telemetry: Check CPU, RAM %, and GPU free VRAM.",
                 "6. Candidate Discovery: Filter unconfigured or infeasible models.",
                 "7. Baseline Scoring: 30% Cap + 30% Cmplx + 15% Res + 15% Ctx + 10% Cost."])

    create_card(s8, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Stages 8 to 14: Execution & Shadow RL", 
                ["8. Tie-Breaking: (Score, Capability, Cost, Count, ContextLen).",
                 "9. Provider Execution: Dispatched to Ollama or Cloud API REST client.",
                 "10. Verification: Structural check & reward calculation.",
                 "11. Reward Calculation: R = 0.30Q + 0.20C + 0.25R + 0.15V + 0.10E.",
                 "12. Experience Storage: Append 12D state transition to JSONL buffer.",
                 "13. Shadow RL Prediction: Q-value prediction (override=false).",
                 "14. SSE Streaming: Stage telemetry streamed live to React UI."])

    # -------------------------------------------------------------
    # SLIDE 9: Dataset / Data
    # -------------------------------------------------------------
    s9 = add_blank_slide()
    add_header(s9, "Datasets, Metadata Registries & Replay Buffers")
    create_card(s9, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Model Registry Metadata (datasets/models/)", 
                ["Model Registry Dataset: datasets/models/model_registry.json",
                 "Contains metadata for 8 registered local and cloud models.",
                 "Attributes: Model ID, display name, provider, execution mode, context limit, capabilities.",
                 "Published Token Pricing: Input cost ($0.00 - $0.00059/1k), Output cost ($0.00 - $0.00079/1k).",
                 "Execution Requirements: Free VRAM threshold or API environment key."],
                badge_text="VERIFIED COMPLETED")

    create_card(s9, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Experience Replay Buffer (data/rl/)", 
                ["Replay Buffer Dataset: data/rl/experience_buffer.jsonl",
                 "JSONL transition persistence logging 12D state vectors.",
                 "State Vector: Intent code, ambiguity flag, overall complexity, 5 structural complexity scores, CPU %, RAM %, GPU availability, baseline score.",
                 "Log Fields: Prompt ID, state vector, selected action, scalar reward R, execution status.",
                 "Capacity: Configurable max capacity with FIFO eviction."],
                badge_text="VERIFIED COMPLETED")

    # -------------------------------------------------------------
    # SLIDE 10: Technology Stack
    # -------------------------------------------------------------
    s10 = add_blank_slide()
    add_header(s10, "Categorized Software & Hardware Tech Stack")
    create_card(s10, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.1), 
                "Languages & Backend", 
                ["Python 3.14",
                 "JavaScript (ES6+)",
                 "FastAPI 0.115",
                 "Uvicorn ASGI Server",
                 "Pydantic v2 Schemas",
                 "SSE-Starlette Streaming"])

    create_card(s10, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.1), 
                "AI / ML & Providers", 
                ["PyTorch CUDA Framework",
                 "sentence-transformers",
                 "BAAI/bge-m3 (1024D)",
                 "Local Ollama Server",
                 "Google GenAI SDK",
                 "HTTPX REST Clients"])

    create_card(s10, Inches(8.9), Inches(1.6), Inches(3.6), Inches(5.1), 
                "Frontend, Data & Testing", 
                ["React 18 SPA Framework",
                 "Vite 5.4 Build Tool",
                 "Lucide React Icons",
                 "psutil & PyTorch pynvml",
                 "JSON / JSONL Storage",
                 "Pytest 9.1 Test Suite"])

    # -------------------------------------------------------------
    # SLIDE 11: Implementation
    # -------------------------------------------------------------
    s11 = add_blank_slide()
    add_header(s11, "Implementation Breakdown & Codebase Deliverables")
    create_card(s11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Core Backend Implementation Services", 
                ["app/services/embedding_service.py: PyTorch BGE-M3 1024D vector engine.",
                 "app/services/complexity_prototype_service.py: 9-intent & 5-complexity classifier.",
                 "app/services/resource_telemetry.py: Host CPU/RAM & GPU free VRAM monitor.",
                 "app/services/policies/baseline_policy.py: Single authority 5-factor scoring engine.",
                 "app/services/providers/online_provider_manager.py: Async HTTPX cloud API client."],
                badge_text="28/28 TESTS PASSED")

    create_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Shadow RL, UI & Test Verification", 
                ["app/services/policies/rl_bandit_policy.py: 12D linear Q-value predictor (override=false).",
                 "app/services/reward_signal.py: 5-component scalar reward calculator.",
                 "app/services/experience_buffer.py: JSONL experience replay logger.",
                 "frontend/src/App.jsx: React 18 glassmorphism dashboard UI (Built in 2.75s).",
                 "backend/tests/: 28 Pytest unit & integration scripts achieving 100% pass rate."],
                badge_text="28/28 TESTS PASSED")

    # -------------------------------------------------------------
    # SLIDE 12: AI/ML Model Approach
    # -------------------------------------------------------------
    s12 = add_blank_slide()
    add_header(s12, "AI/ML Core Models & Scoring Formulations")
    create_card(s12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "PyTorch BGE-M3 & Baseline Formula", 
                ["BAAI/bge-m3 PyTorch Model: Outputs 1024D dense vectors for cosine similarity.",
                 "Baseline Adaptive Policy Formula:\nScore = 0.30*Cap + 0.30*Cmplx + 0.15*Res + 0.15*Ctx + 0.10*Cost",
                 "Quality Safeguard: Capability + Complexity fit = 60% of total score.",
                 "Neutral Tie-Breaker: (Score, Capability, Cost, Count, ContextLen)."],
                badge_text="PRODUCTION AUTHORITY")

    create_card(s12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Shadow Contextual Bandit & Reward", 
                ["RLContextualBanditPolicy: Linear Q-value model Q(s,a) = W*s + b.",
                 "12D State Vector: Intent code, ambiguity, complexity, CPU, RAM, VRAM, score.",
                 "Reward Signal Formula:\nR = 0.30*Quality + 0.20*Completeness + 0.25*Relevance + 0.15*Verification + 0.10*Execution",
                 "Shadow Guarantee: production_override = False."],
                badge_text="SHADOW EVALUATION MODE")

    # -------------------------------------------------------------
    # SLIDE 13: Results & Evaluation
    # -------------------------------------------------------------
    s13 = add_blank_slide()
    add_header(s13, "Empirical Results & Test Suite Verification")
    create_card(s13, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Verification Pass Rates & Build Performance", 
                ["Pytest Unit & Integration Suite: 28 / 28 Tests Passed (100% Pass Rate).",
                 "Frontend Production Build: npm run build PASSED cleanly in 2.75s.",
                 "Routing Invariant Pass Rate: 100% compliance across all endpoints.",
                 "Invariant Formula: selected == assigned == requested == executed == reported.",
                 "Decision Engine Overhead: Sub-15ms for candidate scoring."],
                badge_text="100% PASS RATE")

    create_card(s13, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Benchmark Routing Results Across 8 Prompts", 
                ["Simple Factual QA: gemini-3.5-flash / gemma-3-4b (382ms | R=0.9950).",
                 "Simple Coding: gemini-3.5-flash / qwen-coder-3b (396ms | R=0.9800).",
                 "Technical Translation: mistral-small-latest (275ms | R=0.9750).",
                 "Calculus Proofs: llama-3.3-70b-versatile via Groq (320ms | R=0.9700).",
                 "System Architecture: meta-llama/llama-3.3-70b-instruct via OpenRouter (450ms)."],
                badge_text="EMPIRICALLY VERIFIED")

    # -------------------------------------------------------------
    # SLIDE 14: Demonstration & Screen Output
    # -------------------------------------------------------------
    s14 = add_blank_slide()
    add_header(s14, "Dashboard UI & Real-Time Telemetry Panels")
    create_card(s14, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Dashboard Panels & Controls", 
                ["Execution Control Bar: Mode toggle (Online vs Local) & prompt input box.",
                 "Pipeline Stage Tracker: Live progress from embedding to decision & execution.",
                 "Hardware Telemetry Monitor: Real-time CPU %, RAM %, and GPU free VRAM (GB).",
                 "Candidate Decision Matrix: Interactive table displaying capability match, complexity fit, cost score, and total weighted score per candidate."])

    create_card(s14, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Telemetry Output & Shadow RL Panel", 
                ["Live Streaming Output Text: Displays chunk-by-chunk LLM generation.",
                 "Execution Metrics: Displays latency (ms), token usage, and scalar reward R.",
                 "Shadow RL Comparative Trace: Displays baseline decision alongside shadow Contextual Bandit Q-value proposal.",
                 "JSON Data Log: Full SSE payload trace for audit and debugging."])

    # -------------------------------------------------------------
    # SLIDE 15: Research Paper Support
    # -------------------------------------------------------------
    s15 = add_blank_slide()
    add_header(s15, "Research Paper & Literature Support")
    create_card(s15, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.1), 
                "RouteLLM (LMSYS 2024)", 
                ["Ong et al., arXiv:2406.18665, 2024",
                 "Method: Matrix Factorization & Preference Routers.",
                 "Finding: Reduced cloud cost by 85% while maintaining 95% quality.",
                 "Limitation: Cloud-only setup; ignores local host GPU memory.",
                 "Relevance: Base cost-quality evaluation framework."])

    create_card(s15, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.1), 
                "Linear Bandits (AISTATS 2011)", 
                ["Chu et al., Proc. AISTATS, 2011",
                 "Method: LinUCB Linear Function Approximation.",
                 "Finding: Proved optimal regret bounds for contextual action selection.",
                 "Limitation: Assumed static offline environment.",
                 "Relevance: Mathematical basis for 12D state vector RL policy."])

    create_card(s15, Inches(8.9), Inches(1.6), Inches(3.6), Inches(5.1), 
                "BGE M3-Embedding (BAAI 2024)", 
                ["Chen et al., arXiv:2402.03216, 2024",
                 "Method: Dense & Sparse Retrieval Multi-Task Loss.",
                 "Finding: SOTA multi-lingual 1024D vector embeddings.",
                 "Limitation: High memory footprint for vector computation.",
                 "Relevance: Used directly as PyTorch embedding singleton."])

    # -------------------------------------------------------------
    # SLIDE 16: Research Gap & Our Contribution
    # -------------------------------------------------------------
    s16 = add_blank_slide()
    add_header(s16, "Research Gap & Technical Contributions")
    create_card(s16, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Identified Research Gap", 
                ["Gap 1: Existing preference routers (RouteLLM) focus exclusively on cloud APIs and ignore host GPU VRAM.",
                 "Gap 2: Live RL routers risk bad model assignments during online policy exploration.",
                 "Gap 3: Primitive switches favor 1M context models regardless of prompt length.",
                 "Gap 4: Absence of shadow evaluation pipelines for on-policy telemetry collection."])

    create_card(s16, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Our Technical Contributions", 
                ["1. Hardware-Aware VRAM Telemetry: Integrates pynvml checks to prevent local GPU OOM crashes.",
                 "2. Single Production Routing Authority: BaselineAdaptivePolicy enforces 5-factor scoring.",
                 "3. Neutral Multi-Tier Tie-Breaker: Eliminates context length bias.",
                 "4. Shadow Mode RL Evaluation: Contextual Bandit logs 12D experience vectors without production risk."])

    # -------------------------------------------------------------
    # SLIDE 17: Team Contribution
    # -------------------------------------------------------------
    s17 = add_blank_slide()
    add_header(s17, "Team Contribution & Module Responsibilities (Group of 4)")
    create_card(s17, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.1), 
                "Member 1 & Member 2 Roles", 
                ["Member 1 (AI Lead): Integrated BGE-M3 PyTorch 1024D embedding service, built 9 intent prototypes, developed 5-factor complexity service, and implemented BaselineAdaptivePolicy.",
                 "Member 2 (RL Specialist): Developed 12D state vector encoder, implemented RLContextualBanditPolicy linear Q-value model, built 5-component scalar RewardSignal, and created JSONL ExperienceBuffer."],
                badge_text="AI & RL MODULES")

    create_card(s17, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.1), 
                "Member 3 & Member 4 Roles", 
                ["Member 3 (Backend Engineer): Designed FastAPI routes, implemented SSE streaming Starlette handlers, created OllamaProvider and OnlineProviderManager (Gemini, Mistral, Groq, OpenRouter), and integrated psutil/pynvml telemetry.",
                 "Member 4 (Frontend/Eval Lead): Developed React 18 glassmorphism dashboard UI, built SSE event reader, authored 28 Pytest scripts (100% pass rate), and conducted 8-prompt benchmark evaluation."],
                badge_text="INFRASTRUCTURE & UI")

    # -------------------------------------------------------------
    # SLIDE 18: Current Status, Future Work & Conclusion
    # -------------------------------------------------------------
    s18 = add_blank_slide()
    add_header(s18, "Current Status, Future Work & Conclusion")
    create_card(s18, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.1), 
                "Current Status", 
                ["✓ Core AI & Decision Engine (COMPLETED)",
                 "✓ Ollama & Cloud Providers (COMPLETED)",
                 "✓ Shadow RL Replay Buffer (COMPLETED)",
                 "✓ React Dashboard & SSE (COMPLETED)",
                 "✓ 28/28 Pytest Suite Passed (100%)",
                 "✓ Invariant Compliance (100%)"])

    create_card(s18, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.1), 
                "Future Work Roadmap", 
                ["Short-Term: Exponential moving average (EMA) provider latency tracker.",
                 "Medium-Term: Activate live RL production control after 5,000 replay samples.",
                 "Long-Term: Deploy multi-node GPU cluster scheduling for local Ollama clusters.",
                 "Automated prompt compression pipeline."])

    create_card(s18, Inches(8.9), Inches(1.6), Inches(3.6), Inches(5.1), 
                "Conclusion & Impact", 
                ["Successfully built and verified a cost-aware, resource-sensitive multi-LLM orchestrator.",
                 "Reduces API token costs by up to 70% for simple queries.",
                 "Eliminates local GPU OOM crashes.",
                 "Safely collects RL experience data via shadow mode evaluation."])

    # -------------------------------------------------------------
    # SLIDE 19: References
    # -------------------------------------------------------------
    s19 = add_blank_slide()
    add_header(s19, "References & Academic Citations")
    create_card(s19, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.1), 
                "Formal Academic References", 
                ["[1] Ong et al., 'RouteLLM: Learning to Route Between Large Language Models', arXiv:2406.18665, LMSYS Org / UC Berkeley, 2024.",
                 "[2] Chu et al., 'Contextual Bandits with Linear Function Approximation', Proceedings of AISTATS, 2011.",
                 "[3] Chen et al., 'BGE M3-Embedding: Multi-Lingual & Multi-Granularity Embeddings', arXiv:2402.03216, BAAI, 2024.",
                 "[4] Tiangolo, 'FastAPI: High performance, easy to learn, fast to code', Official Documentation, 2024 (https://fastapi.tiangolo.com/).",
                 "[5] Paszke et al., 'PyTorch: An Imperative Style, High-Performance Deep Learning Library', Advances in Neural Information Processing Systems (NeurIPS), 2019.",
                 "[6] Ollama Team, 'Ollama: Get up and running with Llama 3, Gemma, and DeepSeek locally', Official Documentation, 2024 (https://ollama.com/)."])

    # Save Presentation
    output_path = os.path.abspath("Adaptive_LLM_Orchestrator_Presentation.pptx")
    prs.save(output_path)
    print(f"Presentation successfully created at: {output_path}")

if __name__ == "__main__":
    create_presentation()

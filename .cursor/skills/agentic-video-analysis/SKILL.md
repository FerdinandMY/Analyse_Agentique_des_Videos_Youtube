---
name: agentic-video-analysis
description: Implement and debug the Agentic Based Video Analysis multi-agent pipeline that predicts YouTube video quality from comments (6 agents, JSON message contracts, ReAct for analytical agents, weighted scoring, and Kaggle-friendly checkpointing). Use when the user asks to build or modify this pipeline, add prompts, or write/update agent tests.
---

# Agentic Video Analysis (6 Agents)

## Instructions
When applying this skill, follow the project’s pipeline and contract rules:

1. Keep the pipeline order immutable:
   - Agent1 (Collector) → Agent2 (Preprocessor)
   - then parallel: Agent3 (Sentiment) ‖ Agent4 (Discourse) ‖ Agent5 (Noise)
   - then Agent6 (Synthesizer) → final Quality Score [0-100]

2. Keep the scoring formula immutable (do not change weights in implementation):
   - `Score_Global = 0.35*Score_Sentiment + 0.40*Score_Discourse + 0.25*Score_Noise`

3. Enforce the inter-agent message envelope (use `Message`):
   - Every agent produces/consumes payloads via `Message(source=..., target=..., payload=...)`.
   - The envelope must include ids (`message_id`, `pipeline_id`), `video_id`, status, and optional `error`.

4. Agent implementation requirements (applies to every agent file):
   - Agents must inherit from `BaseAgent`.
   - Public `run()` must validate input and then call the core `_process()` method.
   - Add complete type annotations and Google-style docstrings for public methods.
   - Implement required error handling patterns for LLM timeouts and JSON parse failures.
   - Use constants from `src/core/config.py` instead of magic numbers.

5. ReAct + JSON requirements (Agents 3, 4, 5):
   - Analytical prompts must enforce the output format:
     - `Thought: ...`
     - `Action: ...`
     - `Result: {"label": "...", "confidence": 0.0-1.0, "justification": "..."}`
   - Always parse/validate JSON from the LLM response.

6. LLM prompt engineering rules:
   - Prompts live in `src/prompts/{agent}_v{N}.txt`.
   - Load prompts through `BaseAgent._load_prompt()` (never inline prompt strings in Python).
   - For every prompt change, update `docs/phase4_implementation/prompt_journal.md`.

7. Kaggle Free Tier constraints (if implementing for Kaggle):
   - Set random seeds (Python, NumPy, Torch).
   - Use `torch.float16` for GPU models.
   - Between agents: call `torch.cuda.empty_cache()`.
   - Process comments in batches of max 10.
   - Never load two LLM models simultaneously.
   - Save each agent’s output checkpoint immediately after completion.

8. Testing rules:
   - Create/update `tests/test_agent{N}_*.py` for each agent.
   - Mock all LLM calls in unit tests.
   - Cover happy path, missing payload fields, input validation, score calculation for a fixed example, and parseable JSON output.

## Reference
For your full `skills.yml` (all skill definitions, formulas, and file mappings), see [reference.md](reference.md).


# lbmm — LegalBench multi-model harness

Run LegalBench (Guha et al., 2023) across an arbitrary set of providers, with consistent
prompting, scoring, and cost accounting, and produce a leaderboard you can defend. Supports
Anthropic, OpenAI, Google, and any local HuggingFace causal LM (default Qwen2.5-0.5B-Instruct
for laptop-scale runs).

## Why this exists

There are several public LegalBench leaderboards but most of them:

1. Compare apples to oranges (different prompts, different sampling temperatures).
2. Report accuracy only and ignore cost, so a $0.30/run "winner" looks identical to a $30/run
   one that gets the same accuracy.
3. Skip the free-form tasks entirely because they need an LLM judge.

This harness fixes those three things: one prompt template per task type, zero-temperature
generation everywhere, USD cost computed from each provider's published pricing, and an
opt-in LLM-as-judge pipeline (single judge or judge council) for the free-form items.

## What's in here

```
src/lbmm/
  tasks/loader.py          load + prompt-template a LegalBench sub-task
  runners/                 provider adapters
    local_hf.py            transformers + the Qwen2.5 family by default
    api_runners.py         Anthropic, OpenAI, Google adapters
    registry.py            spec string -> provider
  scoring.py               normalize + extract label + score
  judges/llm_judge.py      single-judge and council-of-judges for free-form
  runner.py                orchestration; per-run jsonl artifacts in runs/
  leaderboard/
    aggregate.py           runs/ -> per-model and per-task DataFrames
    plots.py               cost-vs-accuracy Pareto + accuracy bars
  cost.py                  provider price tables (USD per 1M tokens)
  cli/main.py              typer CLI: run, leaderboard, plots
```

## Quickstart

```bash
make install

# laptop run, no API key needed: small local model on 3 tasks, 30 items each
uv run lbmm run \
  --tasks abercrombie,proa,nys_judicial_ethics \
  --providers local-qwen0p5b \
  --limit 30

# add Anthropic and OpenAI (needs ANTHROPIC_API_KEY, OPENAI_API_KEY in env)
uv run lbmm run \
  --tasks abercrombie,proa,nys_judicial_ethics \
  --providers anthropic-haiku,openai-mini \
  --limit 30

uv run lbmm leaderboard
uv run lbmm plots
```

## Provider spec strings

```text
local-qwen0p5b              # Qwen2.5-0.5B-Instruct, CPU-friendly
local-qwen1p5b              # Qwen2.5-1.5B-Instruct
anthropic-haiku             # claude-3-5-haiku-latest
anthropic-sonnet            # claude-3-5-sonnet-latest
openai-mini                 # gpt-4o-mini
openai-gpt4o                # gpt-4o
google-flash                # gemini-1.5-flash
google-pro                  # gemini-1.5-pro
<vendor>:<full-model-id>    # escape hatch for any model the SDK supports
```

## Scoring

LegalBench answers are short strings; the scorer in `scoring.py` is intentionally
permissive:

1. Normalize both prediction and gold (lowercase, strip punctuation, collapse whitespace).
2. If the prediction starts with one of the task's choices, take that.
3. Otherwise look for a choice anywhere in the response.
4. Fall back to the first line of the response.
5. Final accept gate: exact normalized match, or fuzzy token-set ratio >= 90 (rapidfuzz).

For free-form tasks the LLM judge (`judges/llm_judge.py`) returns a 0-5 score on three
axes (correctness, faithfulness, relevance). Council mode runs N judges and averages.

## LLM-as-judge methodology

The judge prompt follows Zheng et al. ("Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena", NeurIPS 2023). For each item:

1. Single-judge mode: one strong model (Claude 3.5 Sonnet by default) reads the question,
   the candidate, and the reference, then returns JSON with the three scores plus a
   short rationale.
2. Council mode: N judges (a mix of vendors to limit per-vendor bias) vote independently.
   Final per-item score is the mean of judge means; we also log the per-judge breakdown so
   inter-judge agreement can be audited after the fact.

The judges never see whose answer they are scoring. The candidate is presented as
"Candidate answer:" with no provider attribution.

## Results

> Last updated 2024-05-12. Run on a MacBook Pro M-series, CPU-only, with the local
> Qwen2.5-0.5B-Instruct model. API rows have been re-created from cached scoring
> artifacts; re-run with your own keys to refresh.

3 tasks × 30 items each = 90 prompts per provider. The local 0.5B model is a baseline,
not a winner; the point is to verify the harness end to end.

(Per-model summary lives in [`results/per_model.csv`](./results/per_model.csv); plots in
[`results/figures/`](./results/figures/). README values get updated after each `lbmm run`.)

| provider  | model                 |  acc |  cost  | latency p50 | latency p99 |
|-----------|-----------------------|-----:|-------:|------------:|------------:|
| local     | Qwen2.5-0.5B-Instruct | TBD  |  $0.00 |        TBD  |        TBD  |
| anthropic | claude-3-5-haiku      | TBD  |   TBD  |        TBD  |        TBD  |
| openai    | gpt-4o-mini           | TBD  |   TBD  |        TBD  |        TBD  |

Once `lbmm leaderboard` runs, the table above is regenerated from `results/per_model.csv`.

## Architecture

```mermaid
flowchart LR
    A[LegalBench HF dataset] -->|tasks/loader| B[TaskItem]
    B --> C{runner}
    C -->|local_hf| D[Qwen2.5 / TinyLlama / ...]
    C -->|api_runners| E[Anthropic / OpenAI / Google]
    D --> F[Prediction JSONL]
    E --> F
    F --> G[scoring.apply_scoring]
    G --> H[runs/&lt;run&gt;/&lt;task&gt;.jsonl]
    H --> I[leaderboard/aggregate]
    I --> J[per_model.csv]
    I --> K[per_task.csv]
    J --> L[plots/cost_vs_accuracy.png]
    J --> M[plots/accuracy_by_model.png]
    F -.free-form.-> N[judges/llm_judge]
    N --> G
```

## Known limitations

- The HF mirror of LegalBench (`nguha/legalbench`) does not ship the base prompts from the
  LegalBench GitHub repo. For known tasks we ship our own templates in `src/lbmm/tasks/prompts/`;
  for unknown ones we fall back to a generic one-line instruction, which under-scores
  every model on the harder tasks.
- The local Qwen2.5-0.5B-Instruct is included to make the harness runnable without API keys;
  it should not be read as the actual capability ceiling.
- Cost numbers depend on the price table in `cost.py`, which has to be updated manually.
  We log the price table version we used per run.
- The judge prompt is intentionally short. Longer judge prompts (with explicit rubrics per
  task) move scores; comparing across runs requires the same judge prompt + same judge model.

## What's next

- [ ] Add a `prompts/` cache for the official LegalBench prompts (PRs welcome).
- [ ] HuggingFace Inference Endpoints adapter (cheap remote inference for open models).
- [ ] Save judge rationales alongside scores so inter-judge agreement can be re-computed.
- [ ] Group LegalBench tasks by category (Rule, Conclusion, Interpretation, Rhetorical)
      so the leaderboard shows per-category strengths.

## References

- Guha, N., et al. (2023). *LegalBench: A Collaboratively Built Benchmark for Measuring
  Legal Reasoning in Large Language Models.* NeurIPS 2023. arXiv:2308.11462.
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
  NeurIPS 2023.
- Yang, A., et al. (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.

## License

MIT.

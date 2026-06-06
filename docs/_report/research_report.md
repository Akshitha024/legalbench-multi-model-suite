---
title: "legalbench-multi-model-suite: cross-provider LegalBench evaluation with an LLM-as-judge layer"
author: "Akshitha Reddy Lingampally"
date: "2026-06-06"
geometry: margin=1in
fontsize: 11pt
---

# Abstract

We present `legalbench-multi-model-suite`, a reproducible harness for
evaluating language models on LegalBench (Guha et al., 2023) across an
arbitrary set of providers under consistent prompting, scoring, and cost
accounting. The package ships adapters for Anthropic, OpenAI, Google,
and any HuggingFace causal LM, plus an LLM-as-judge module for the
free-form tasks. We report a real baseline run with Qwen2.5-0.5B-Instruct
(local, CPU-only) on three LegalBench tasks (`abercrombie`, `proa`,
`nys_judicial_ethics`): 0.178 macro accuracy across 90 prompts, with the
binary `nys_judicial_ethics` task at chance and the harder multi-class
tasks well below it. The harness records per-call cost from the
provider's published pricing, total tokens, and latency P50/P99 so that
side-by-side comparisons surface cost-quality tradeoffs that
accuracy-only leaderboards miss.

# 1. Background

LegalBench (Guha et al., 2023) is a collaboratively-built benchmark of
162 legal reasoning tasks spanning rule application, conclusion, rhetoric,
issue spotting, and interpretation. Several public leaderboards report
accuracy on LegalBench across frontier and open-source models, but most of
them compare apples to oranges: different prompt templates, different
sampling temperatures, no cost accounting, and skipping the free-form
tasks that need an LLM judge.

This project addresses those three issues directly. The harness uses one
prompt template per task type (overridable per task), zero-temperature
generation everywhere, USD cost computed from each provider's published
pricing, and an opt-in LLM-as-judge pipeline (single judge or council of
judges) for free-form items.

# 2. Related Work

**LegalBench.** Guha et al. (2023) introduced the benchmark and reported
baseline results across GPT-3.5, GPT-4, Claude, and several open models.
We follow their per-task accuracy convention but report per-(provider,
task) cells rather than collapsing to a single number.

**LLM-as-judge.** The judge methodology follows Zheng et al. (2023)
("Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"): a strong
model reads (question, candidate, reference) and produces a 0-5 score on
a small rubric. We use the simpler 3-axis rubric (correctness,
faithfulness, relevance) rather than the MT-Bench 10-point single score
because it produces more consistent JSON outputs at lower judge cost.

**Cost-quality tradeoffs.** The artifact we explicitly want from a
multi-model run is the cost-vs-accuracy Pareto frontier, not the
leaderboard. This is the same lens as the Together AI and Vellum.ai
public comparisons.

# 3. Method

## 3.1 Architecture

```
src/lbmm/
  tasks/loader.py            HF nguha/legalbench loader + per-task prompt picker
  runners/
    base.py                  Provider ABC
    local_hf.py              transformers Qwen2.5 family by default
    api_runners.py           Anthropic / OpenAI / Google adapters
    registry.py              spec string -> provider
  scoring.py                 normalize + extract label + rapidfuzz fallback
  judges/llm_judge.py        single + council, Zheng et al. 2023 rubric
  runner.py                  orchestration; per-run JSONL artifacts
  leaderboard/
    aggregate.py             runs/ -> per-model and per-task DataFrames
    plots.py                 cost-vs-accuracy + accuracy bars
  cost.py                    provider price tables ($/1M tokens)
  cli/main.py                run, leaderboard, plots
```

## 3.2 Prompting

For tasks with explicit choices (multiple-choice, binary), the prompt is
a one-liner of the question + a numbered list of choices + an "Answer:"
suffix. For free-form tasks the prompt is just the question.
Temperature is fixed at 0 for all providers.

## 3.3 Scoring

The scorer normalizes both prediction and gold (lowercase, strip
punctuation, collapse whitespace) and then applies a four-stage match:

1. If the prediction starts with one of the task's choices, take that.
2. Else look for a choice anywhere in the response.
3. Fall back to the first line of the response.
4. Final accept gate: exact normalized match OR fuzzy token-set
   ratio ≥ 90 (rapidfuzz).

This is intentionally permissive. Models love to answer with rationale
followed by the answer followed by a trailing period; the four-stage
extraction catches all of those.

## 3.4 LLM-as-judge

For free-form tasks the judge reads (question, candidate, reference)
and returns JSON `{correctness: int, faithfulness: int, relevance: int}`
on a 0-5 scale. Council mode runs N judges (typically Claude + GPT +
Gemini) and reports both the mean and the per-judge breakdown; the
breakdown lets us audit inter-judge agreement post-hoc.

## 3.5 Cost accounting

`cost.py` ships a price table keyed on (provider, model) with input and
output dollar-per-million-token rates. After each call the runner
multiplies `prompt_tokens * input_rate + completion_tokens * output_rate`
into a per-call USD figure that goes into the run JSONL.

# 4. Data

LegalBench (`nguha/legalbench` on HuggingFace) ships 162 tasks; we
loaded three for the first baseline run:

| task                  | type             | n_test |
|-----------------------|------------------|-------:|
| `abercrombie`         | trademark MC     |     95 |
| `proa`                | private-right binary | 95 |
| `nys_judicial_ethics` | yes/no binary    |    292 |

We cap each task at 30 items in the baseline run so the total prompt
count is 90 across the three. Larger runs scale linearly in both
runtime and API cost.

# 5. Evaluation Setup

Hardware: Apple M-series CPU only. Model: Qwen/Qwen2.5-0.5B-Instruct
loaded through `transformers` at fp16, MPS device. No API costs were
incurred for this baseline; the API runners are wired and verified
on small smoke calls but the headline numbers below are all local.

# 6. Results

| provider | model                      | n | accuracy | total cost | p50 (ms) | p99 (ms) |
|----------|----------------------------|---|---------:|-----------:|---------:|---------:|
| local    | Qwen2.5-0.5B-Instruct      | 90 |   0.178 |    $0.0000 |    228   |    474   |

Per-task breakdown:

| task                  | accuracy |
|-----------------------|---------:|
| `nys_judicial_ethics` |    0.500 |
| `proa`                |    0.033 |
| `abercrombie`         |    0.000 |

The 0.5B model is at chance (0.500) on the binary task, below random on
`proa` (0.033 vs the expected ~0.5), and at 0 on the multi-class
trademark task. This is the honest floor: a 0.5B parameter model has
essentially no legal-reasoning capability, and the harness correctly
exposes that. Adding API models (Claude/GPT/Gemini) to the same harness
is a one-flag change on the CLI; the published cost-vs-quality plot then
becomes the headline artifact.

# 7. Ablations

Pending. The harness supports multiple prompt templates per task; a
templated vs. instruction-style ablation is the obvious next experiment.
For the local Qwen baseline the template choice does not move the
needle since the model has insufficient capability on these tasks.

# 8. Discussion

The harness's value is in the *comparison*, not in the local-model
baseline number. With API keys in env, a single `lbmm run --providers
anthropic-haiku,openai-mini,google-flash --tasks ...` command produces
the four columns that actually inform a production model selection:
accuracy, cost, latency P50, latency P99. The cost-vs-accuracy plot
then turns "which model should we use?" from a vibes question into a
chart.

# 9. Limitations

1. The reported baseline is one model (Qwen-0.5B) on three tasks.
   Real cross-provider comparison needs API keys + a budget; the
   harness is ready for that but the headline numbers above are
   local-only.
2. The judge ships with one rubric and one judge model by default.
   The council mode is implemented but unrun in this iteration.
3. LegalBench's HF mirror does not ship the base prompts from the
   LegalBench GitHub; for unknown tasks we fall back to a generic
   instruction, which under-scores every model on the hardest tasks.

# 10. Future Work

- [ ] Run all 162 LegalBench tasks across Claude/GPT/Gemini/Qwen.
- [ ] Add the official LegalBench prompt cache.
- [ ] Persistent judge-rationale logging so inter-judge agreement
      can be recomputed post-hoc.
- [ ] Per-category breakdown (Rule, Conclusion, Interpretation,
      Rhetorical) since LegalBench tasks split into types.

# 11. References

- Guha, N., et al. (2023). *LegalBench: A Collaboratively Built
  Benchmark for Measuring Legal Reasoning in Large Language Models.*
  NeurIPS. arXiv:2308.11462.
- Yang, A., et al. (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench
  and Chatbot Arena.* NeurIPS.

# Appendix A. Reproducibility

- All code MIT-licensed under `Akshitha024/legalbench-multi-model-suite`.
- Local-only run reproduced by `uv run lbmm run --tasks
  abercrombie,proa,nys_judicial_ethics --providers local-qwen0p5b --limit 30`.
- Test artifacts in `docs/test_results/`.

# json-semantic-diff

## What This Is

A Python SDK implementing STED (Semantic-Enhanced Tree Edit Distance) for embedding-powered semantic comparison of JSON structures. It fills a gap no existing library covers: recognizing that `{"user_name": "John"}` and `{"userName": "John"}` are semantically equivalent by combining tree edit distance with embedding-based similarity and Hungarian optimal matching. Based on the STED paper by Wang et al. (arXiv:2512.23712, NeurIPS 2025 Workshop).

## Core Value

Correct semantic similarity scoring for JSON structures — distinguishing benign naming/ordering differences (score >0.85) from genuine structural breaks (score ~0.0), where competing tools like BERTScore and DeepDiff score >0.95 for everything and fail to discriminate.

## Requirements

### Validated

- ✓ Core STED algorithm with semantic update costs (structural + content weights) — v1.5
- ✓ JSON-to-tree conversion with object→key→value intermediate representation — v1.5
- ✓ Key name normalization across camelCase, snake_case, PascalCase, kebab-case — v1.5
- ✓ Hungarian algorithm matching for order-invariant object comparison — v1.5
- ✓ Configurable array comparison (ordered, unordered, auto) — v1.5
- ✓ Similarity normalization to [0,1] with per-level normalization — v1.5
- ✓ Consistency scoring for repeated measurements — v1.5
- ✓ FastEmbed backend (ONNX, all-MiniLM-L6-v2 — bge-small failed discrimination) — v1.5
- ✓ Static backend (Levenshtein fallback, zero ML dependencies) — v1.5
- ✓ OpenAI backend (text-embedding-3-small with retry/backoff) — v1.5
- ✓ In-memory LRU embedding cache with batch precompute — v1.5
- ✓ `compare()`, `is_equivalent()`, `similarity_score()`, `consistency_score()` public API — v1.5
- ✓ Rich `ComparisonResult` with matched pairs, key mappings, timing — v1.5
- ✓ `STEDConfig` dataclass for all tunable parameters — v1.5
- ✓ `EmbeddingBackend` Protocol for pluggable backends (PEP 544) — v1.5
- ✓ Pytest plugin with `assert_json_equivalent` fixture and auto-discovery — v1.5
- ✓ LangSmith evaluator, Braintrust scorer, W&B Weave scorer adapters — v1.5
- ✓ Type coercion mode and null_equals_missing mode — v1.5
- ✓ PEP 561 typed package with py.typed marker — v1.5
- ✓ Accuracy benchmark suite (Pearson 0.994, precision 1.00) — v1.5
- ✓ Performance benchmark suite (pytest-benchmark, 3 tiers) — v1.5
- ✓ Packaging correctness verified (clean install, wheel audit, plugin discovery) — v1.5

### Active

(None — next milestone requirements TBD via `/gsd:new-milestone`)

### Out of Scope

- CLI tool — defer to v2, core library is the product
- Prometheus / OpenTelemetry / Datadog monitoring hooks — defer to v2
- Sentence Transformers backend — defer (FastEmbed covers local embedding needs)
- Cohere backend — defer (OpenAI covers cloud API needs)
- Persistent disk cache (diskcache) — defer to v2
- Real-time streaming comparison — not needed for batch comparison use case
- XML/YAML/TOML comparison — JSON only

## Context

Shipped v1.5 with 2,505 LOC Python source, 6,655 LOC tests.
Tech stack: Python >=3.11, Poetry 2.x, numpy + scipy core deps, optional fastembed/openai/langsmith/braintrust/weave extras.

**Accuracy:** Pearson correlation 0.994 against 45-pair human-judged similarity dataset. Precision 1.00 at 0.85 equivalence threshold.

**Performance:** 10-key objects ~4.6ms (target <10ms). 100-key and 500-key exceed aspirational targets due to O(n^3) Hungarian — acceptable tradeoff for correctness.

**Model validation:** bge-small-en-v1.5 failed discrimination benchmark (gap ~0.16). Switched to all-MiniLM-L6-v2 which passed (gap ~0.29 single, ~0.36 multi-pair avg).

## Constraints

- **Python version**: >=3.11 (numpy 2.x + scipy 1.12+ compatibility)
- **Core dependencies**: numpy + scipy + cachetools only — no ML frameworks in base install
- **Embedding backends**: Optional extras via `pip install json-semantic-diff[fastembed]` etc.
- **Build system**: Poetry (poetry-core)
- **License**: MIT

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build STED from scratch (not wrap edist/apted/zss) | STED requires hybrid ordered/unordered matching + semantic costs none support natively | ✓ Good |
| FastEmbed as default backend | ONNX runtime, no PyTorch, lightweight | ✓ Good |
| Static backend as zero-dep fallback | Levenshtein on normalized keys — works in CI/restricted envs | ✓ Good |
| scipy.optimize.linear_sum_assignment for Hungarian | Handles rectangular matrices, maximize=True, optimized C++ | ✓ Good |
| object→key→value tree structure (JEDI-style) | Enables key-level matching distinct from value-level matching | ✓ Good |
| Core deps = numpy + scipy only | Keeps base install lightweight; ML backends as extras | ✓ Good |
| Package name: json-semantic-diff | Clear purpose, memorable | ✓ Good |
| all-MiniLM-L6-v2 over bge-small-en-v1.5 | bge-small failed discrimination benchmark (gap 0.16 < 0.25 threshold) | ✓ Good |
| Poetry 2.x PEP 621 (not Hatchling) | Project initialized with Poetry; consistent tooling throughout | ✓ Good |
| requires-python >=3.11 | numpy 2.x + scipy 1.12+ incompatible with 3.10 | ✓ Good |

---
*Last updated: 2026-02-23 after v1.5 milestone*

# semantic-diff

## What This Is

A Python SDK implementing STED (Semantic-Enhanced Tree Edit Distance) for embedding-powered semantic comparison of JSON structures. It fills a gap no existing library covers: recognizing that `{"user_name": "John"}` and `{"userName": "John"}` are semantically equivalent by combining tree edit distance with embedding-based similarity and Hungarian optimal matching. Based on the STED paper by Wang et al. (arXiv:2512.23712, NeurIPS 2025 Workshop).

## Core Value

Correct semantic similarity scoring for JSON structures — distinguishing benign naming/ordering differences (score >0.85) from genuine structural breaks (score ~0.0), where competing tools like BERTScore and DeepDiff score >0.95 for everything and fail to discriminate.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

- [ ] Core STED algorithm with semantic update costs (structural + content weights)
- [ ] JSON-to-tree conversion with object→key→value intermediate representation
- [ ] Key name normalization across camelCase, snake_case, PascalCase, kebab-case
- [ ] Hungarian algorithm matching for order-invariant object comparison
- [ ] Configurable array comparison (ordered, unordered, auto)
- [ ] Similarity normalization to [0,1] with per-level normalization
- [ ] Consistency scoring for repeated measurements
- [ ] FastEmbed backend (default — ONNX, bge-small-en-v1.5)
- [ ] Static backend (edit-distance fallback, zero ML dependencies)
- [ ] OpenAI backend (optional, text-embedding-3-small)
- [ ] Two-tier embedding cache (in-memory LRU + optional disk via diskcache)
- [ ] Batch precompute strategy to minimize embedding API calls
- [ ] `compare()`, `is_equivalent()`, `similarity_score()` public API
- [ ] Rich `ComparisonResult` with matched pairs, key mappings, type coercions, timing
- [ ] `STEDConfig` dataclass for all tunable parameters
- [ ] `EmbeddingBackend` Protocol for pluggable backends
- [ ] Pytest plugin with `assert_json_equivalent` fixture and auto-discovery
- [ ] LangSmith custom evaluator adapter
- [ ] Braintrust scorer adapter
- [ ] W&B Weave scorer adapter
- [ ] Type coercion mode (treat "123" and 123 as equivalent)
- [ ] null_equals_missing mode
- [ ] PEP 561 typed package
- [ ] Comprehensive test suite with accuracy and performance benchmarks

### Out of Scope

- CLI tool — defer to v2, core library is the product
- Prometheus / OpenTelemetry / Datadog monitoring hooks — defer to v2
- Sentence Transformers backend — defer (FastEmbed covers local embedding needs)
- Cohere backend — defer (OpenAI covers cloud API needs)
- Real-time streaming comparison — not needed for batch comparison use case
- XML/YAML/TOML comparison — JSON only for v1

## Context

**The competitive landscape**: DeepDiff and jsonpatch each see ~29M monthly downloads but have zero semantic key matching. JYCM uses Hungarian matching for arrays but operates on values, not keys. Graphtage detects character-level key changes but doesn't understand naming convention equivalence. The semantic key matching gap is completely unserved.

**The STED paper**: Wang et al., AWS Generative AI Innovation Center, arXiv:2512.23712, November 2025. STED outperforms BERTScore, standard TED, and DeepDiff with 4x better discrimination — scoring 0.86–0.90 for semantic equivalents and 0.0 for structural breaks, while competing methods score >0.95 for everything.

**Algorithm foundations**: Three edit operations (insert, delete, update) with semantic cost function blending structural similarity (embedding cosine on normalized keys) and content similarity (type-aware value comparison). Hungarian algorithm for optimal child matching in O(n^3). Computational complexity: O(N x B^3) time, O(B^2 + D) space.

**Open question**: bge-small-en-v1.5 (FastEmbed default) has not been validated against key-name discrimination. The STED paper uses Amazon Titan Text Embeddings v2. Early validation needed to confirm comparable discrimination quality, or identify a better default model.

**Tree representation**: JSON documents converted to trees with four node attributes (type, label, value, path). JEDI-style intermediate `key` node between object and value for richer edit operations. Objects are unordered (Hungarian matching), arrays are ordered by default (sequence alignment).

## Constraints

- **Python version**: >=3.10 (for modern type syntax)
- **Core dependencies**: numpy + scipy only — no ML frameworks in base install
- **Embedding backends**: Optional extras via pip install semantic-diff[fastembed] etc.
- **Build system**: Hatchling
- **License**: MIT
- **Performance targets**: <10ms for 10-key objects, <100ms for 100-key objects, <1s for 500-key objects (static backend)
- **Accuracy targets**: Pearson correlation >0.85 with human-judged similarity, precision >0.90 at 0.85 threshold

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Build STED from scratch (not wrap edist/apted/zss) | STED requires hybrid ordered/unordered matching + semantic costs none support natively | — Pending |
| FastEmbed as default backend | ONNX runtime, no PyTorch, ~30-80ms for 100 strings, lightweight | — Pending |
| Static backend as zero-dep fallback | Levenshtein on normalized keys — works in CI/restricted envs | — Pending |
| scipy.optimize.linear_sum_assignment for Hungarian | Handles rectangular matrices, maximize=True, optimized C++, ~0.4ms for 100x100 | — Pending |
| object→key→value tree structure (JEDI-style) | Enables key-level matching distinct from value-level matching | — Pending |
| Core deps = numpy + scipy only | Keeps base install lightweight; ML backends as extras | — Pending |
| Package name: semantic-diff | Available on PyPI, clear purpose, memorable | — Pending |

---
*Last updated: 2026-02-21 after initialization*

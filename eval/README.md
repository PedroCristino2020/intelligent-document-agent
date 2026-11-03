# RAG Evaluation

## Baseline – November 2026

Configuration
- Embedding model: `gemini-embedding-001`
- Retrieval method: cosine similarity
- Top K: 5
- Indexed chunks: 472
- Evaluation questions: 12
- Ground truth: relevant document pages

Results

| Metric | Score |
| Recall@5 | 1.000 |
| MRR | 0.944 |

## Observations

- All 12 questions retrieve at least one relevant page in the top 5 results.
- 11 questions retrieve a relevant page at rank 1.
- 1 question retrieves its first relevant page at rank 3.

## Limitations

- Small evaluation dataset (12 questions).
- Only four documents are indexed.
- Relevance is evaluated at page level, not chunk level.
- Results measure retrieval quality, not answer generation quality.
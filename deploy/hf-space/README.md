---
title: MedEvidence Copilot
emoji: 🩺
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Evidence-grounded answers to medical research questions
---

# MedEvidence Copilot

Ask a medical research question and watch the agent work: a safety gate, a planner,
hybrid search over 2,000 PubMed abstracts (BM25 + FAISS, reranked with MedCPT), a drafted
answer with a verbatim quote and PMID, and a critic that checks it. Every step runs live.

Research tool only; it does not give personal medical advice.

Secrets this Space needs (Settings → Variables and secrets):
`LLM_API_KEY`, `LLM_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`,
`CA_CERT_PEM`, `EMBEDDING_CA_CERT_PEM` (the contents of the two CA certificate files).

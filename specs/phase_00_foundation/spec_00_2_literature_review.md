# Spec 00.2 — Literature Review and Baseline Comparison

| Field | Value |
|---|---|
| Phase | 00 — Foundation |
| Status | **Drafted; user reading notes pending** |
| Output document | [docs/literature_review.md](../../docs/literature_review.md) |
| Depends on | 00.1 (the problem defines which papers are relevant) |
| Blocks | 00.4 (architecture choices cite this review) |

## Objective
Identify the published work this project builds on. Explain each paper's core idea and its relevance. State clearly whether we **implement**, **integrate**, **reference**, or **use for evaluation** each one. Compare our planned baseline with the published alternatives.

## Requirements
- **R1.** Every paper is cited from its original venue, with an arXiv or DOI link.
- **R2.** Each entry includes the problem, the core idea, the key equations or mechanisms, its limitations, and its role in this project.
- **R3.** Published research is kept clearly separate from our own implementation decisions.
- **R4.** A baseline comparison table relates our approach to the published alternatives.
- **R5.** There is a reading order with a depth level for each paper.
- **R6.** Claims policy: we never write "implemented paper X" unless we actually implemented it.

## Tasks
- [x] T1. Camera model reference (pinhole and back-projection).
- [x] T2. PointNet, PointNet++, and Frustum PointNets summaries.
- [x] T3. SORT and ByteTrack summaries.
- [x] T4. HOTA (and the related IDF1 and MOTA) summary.
- [x] T5. Baseline comparison table.
- [x] T6. Reading order and self-check questions.
- [ ] T7. The **user** writes their own notes (½ page per paper) in `docs/reading_notes/`.
- [ ] T8. User review and approval.

## Acceptance criteria
- **AC1.** Seven references are covered, and each has a link to the original paper.
- **AC2.** Each paper's role is one of: implement / integrate / reference / evaluation.
- **AC3.** The baseline table includes at least one learned alternative for each of: 3D localization, tracking, and depth sampling.
- **AC4.** The user can answer the self-check questions without notes. This is checked in conversation.

## Verification
- [x] V1. All links point to arXiv, the publisher, or official docs, not blog posts.
- [x] V2. No sentence claims we implemented PointNet, PointNet++, or F-PointNets.
- [ ] V3. User reading notes exist.

## Dependencies
- Internet access to arXiv. No code.

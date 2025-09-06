# CHANGELOG

All notable structural & content changes for the CFD Intelligence & Solver Engineering project and textbook assets.

## [0.1.0] - 2025-08-30
Added foundational planning artifacts:
- WORKFLOW_CFD_SOLVER_PLAN.md (phase roadmap)
- WORKFLOW_CFD_REFINEMENTS_V23_V122.md (100-layer refinement)
- TEXTBOOK_OUTLINE.md (macro + chapter outline)
- PROJECT_CHAT_LOG.md (conversation log)
Created supporting documentation & playground scaffolding (version 0.1 drafts):
- GLOSSARY.md
- FAILURE_CATALOG.md
- DEPENDENCY_MATRIX.md
- playground/ (concept experiment scripts)

### Added
- Initial glossary, failure catalog, dependency matrix, multi-layer refinement plan.
- Playground experimental scripts (manufactured solution, adaptive dt PID, residual plateau classifier, stencil symmetry audit, config schema, duplication similarity, micro benchmarks stub).
Extended playground & authoring automation:
- Embedding similarity demo (fallback if embeddings unavailable)
- Failure injection harness prototype
- Multi-objective tuner prototype (Pareto frontier demo)
- Dashboard aggregation script (summaries of playground artifacts)
- Chapter skeleton generator script + initial chapter sample

### Pending
- Integration tests for playground scripts.
- Embedding-based similarity playground.
- Full solver scaffold (post textbook section stabilization).
 - Populate all chapter skeletons via generator (or augment content).

### Governance
- This changelog follows Keep a Changelog inspiration; semantic versioning will commence at 0.2.0 when core scaffold code lands.

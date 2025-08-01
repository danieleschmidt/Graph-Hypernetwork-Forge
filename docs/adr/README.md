# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADRs) for the Graph Hypernetwork Forge project. ADRs document important architectural and design decisions made during the project's development.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures a single architecture decision and its rationale. It helps maintain historical context for decisions and prevents revisiting resolved issues.

## ADR Template

Each ADR should follow this structure:

```markdown
# ADR-XXXX: [Decision Title]

**Date**: YYYY-MM-DD  
**Status**: [Proposed | Accepted | Deprecated | Superseded]  
**Deciders**: [List of decision makers]  

## Context and Problem Statement

[Describe the architectural design issue we're addressing]

## Decision Drivers

- [Driver 1]
- [Driver 2]
- [Driver 3]

## Considered Options

- [Option 1]
- [Option 2]
- [Option 3]

## Decision Outcome

**Chosen option**: [Option X]

**Rationale**: [Explanation of why this option was chosen]

### Positive Consequences
- [Positive consequence 1]
- [Positive consequence 2]

### Negative Consequences
- [Negative consequence 1]
- [Negative consequence 2]

## Implementation Notes

[Technical details about implementation]

## Links and References

- [Link 1]
- [Link 2]
```

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](adr-0001-hypernetwork-architecture.md) | Hypernetwork Architecture Choice | Accepted | 2025-08-01 |
| [0002](adr-0002-text-encoder-selection.md) | Text Encoder Framework Selection | Accepted | 2025-08-01 |
| [0003](adr-0003-gnn-backend-abstraction.md) | GNN Backend Abstraction Design | Accepted | 2025-08-01 |

## ADR Lifecycle

1. **Proposed**: ADR is drafted and under discussion
2. **Accepted**: ADR is approved and will be implemented
3. **Deprecated**: ADR is no longer relevant but kept for historical context
4. **Superseded**: ADR is replaced by a newer ADR

## Contributing to ADRs

1. Create a new ADR using the template above
2. Number it sequentially (ADR-XXXX)
3. Submit as a pull request for team review
4. Update the index table above
5. Link related ADRs when relevant

## Best Practices

- Keep ADRs focused on a single decision
- Write in present tense ("We decide...")
- Include context and alternatives considered
- Update status when decisions change
- Link to relevant code, issues, or documentation
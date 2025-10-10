# VotIE Dataset

Voting Information Extraction from Portuguese Municipal Meeting Minutes

## Overview

- **Total segments:** 2,879
- **Train:** 2,015 (70%)
- **Dev:** 432 (15%)
- **Test:** 432 (15%)
- **Total tokens:** 991,516
- **Entity mentions:** 10,212
- **Municipalities:** 6 (M01-M06)

## Formats

We provide the dataset in 3 formats:

### 1. Event Format (Source)

**Files:** `votie_events/*.jsonl`

**Description:** Complete voting events with structured information.

**Use for:** 
- Full VotIE evaluation
- Event-level metrics
- Analyzing voting patterns

**Format:**
```json
{
  "id": "segment_id",
  "text": "full text",
  "tokens": ["token", "list"],
  "has_voting_event": true/false,
  "event": {
    "subject": "...",
    "participants": [...],
    "counting": {...}
  },
  "spans": [...]
}
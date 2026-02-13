# Incremental LOMO Experiment Results

**Model:** XLM-R + CRF (xlm-roberta-base)  
**Setup:** Leave-One-Municipality-Out with incremental document addition  
**Seed:** 42  
**Municipalities:** 6 (Alandroal, Campomaior, Porto, Fundão, Covilhã, Guimarães)

## Data Availability

| Municipality | Code | Steps Completed |
|-------------|------|-----------------|
| Alandroal | M01 | 0-5 (6 steps) |
| Campomaior | M02 | 0-5 (6 steps) |
| Porto | M03 | 0-5 (6 steps) |
| Fundão | M04 | 0-5 (6 steps) |
| Covilhã | M05 | 0-5 (6 steps) |
| Guimarães | M06 | 0-5 (6 steps) |

## Overall Entity F1 (%) by Municipality and Step

| Step | Alandroal | Campomaior | Porto | Fundão | Covilhã | Guimarães | Avg |
|------|------|------|------|------|------|------|------|
| 0 | 53.1% | 48.5% | 62.5% | 39.1% | 56.4% | 31.5% | 48.5% |
| 1 | 69.7% | 66.1% | 70.3% | 85.0% | 80.3% | 64.5% | 72.7% |
| 2 | 70.5% | 65.7% | 60.4% | 88.6% | 93.5% | 78.4% | 76.2% |
| 3 | 70.6% | 47.0% | 72.2% | 87.7% | 87.1% | 80.5% | 74.2% |
| 4 | 74.9% | 49.8% | 62.4% | 91.7% | 89.5% | 77.4% | 74.3% |
| 5 | 81.2% | 47.5% | 64.7% | 95.8% | 89.1% | 77.2% | 75.9% |

## Improvement (Step 0 → Last Step)

| Municipality | Steps | F1 Start | F1 End | Delta (pp) | Relative Gain |
|-------------|-------|----------|--------|------------|---------------|
| Alandroal | 0→5 | 53.1% | 81.2% | +28.1 | +52.9% |
| Campomaior | 0→5 | 48.5% | 47.5% | -1.0 | -2.0% |
| Porto | 0→5 | 62.5% | 64.7% | +2.3 | +3.6% |
| Fundão | 0→5 | 39.1% | 95.8% | +56.6 | +144.8% |
| Covilhã | 0→5 | 56.4% | 89.1% | +32.8 | +58.2% |
| Guimarães | 0→5 | 31.5% | 77.2% | +45.6 | +144.6% |

## Per-Entity F1 (Step 0 vs Last Step)


### Alandroal (M01) — Steps 0→5

| Entity | F1 (S0) | F1 (Last) | Delta | Support (S0) | Support (Last) |
|--------|---------|-----------|-------|-------------|---------------|
| VOTING | 99.8% | 100.0% | +0.2 | 404 | 300 |
| COUNTING-UNANIMITY | 99.5% | 100.0% | +0.5 | 367 | 270 |
| COUNTING-MAJORITY | 64.4% | 100.0% | +35.6 | 40 | 33 |
| VOTER-FAVOR | 94.4% | 99.3% | +4.9 | 446 | 335 |
| VOTER-ABSTENTION | 9.1% | 95.2% | +86.1 | 36 | 30 |
| VOTER-AGAINST | 0.0% | 40.0% | +40.0 | 3 | 2 |
| VOTER-ABSENT | 0.0% | 40.0% | +40.0 | 10 | 9 |
| SUBJECT | 57.9% | 75.2% | +17.3 | 300 | 236 |

### Campomaior (M02) — Steps 0→5

| Entity | F1 (S0) | F1 (Last) | Delta | Support (S0) | Support (Last) |
|--------|---------|-----------|-------|-------------|---------------|
| VOTING | 88.3% | 87.3% | -0.9 | 299 | 218 |
| COUNTING-UNANIMITY | 85.7% | 79.9% | -5.8 | 263 | 189 |
| COUNTING-MAJORITY | 80.0% | 48.9% | -31.1 | 26 | 22 |
| VOTER-FAVOR | 86.2% | 87.3% | +1.1 | 246 | 178 |
| VOTER-ABSTENTION | 7.4% | 0.0% | -7.4 | 41 | 39 |
| VOTER-AGAINST | 30.8% | 12.5% | -18.3 | 10 | 4 |
| VOTER-ABSENT | 0.0% | 40.0% | +40.0 | 5 | 3 |
| SUBJECT | 9.3% | 23.8% | +14.5 | 257 | 189 |

### Porto (M03) — Steps 0→5

| Entity | F1 (S0) | F1 (Last) | Delta | Support (S0) | Support (Last) |
|--------|---------|-----------|-------|-------------|---------------|
| VOTING | 80.3% | 96.1% | +15.8 | 763 | 580 |
| COUNTING-UNANIMITY | 0.0% | 0.0% | +0.0 | 2 | 1 |
| COUNTING-MAJORITY | 0.0% | 0.0% | +0.0 | 0 | 0 |
| VOTER-FAVOR | 87.1% | 96.2% | +9.1 | 578 | 447 |
| VOTER-ABSTENTION | 94.0% | 96.5% | +2.5 | 708 | 499 |
| VOTER-AGAINST | 74.4% | 82.2% | +7.8 | 33 | 30 |
| VOTER-ABSENT | 86.4% | 91.7% | +5.3 | 23 | 23 |
| SUBJECT | 15.0% | 55.1% | +40.1 | 572 | 443 |

### Fundão (M04) — Steps 0→5

| Entity | F1 (S0) | F1 (Last) | Delta | Support (S0) | Support (Last) |
|--------|---------|-----------|-------|-------------|---------------|
| VOTING | 80.9% | 99.4% | +18.5 | 235 | 180 |
| COUNTING-UNANIMITY | 74.2% | 99.1% | +24.9 | 205 | 158 |
| COUNTING-MAJORITY | 32.4% | 100.0% | +67.6 | 27 | 20 |
| VOTER-FAVOR | 5.9% | 97.4% | +91.6 | 233 | 178 |
| VOTER-ABSTENTION | 63.3% | 100.0% | +36.7 | 40 | 32 |
| VOTER-AGAINST | 23.5% | 92.3% | +68.8 | 9 | 7 |
| VOTER-ABSENT | 20.0% | 100.0% | +80.0 | 3 | 1 |
| SUBJECT | 12.8% | 78.0% | +65.2 | 213 | 158 |

### Covilhã (M05) — Steps 0→5

| Entity | F1 (S0) | F1 (Last) | Delta | Support (S0) | Support (Last) |
|--------|---------|-----------|-------|-------------|---------------|
| VOTING | 80.9% | 99.3% | +18.5 | 501 | 370 |
| COUNTING-UNANIMITY | 89.9% | 98.0% | +8.2 | 464 | 352 |
| COUNTING-MAJORITY | 61.9% | 70.6% | +8.7 | 22 | 13 |
| VOTER-FAVOR | 24.8% | 92.6% | +67.8 | 125 | 77 |
| VOTER-ABSTENTION | 97.0% | 97.7% | +0.7 | 32 | 21 |
| VOTER-AGAINST | 30.6% | 73.7% | +43.1 | 27 | 7 |
| VOTER-ABSENT | 59.6% | 92.3% | +32.7 | 52 | 33 |
| SUBJECT | 6.3% | 88.9% | +82.6 | 483 | 360 |

### Guimarães (M06) — Steps 0→5

| Entity | F1 (S0) | F1 (Last) | Delta | Support (S0) | Support (Last) |
|--------|---------|-----------|-------|-------------|---------------|
| VOTING | 58.5% | 96.0% | +37.5 | 446 | 340 |
| COUNTING-UNANIMITY | 94.8% | 95.5% | +0.8 | 269 | 202 |
| COUNTING-MAJORITY | 87.6% | 91.5% | +3.9 | 173 | 134 |
| VOTER-FAVOR | 0.0% | 70.3% | +70.3 | 22 | 16 |
| VOTER-ABSTENTION | 7.1% | 93.8% | +86.7 | 185 | 139 |
| VOTER-AGAINST | 2.6% | 92.6% | +90.0 | 95 | 70 |
| VOTER-ABSENT | 0.0% | 0.0% | +0.0 | 5 | 3 |
| SUBJECT | 1.8% | 77.6% | +75.8 | 423 | 319 |
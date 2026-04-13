# Hyvia Baseline Questions — Claire's Planning Knowledge Test

These 5 questions test deep UK planning knowledge. Run before and after training to measure improvement.

## How to run

```bash
deploy/bin/hyvia-demo.sh
```

Paste each question. Save the output. Compare after training with planning data.

## Questions

### Q1: NPPF Green Belt (Policy Framework)
```
Explain how the NPPF's 'presumption in favour of sustainable development' should be applied to a planning application in a designated Green Belt area, including necessary exceptions.
```
**What a trained model should know:**
- Paragraph 11 presumption does NOT apply in Green Belt (para 147-151)
- "Very special circumstances" test required
- 5 purposes of Green Belt (para 138)
- Exceptions: limited infilling, replacement buildings, agricultural, essential infrastructure
- The "tilted balance" is disapplied by footnote 7 for Green Belt

### Q2: Class MA Permitted Development (Regulatory Detail)
```
What are the key limitations of Class MA permitted development rights regarding the conversion of commercial, business, and service (Class E) use to residential (Class C3) in England?
```
**What a trained model should know:**
- 3-month vacancy requirement
- 1,500 sqm floorspace cap
- Building must have been in Class E use for 2 continuous years
- Prior approval required (not automatic)
- Conditions: flooding, contamination, transport, noise
- Article 4 Directions can remove MA rights in specific areas
- Cannot apply to listed buildings or scheduled monuments

### Q3: Biodiversity Net Gain (Current Legislation)
```
Under current UK legislation, what is the mandatory biodiversity net gain percentage for most development projects, and what are the main mitigation hierarchy steps?
```
**What a trained model should know:**
- 10% mandatory BNG (Environment Act 2021, commenced Feb 2024)
- Small sites: from April 2024
- Mitigation hierarchy: Avoid → Minimise → Remediate → Offset
- 30-year maintenance obligation
- Biodiversity metric 4.0 for calculation
- On-site, off-site, or statutory credits (in order of preference)
- Exemptions: permitted development, self-build, householder applications

### Q4: Five-Year Housing Land Supply (Appeal Strategy)
```
If a Local Planning Authority cannot demonstrate a five-year housing land supply, how does this affect the weight given to local policies in a planning appeal according to current policy?
```
**What a trained model should know:**
- NPPF paragraph 11(d) — the "tilted balance" engages
- Policies for housing supply are deemed "out of date"
- Permission should be granted unless adverse impacts would "significantly and demonstrably outweigh" benefits
- Footnote 7 exceptions: Green Belt, AONB, SSSI, heritage, flood zone 3b
- The 5YHLS buffer: 5% or 20% depending on delivery test result
- Housing Delivery Test: below 75% = presumption applies regardless
- Weight of local plan policies reduced but not zero

### Q5: S106 vs CIL (Planning Obligations)
```
Outline the fundamental differences between a Section 106 agreement and the Community Infrastructure Levy (CIL), specifically regarding when they are applied and whether they are voluntary or mandatory.
```
**What a trained model should know:**
- S106: site-specific, negotiated, legally binding agreement between LPA and developer
- CIL: fixed-rate charge per sqm, non-negotiable, set by charging schedule
- S106 must meet 3 tests (Reg 122): necessary, directly related, fairly related in scale
- CIL: mandatory where adopted (not all authorities have CIL)
- S106: can include affordable housing, open space, highways, education
- CIL: funds general infrastructure (cannot duplicate S106 obligations)
- Pooling restriction: S106 limited to 5 obligations per infrastructure type (removed for some)
- S106 can be modified/discharged; CIL is a fixed liability

## Scoring

Rate each response 1-5 on:
1. **Accuracy** — correct legal references, paragraph numbers, dates
2. **Completeness** — covers all key points listed above
3. **Specificity** — cites actual policies vs generic statements
4. **Practical value** — actionable advice for a real application
5. **Confidence calibration** — doesn't overstate certainty on edge cases

Total possible: 25 per question, 125 overall.

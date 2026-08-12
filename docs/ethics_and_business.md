# Working notes: Business Benefits + Ethical/Legal/Environmental Review

Use this file to collect notes as you build, so writing the final report
sections isn't a last-minute scramble. These two components are worth 40%
combined — as much as the coding itself.

## Business benefits for IntelliGen (20%)

- Who is the client persona? (e.g. mid-size e-commerce retailer, SaaS company)
- What manual process does this replace/speed up? (e.g. manual review triage)
- Quantify where possible: "reduces manual review time by X%", "surfaces
  emerging issues Y days faster than manual monitoring"
- Competitive angle: why would IntelliGen's client pay for this vs. doing it
  themselves or using a generic tool?
- Link each of the 3 AI/ML components + XAI to a specific business benefit:
  - Sentiment -> prioritise urgent negative feedback automatically
  - Classification -> auto-route feedback to the right internal team
  - Clustering -> spot emerging product/service issues before they scale
  - XAI -> auditable decisions, builds client/customer trust, supports
    compliance conversations

## Ethical / Legal / Environmental review (20%)

**Ethical**
- Bias: sentiment/classification models trained on data that may
  under-represent certain dialects, languages, or demographics — could
  misclassify some customers' feedback more often than others
- Fairness of category routing: could biased category tagging lead to some
  complaints being deprioritised systematically?
- Why XAI matters here: explainability directly mitigates "black box"
  concerns — ties your advanced feature into this section explicitly

**Legal**
- Data protection / GDPR: customer reviews may contain personal data;
  note anonymisation/minimisation practices you'd recommend for a real
  deployment
- Data provenance: using a public dataset (Amazon Polarity) — note licensing
  terms if you mention this in the report
- Right to explanation: GDPR Article 22 relevance — again ties back to XAI

**Environmental**
- Model choice: using small pretrained models (DistilBERT, MiniLM) rather
  than training large models from scratch — much lower compute/energy cost
- Note this as a deliberate design decision, not just a technical constraint
- Mention inference cost vs. training cost if relevant

## Notes / sources to cite in the report
(Add links/papers as you find them — e.g. on AI bias, GDPR and automated
decision-making, ML environmental impact studies.)

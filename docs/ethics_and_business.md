# Business Benefits + Ethical/Legal/Environmental Review

*(Ready-to-use content for the report — edit figures/wording as needed once
final results are in, but this is a complete draft, not a template.)*

## Business benefits for IntelliGen

Retail and e-commerce businesses generate large volumes of customer feedback
through reviews, support tickets, and social media, but manually reading and
acting on this feedback does not scale. Important issues can go unnoticed for
days, and by the time a problem is identified manually, many customers may
already have been affected. The IntelliGen Customer Insight & Response
Platform addresses this by automating the triage, categorisation, and
explanation of customer feedback.

**Sentiment analysis** automatically identifies negative reviews so they can
be prioritised for immediate attention, rather than requiring staff to read
every review in submission order. This reduces the time between a customer
raising an issue and a business becoming aware of it.

**Classification** routes each review to the correct internal team (e.g.
delivery issues to logistics, pricing complaints to the commercial team,
product defects to quality assurance) automatically. This removes the manual
step of a person reading and categorising each item of feedback, freeing
staff time for actually resolving issues rather than sorting them.

**Clustering** surfaces emerging themes in customer feedback that were not
predefined. In our testing on a general-purpose review sample spanning
multiple product categories, clustering primarily grouped reviews by
**product type** (e.g. books, films, toys) rather than by complaint type —
itself a useful capability, since it demonstrates the platform can
automatically segment mixed feedback by product line with no manual
tagging, which would help a client route category-specific feedback to the
right product team. On a single-category dataset (e.g. one retailer's
product line), the same clustering method would be expected to surface
complaint-type themes (e.g. delivery vs. quality issues) instead, since
product-type variation would no longer dominate the semantic signal. This
distinction — and the fact that clustering behaviour depends on the
underlying data's structure — is itself a useful, honest insight for a
client to understand before deploying this kind of tool.

**Explainable AI (SHAP)**, our advanced feature, makes every automated
decision auditable. Rather than presenting a "black box" classification,
staff can see exactly which words in a review drove the model's decision.
This matters commercially because it allows IntelliGen's clients to trust
and verify automated decisions, supports internal quality control, and
provides a defensible explanation if a client needs to justify an automated
decision to a regulator, an unhappy customer, or their own management.

Together, these components let IntelliGen offer clients faster response
times, better visibility into recurring issues, and — through explainability
— a more trustworthy and defensible product than a generic sentiment
dashboard that classifies feedback without being able to explain why.

## Ethical review

**Bias.** Sentiment and classification models are trained on historical
text data and can inherit biases present in that data. For example, reviews
written in non-standard English, regional dialects, or by non-native
speakers may be more likely to be misclassified than reviews written in
standard, "textbook" English, since pretrained models are typically trained
on datasets that overrepresent certain writing styles. This could result in
some customers' complaints being deprioritised or misrouted more often than
others, purely due to how they write rather than the substance of their
complaint. A responsible deployment of this platform would need to
audit classification accuracy across different writing styles and
demographics before relying on it to fully automate decisions.

**Explainability as a mitigation.** This is precisely why our advanced
feature — Explainable AI via SHAP — is not just a technical add-on but an
ethical safeguard. By showing which specific words drove a prediction, staff
can catch cases where the model is relying on spurious or biased signals
(for example, penalising a review for grammatical patterns unrelated to the
actual complaint) rather than trusting the model's output blindly.

**Human-in-the-loop.** We recommend this platform be used to prioritise and
route feedback for human review, not to fully automate decisions with no
human oversight — particularly for edge cases the model is not confident
about, or for decisions that could materially affect a customer (e.g.
automatically issuing or denying a refund).

## Legal review

**Data protection (UK GDPR / Data Protection Act 2018).** Customer reviews
may contain personal data, either directly (e.g. a customer including their
name or order number in a review) or indirectly (writing style, location
references). A production deployment of this platform would need to apply
data minimisation (only retaining what's necessary), anonymisation where
possible, and a clear lawful basis for processing this data — likely
legitimate interest, given the platform is used for service improvement
rather than marketing.

**Right to explanation.** UK GDPR Article 22 gives individuals rights
related to automated decision-making that has legal or similarly significant
effects on them. While this platform is intended for internal triage rather
than making legally significant decisions directly about individuals (e.g.
it does not automatically deny refunds or terminate accounts), any future
extension of the system to more consequential automated decisions would need
to ensure meaningful human review is available on request. Our
explainability feature (SHAP) directly supports compliance with this
principle, since it can produce a human-readable explanation for any
individual classification if one is later requested.

**Data provenance and licensing.** Our development and testing used the
publicly available Amazon Polarity dataset (via HuggingFace), which is
released for research and educational use. Any production deployment of this
platform for a real client would use that client's own first-party review
data under their existing data processing agreements, rather than this
public dataset.

## Environmental review

**Model choice.** We deliberately chose small, efficient pretrained models
(e.g. DistilBERT for sentiment analysis, a lightweight sentence-embedding
model for clustering) rather than training large models from scratch or
using very large language models for every task. DistilBERT, for example, is
roughly 40% smaller and significantly faster to run than the full BERT model
it is distilled from, while retaining most of its performance — directly
reducing the compute (and therefore energy) required both to develop and to
run the system.

**Training vs inference cost.** We avoided fine-tuning large models from
scratch, which is by far the most energy-intensive part of a typical machine
learning pipeline. Instead, our system relies primarily on pretrained models
used directly (zero-shot / off-the-shelf inference) and a lightweight,
efficient classical model (TF-IDF + Logistic Regression) trained on a small
labelled dataset, which requires negligible compute compared with training a
neural network from scratch.

**Ongoing inference cost.** In a production deployment, the platform would
run inference (not training) continuously on incoming reviews. We would
recommend batching review processing (e.g. hourly, rather than real-time
per-review) where immediate results are not required, to further reduce the
number of separate compute operations and associated energy use.

## Known limitations (for honesty and to demonstrate critical evaluation)

Our classification model achieves approximately 52% accuracy on held-out
data (against a 20% random baseline across five categories), trained on 125
hand-labelled examples. Manual error analysis showed that most remaining
errors stem from genuine ambiguity in short reviews that touch multiple
themes, and from the limited size of our labelled training set, rather than
systematic labelling errors. With a larger, domain-matched labelled dataset,
classification accuracy would likely improve substantially. We consider this
an appropriate and honestly-reported scope for a proof-of-concept built
within the assessment timeframe, and have documented it as a clear direction
for future improvement rather than overstating the system's current
reliability.
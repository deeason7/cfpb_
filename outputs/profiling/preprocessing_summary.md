 Deep Preprocessing Summary

**Records Retained:** 66,806

###  Major Preprocessing Steps:
- Dropped columns irrelevant for sentiment modeling- Text cleaning applied: Lowercased, special characters/HTML/URLs/emails/numbers removed, normalized spacing.
- Smart Weak Sentiment Labeling using:
  - TextBlob polarity & subjectivity
  - Keyword-based checks (extreme negative, negative, neutral indicators)
  - POS tagging for emphasis and negation adjustments
- Added Encoded/Binary Columns:
  - `consumer_disputed_binary`
  - `timely_response_binary`
  - `sentiment_encoded` (0: Neutral, 1: Negative, 2: Extreme Negative)
- Added Features:
  - `text_length` (word count of cleaned text)
  - `product_dispute_rate` (target encoding with smoothing)
  - `company_dispute_rate` (target encoding with smoothing)
  - `sentiment_timely_interaction` (sentiment * timely response)
  - `company_timely_interaction` (company dispute rate * timely response)
- Missing states filled with 'Unknown'.
- Carefully preserved complaint narratives with sufficient information.

---

Sample of a Cleaned Complaint Narrative:

> xxxx has claimed i owe them for xxxx years despite the proof of payment i sent them canceled check and their ownpaid invoice for they continue to insist i owe them and collection agencies are after me how can i stop this harassment for a bill i already paid four years ago

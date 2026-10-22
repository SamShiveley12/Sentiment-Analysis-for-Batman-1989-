# Batman (1989) Review Sentiment Analysis

## Overview
Sentiment analysis of 100 IMDb user reviews for Tim Burton's *Batman* (1989), 
predicting whether a review is negative, neutral, or positive from its text alone. 
Labels come from each reviewer's own star rating, bucketed consistently.

## Key Findings
- A TF-IDF + logistic regression model reached ~58% cross-validated accuracy 
  (±9%), well above the 35% baseline of always guessing the most common class.
- Neutral reviews were by far the hardest class: the model correctly identified 
  only 1 of 7 neutral reviews on the hold-out set. Many neutral reviews mix praise 
  and criticism, which pulls word-level features in both directions.
- With only 100 reviews, results vary noticeably between folds (45%–70%), so 
  these numbers show the approach works rather than serving as a precise benchmark.

## Data Source
User reviews collected manually from IMDb for Batman (1989): https://www.imdb.com/title/tt0096895/reviews/?ref_=ttrt_ov_ql_2

100 reviews total, balanced by class: 35 positive, 35 negative, 30 neutral.

| Bucket | IMDb star rating |
|---|---|
| Negative | 1–4 |
| Neutral | 5–6 |
| Positive | 7–10 |

## Tools Used
- PostgreSQL / pgAdmin (data storage)
- Python (pandas, scikit-learn, matplotlib, SQLAlchemy)
- Jupyter Notebook

## Methodology
1. Collected reviews by hand and bucketed them by star rating. I used a 3-class 
   scheme rather than binary so that mixed or lukewarm reviews weren't forced 
   into "positive" or "negative."
2. Generated SQL INSERT statements from a spreadsheet (escaping apostrophes 
   automatically) and loaded the reviews into PostgreSQL.
3. Checked class balance and review length by bucket before modeling.
4. Set a baseline with a "most frequent class" dummy model (35% accuracy).
5. Built a scikit-learn pipeline: TF-IDF vectorizer (single words and two-word 
   phrases) feeding a logistic regression classifier. Keeping both steps in one 
   pipeline means the vectorizer is fit only on training data in each fold, 
   which avoids data leakage.
6. Evaluated with 5-fold stratified cross-validation, plus a stratified 75/25 
   hold-out split for a confusion matrix and error analysis.

## Results
<img width="779" height="466" alt="Screenshot 2026-10-22 114150" src="https://github.com/user-attachments/assets/b44eee9a-a142-404b-8f8a-b78a250a156c" />



Cross-validated accuracy was 58% (±9%) against a 35% baseline. On the hold-out 
set, positive reviews were identified most reliably, while neutral reviews were 
mostly misclassified as negative or positive. Reading the misclassified reviews 
showed why: neutral reviews often contain real praise and real criticism in the 
same text, so there is no distinct "neutral vocabulary" for the model to learn.

The model's highest-weighted terms were mostly common function words rather than 
clear sentiment words, which suggests that at this sample size it is partly 
fitting noise.

## Limitations
- Small dataset (100 reviews, only 75 used for training in each split)
- A single film and a single source (IMDb)
- Labels come from star ratings, which don't always match the tone of the text

## What I'd Do Differently
With more reviews, I'd compare this baseline against a pretrained sentiment 
model, test removing stop words, and tune the regularization strength. I'd also 
try treating the classes as ordered (negative < neutral < positive) instead of 
independent categories.

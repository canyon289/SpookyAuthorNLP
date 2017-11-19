## Why am I doing this
Test that Naive Bayes model performance


## Implementation Notes
What I want to be able to do is take an input df X, pass it to a text transformer, and get

## Things I'm learning
If something doesn't look right, or it looks like there's a bug. There's probably a bug
Multinominal Naive Bayes is really sensitive to some features

## Outcome
Using TDIDF and Naive Bayes got me to 0.71464 on leaderboard
Using CountVectorizer and Naive Bayes got me to 0.48 on the leaderboard

## Lessons Learned
Multinomial Naive Bayes is super sensitive to features. For some reason adding count and string length really hurt the model
Switching from TDIDF to Count Vectorizer substantially increased my score from .7164 to .48




# Sarcasm Classifier.

## About:
This program runs a two-stage binary classifier *(sarcasm | not sarcasm)*. The first stage utilizes a custom Contrast approach
to classifying sarcasm utilizing the assumption that sarcasm can be recognized by looking for a Positive Verb Phrase or Positive
Predicate Expression closely followed by a Negative Situational Phrase. The second stage is simply a SVM with a RBF kernel. The final
classification result is decided upon either stage tagging the provided sentence as sarcastic; if neither stage does so then the result
is deemed "normal".
[Read More.](https://www.aclweb.org/anthology/D13-1066.pdf)

## Data:
The data used to train can be found here: [en-balanced](http://liks.fav.zcu.cz/sarcasm/).
The data loaded into the classifier came from two seperate csv files, one with sarcastic tweets and the other with normal tweets.
Both csv files used has two columns each. One with the Tweet ID from the above link and one with the corrosponding Tweets.

## Todo:
- Clean up the Contrast classifier functions.
- Rework the **expensive** tail recursion in the *_train_contrast* function.
- Add some comments.

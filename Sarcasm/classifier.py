import sys
from math import floor

import numpy as np
import pandas as pd
from nltk.tag import pos_tag
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

class Classify:
    def __init__(self, data):
        # Train/Test SVM Stage.
        self.X = data['tweets'].apply(lambda x: np.str_(x))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, data['labels'], test_size=0.20)

        self.svc, self.vectorizer = self._train_svm()
        self._test_svm()

        # Train/Test Contrast Stage.
        self.sarcasm_train = shuffle(data.copy(deep=True))
        index = len(self.sarcasm_train) - floor(len(self.sarcasm_train) * 0.20)     # Train and Test index.
        self.sarcasm_train, self.test = self.sarcasm_train[:index], self.sarcasm_train[index:].values.tolist()
        self.sarcasm_train = self.sarcasm_train[self.sarcasm_train['labels'] == 'sarcasm']
        self.sarcasm_train = self.sarcasm_train.values.tolist()
        self.positive_verb_phrases = {"love": 0}
        self.negative_situation_phrases = {}
        self.positive_pred_expressions = {}

        self._train_contrast(verb="love")

        # Filter out sparse data.
        self.positive_verb_phrases = {k: v for k, v in self.positive_verb_phrases.items() if v > 3}
        self.negative_situation_phrases = {k: v for k, v in self.negative_situation_phrases.items() if v > 3}
        self.positive_pred_expressions = {k: v for k, v in self.positive_pred_expressions.items() if v > 3}

        self._test_contrast()
        
    def _find_verb_phrase(self, pos):
        seed = None
        for tag in pos:
            if tag[1].startswith("VB"):
                seed = tag[0].lower()
                break
        return seed

    def _find_pred(self, pos):
        seed = None
        for tag in range(len(pos)):
            if pos[tag][1].startswith("JJ"):
                seed = pos[tag][0].lower()
                break
            elif pos[tag][1].startswith("RB"):
                if tag < len(pos) - 1 and pos[tag + 1][1].startswith("JJ"):
                    seed = pos[tag][0].lower() + ' ' + pos[tag + 1][0].lower()
                    break
            elif pos[tag][1].startswith("DT"):
                if (tag + 1 < len(pos) - 1
                    and pos[tag + 1][1].startswith("JJ")
                    and pos[tag + 2][1].startswith("N")):
                    seed = pos[tag][0].lower() + ' ' + pos[tag + 1][0].lower() + ' ' + pos[tag + 2][0].lower()
                    break
        return seed

    def _find_situation(self, pos):
        seed = None
        for tag in range(len(pos) - 1):
            if pos[tag][1].startswith("VB"):
               seed = pos[tag][0].lower() 
               break
            elif pos[tag][0].lower() == "to" or pos[tag][1].startswith("RB"):
                if tag < len(pos) - 1 and pos[tag + 1][1].startswith("VB"):
                    seed = pos[tag][0].lower() + ' ' + pos[tag + 1][0].lower()
                    break
        return seed

    def _train_contrast(self, verb=None, pred=None, situation=None):
        ''' Replace recursion with something more appropriate.

            Starts with seed "love" and hops back and forth between
            searching for Positive Verb Phrases/Positive Predicate Phrases
            to Negative Situation Phrases given the results that appeared in 
            the previous iteration as the new seed. 
            
            Ex: "love" -> "I love being alone." -> "being alone"
                "being alone" -> "It's great being alone!" -> "great"
                "great" -> "..." -> "..." '''

        count = 0
        seed = verb if verb != None else situation
        if seed == None: seed = pred
        while count < len(self.sarcasm_train):
            tokens = self.sarcasm_train[count][1].lower().split(' ')
            tokens = [token.strip('.') for token in tokens]
            tokens = [token.strip(',') for token in tokens]
            if seed not in tokens:
                count += 1
                continue
            del self.sarcasm_train[count]
            index = tokens.index(seed)
            if verb != None:
                self.positive_verb_phrases[seed] = self.positive_verb_phrases.get(seed, 0) + 1
                pos = pos_tag([token for token in tokens[index+1:] if token])
                next_seed = self._find_situation(pos)
                if next_seed == None: continue
                self.negative_situation_phrases[next_seed] = self.negative_situation_phrases.get(next_seed, 0) + 1
                self._train_contrast(situation=next_seed)
            elif situation != None:
                self.negative_situation_phrases[seed] = self.negative_situation_phrases.get(seed, 0) + 1
                pos = pos_tag([token for token in tokens[:index] if token])
                next_seed = self._find_verb_phrase(pos)
                if next_seed == None: 
                    next_seed = self._find_pred(pos)
                    if next_seed == None: continue
                    self.positive_pred_expressions[next_seed] = self.positive_pred_expressions.get(next_seed, 0) + 1
                    self._train_contrast(pred=next_seed)
                    continue
                self.positive_verb_phrases[next_seed] = self.positive_verb_phrases.get(next_seed, 0) + 1
                self._train_contrast(verb=next_seed)
            elif pred != None:
                self.positive_pred_expressions[seed] = self.positive_pred_expressions.get(seed, 0) + 1
                pos = pos_tag([token for token in tokens[index+1:] if token])
                next_seed = self._find_situation(pos)
                if next_seed == None: continue
                self.negative_situation_phrases[next_seed] = self.negative_situation_phrases.get(next_seed, 0) + 1
                self._train_contrast(situation=next_seed)
            else:
                return

    def _predict_contrast(self, tokens):
        ''' Finds the distance between Positive sentiments and Negative situations.
            If distance is less than or equal to 5 then the given message is 
            believed to be sarcastic. '''
        positive_index = np.array([token for token in range(len(tokens))
                            if tokens[token] in self.positive_verb_phrases or tokens[token] in self.positive_pred_expressions])
        negative_situation = np.array([token for token in range(len(tokens)) if tokens[token] in self.negative_situation_phrases])
        distance = np.array([positive_index - x for x in negative_situation])
        distance = np.absolute(distance.flatten())
        distance = np.nonzero(distance[distance <= 5])[0]
        if distance.size > 0:
            return "sarcasm"
        else:
            return "normal"

    def _test_contrast(self):
        true_pos = 0
        false_pos = 0
        false_neg = 0
        true_neg = 0
        for tweet in self.test:
            tweet, label = str(tweet[1]), tweet[2]
            tokens = tweet.lower().split(' ')
            tokens = [token.strip('.') for token in tokens]
            tokens = [token.strip(',') for token in tokens]
            predicted_label = self._predict_contrast(tokens)
            if label == "sarcasm" and predicted_label == "sarcasm":
                true_pos += 1
            elif label == "sarcasm" and predicted_label == "normal":
                false_neg += 1
            elif label == "normal" and predicted_label == "sarcsam":
                false_pos += 1
            else:
                true_neg += 1
        print(f"True Positives: {true_pos} \nFalse Positives: {false_pos} \nFalse Negatives: {false_neg} \nTrue Negatives: {true_neg}\n\n")
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f_score = 2 * ((precision * recall) / (precision + recall))
        print(f"Precision: {precision} \nRecall: {recall} \nF-score: {f_score}")
   
    def _train_svm(self):
        vectorizer = CountVectorizer()
        vectorizer.fit(self.X)
        X_train = vectorizer.transform(self.X_train)
        self.X_test = vectorizer.transform(self.X_test)
        svc = SVC(kernel='rbf')
        svc.fit(X_train, self.y_train)
        return svc, vectorizer
    
    def _test_svm(self):
        y_pred = self.svc.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))
    
    def predict(self, message):
        tokens = message.lower().split(' ')
        tokens = [token.strip('.') for token in tokens]
        tokens = [token.strip(',') for token in tokens]
        contrast = self._predict_contrast(tokens)
        if contrast == "sarcasm":
            return ["sarcasm"]

        df = pd.DataFrame([message])
        return self.svc.predict(self.vectorizer.transform(df[0]))

if __name__ == '__main__':
    sys.setrecursionlimit(10**6)    # Due to expensive recursion in _train_contrast function.
    header_list = ['tweet_id', 'tweets']
    
    # Reads and Combines both data files into one DataFrame.
    sarcasm_data = pd.read_csv("data/sarcastic_data.csv", delimiter='…',
                               engine='python', names=header_list, header=None)
    sarcasm_data['labels'] = ['sarcasm'] * len(sarcasm_data)
    normal_data = pd.read_csv("data/normal_data.csv", delimiter='…',
                               engine='python', names=header_list, header=None)
    normal_data['labels'] = ['normal'] * len(normal_data)
    data = pd.concat([sarcasm_data, normal_data], ignore_index=True, sort=False, names=header_list)

    c = Classify(data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

data_corpus = pd.read_csv("Data/Data/corpus.txt", sep="\t", header=None)
data_test = pd.read_csv("Data/Data/test.txt", sep="\t", header=None)

state_space = ["playing", "won", "lost"]
observation_space = ["correct", "incorrect"]
letters = "abcdefghijklmnopqrstuvwxyz"
incorrect_guesses = 0
encoder = LabelEncoder()
encoder.fit(list(letters))

#converting all words in corpus to numeric sequences
sequences = []
for word in data_corpus.values:
    cleaned_word = re.sub(r'[^a-z]', '', word[0].lower())  # keep only a–z
    if cleaned_word:  # skip empty strings
        seq = encoder.transform(list(cleaned_word))
        sequences.append(seq)
X = np.concatenate(sequences)
lengths = [len(seq) for seq in sequences]

startprob = np.array([1.0, 0.0, 0.0])

transmat = np.array([
    [0.85, 0.10, 0.05],
    [0.00, 1.00, 0.00],
    [0.00, 0.00, 1.00],
])

emissionprob = np.array([
    [0.30, 0.70],
    [1.00, 0.00],
    [0.00, 1.00],
])

model = hmm.MultinomialHMM(n_components=len(state_space), n_iter=100, tol=0.01, random_state=42)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob
model.fit(X.reshape(-1, 1), lengths)

test_sequences = [encoder.transform(list(word[0].lower())) for word in data_test.values]
X_test = np.concatenate(test_sequences)
lengths_test = [len(seq) for seq in test_sequences]

logprob, hidden_states = model.decode(X_test.reshape(-1, 1), algorithm = "viterbi")

predicted_letters = []
true_letters = []

for i in range(len(hidden_states) - 1):
    state = hidden_states[i]
    next_obs = X_test[i+1]
    emission_probs = model.emissionprob_[state]
    predicted_letter = np.argmax(emission_probs)
    predicted_letters.append(predicted_letter)
    true_letters.append(next_obs)

hmm_accuracy = accuracy_score(true_letters, predicted_letters)
print(hmm_accuracy)
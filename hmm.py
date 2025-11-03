import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder

data_corpus = pd.read_csv("Data/Data/corpus.txt", sep="\t", header=None)
data_test = pd.read_csv("Data/Data/test.txt", sep="\t", header=None)

state_space = ["playing", "won", "lost"]
observation_space = ["correct", "incorrect"]
letters = "abcdefghijklmnopqrstuvwxyz"
incorrect_guesses = 0
encoder = LabelEncoder()
encoder.fit(list(letters))

#converting all words in corpus to numeric sequences
sequences = [encoder.transform(list(word[0].lower())) for word in data_corpus.values]
X = np.concatenate(sequences)
lengths = [len(seq) for seq in sequences]

startprob = np.array([1.0, 0.0, 0.0])

transmat = np.array([
    [0.85, 0.10, 0.05],
    [0.00, 1.00, 0.00],
    [0.00, 0.00, 1.00],
])

emissionprob = np.array([
    [0.30, 0.70],   # Playing
    [1.00, 0.00],   # Won
    [0.00, 1.00],   # Lost
])

model = hmm.MultinomialHMM(n_components=len(state_space), n_iter=100, tol=0.01, random_state=42)
model.startprob_ = startprob
model.transmat_ = transmat
model.emmissionprob_ = emissionprob
model.fit(X.reshape(-1, 1), lengths)
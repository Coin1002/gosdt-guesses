"""
Owner: Ava Nederlander
Last Modified: November 4, 2023
ESE 530 - Project 1 

To Run in Terminal: 
pip3 install dist/gosdt-1.0.5-cp310-cp310-macosx_12_0_x86_64.whl
pip3 install attrs packaging editables pandas sklearn sortedcontainers gmpy2 matplotlib
python3 gosdt/ava_code.py
"""
## Import Libraries
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
import pandas as pd
import numpy as np
import time
import pathlib
from sklearn.ensemble import GradientBoostingClassifier
from model.threshold_guess import compute_thresholds
from model.gosdt import GOSDT
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import functools as ft

## Read the file
df = pd.read_csv("/Users/anederlander/Documents/GitHub/gosdt-guesses/experiments/datasets/test.csv")

## Process the data to ensure that X and y have the same number of samples
X = df.iloc[:, -1].values  # Data
y = df.iloc[:, :-1].values  # Labels

## Tokenize the text data
X_tokenize = [re.split(',', str(text)) for text in X]

## Create tagged documents for Doc2Vec
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_tokenize)]

## Train a Doc2Vec model
model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)

## Infer vectors for X
X_vectors = [model.infer_vector(doc) for doc in X_tokenize]

## Calculate the average decade label from the lists of years
def calculate_average_decade(years):
    if not years:
        return None
    return int(sum(years) / len(years) // 10 * 10)

## Calculate average decades for each data point
y_encoded = [calculate_average_decade(year) for year in y]

## GBDT parameters for threshold and lower bound guesses - adjust parameters to get better accuracy
n_est = 20
max_depth = 3

## Create unique column names for X
h = [f"Column_{i}" for i in range(1, len(X_vectors[0]) + 1)]

## Convert X vectors to a DataFrame with the new column names
X_encoded_df = pd.DataFrame(X_vectors, columns=h)

print("X:", X_encoded_df.shape)  # Data
print("y:", len(y_encoded))  # Labels

## Create and fit the GBDT classifier
clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
clf.fit(X_encoded_df, y_encoded)

## Guess thresholds
X_train, thresholds, header, threshold_guess_time = compute_thresholds(X_encoded_df, y_encoded, n_est, max_depth)
y_train = pd.DataFrame(y_encoded)

## Guess lower bound
start_time = time.perf_counter()
clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train.values.flatten())
warm_labels = clf.predict(X_train)

elapsed_time = time.perf_counter() - start_time

lb_time = elapsed_time

## Save the labels as a tmp file and return the path to it.
labelsdir = pathlib.Path('/tmp/warm_lb_labels')
labelsdir.mkdir(exist_ok=True, parents=True)

labelpath = labelsdir / 'warm_label.tmp'
labelpath = str(labelpath)
pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels", index=None)

## Train GOSDT model - adjust parameters to get better accuracy
config = {
    "regularization": 0.0002, #to adjust from 0.001-0.0002 
    "depth_budget": 5,
    "time_limit": 60,
    "warm_LB": True,
    "path_to_labels": labelpath,
    "similar_support": False,
}

## Run the model
model = GOSDT(config)
model.fit(X_train, y_train)
print("evaluate the model, extracting tree and scores", flush=True)

## Results
train_acc = model.score(X_train, y_train)
n_leaves = model.leaves()
n_nodes = model.nodes()
time = model.utime

print("Model training time: {}".format(time))
print("Training accuracy: {}".format(train_acc))
print("# of leaves: {}".format(n_leaves))
print(model.tree)
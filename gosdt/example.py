#must change line 19 in threshold_guess.py to log_loss instead of deviance
#inspired by code from stack exchange 
import re
import pandas as pd
import numpy as np
import time
import pathlib
from sklearn.ensemble import GradientBoostingClassifier
from model.threshold_guess import compute_thresholds
from model.gosdt import GOSDT
from sklearn.preprocessing import MultiLabelBinarizer
import functools as ft

#create lookup dictionary of all keywords in data file
txt = open(r"C:\Users\Owner\Downloads\gosdt-guesses-main\gosdt-guesses-main\experiments\datasets\keywords.txt")
def getUniqueWords(allWords) : #create array of just unique keywords
    uniqueWords = [] 
    for i in allWords:
        h = i.strip()
        if h != "\n":
            if not h in uniqueWords:
                uniqueWords.append(h)
    return uniqueWords
  
hi = getUniqueWords(txt) #run function
#print(hi)
my_list=[]
j = 0
for j in range(1, len(hi)+1): #create array of numbers
    my_list.append(j)
    j+1
#print(my_list)

lookup = dict(zip(hi, my_list)) #lookup dictionary where every keyword is assigned an integer value
lookup['nan'] = 0
lookup["'Green's function methods'"]=2553
print(len(lookup))
# read the file
fields = ['publication_year', 'keywords']
df = pd.read_csv(r"C:\Users\Owner\Downloads\gosdt-guesses-main\gosdt-guesses-main\experiments\datasets\test.csv")
#number = sum(lookup.get(word.lower(), 0) * freq for word, freq in collections.Counter(df).items())

X = df.iloc[:,:-1].values 
y = df.iloc[:,-1].values
h = df.columns[:-1]
#print(lookup)
for h in range (0, len(y)):
    words = re.split(',', str(y[h]))
    #ww = []
    #for j in words:
        #j.replace(" '", "'")
    #print(words)
    digits = [lookup[word] for word in words]
    y[h] = digits
#print(res)
print(X)
print(y)

y=MultiLabelBinarizer().fit_transform(X)
X=X.transpose()
frames = [df]

df = pd.concat(frames)
# GBDT parameters for threshold and lower bound guesses
n_est = 40
max_depth = 1

# guess thresholds
X = pd.DataFrame(X, columns=[h])
print("X:", X.shape)
print("y:",y.shape)
X_train, thresholds, header, threshold_guess_time = compute_thresholds(X, y, n_est, max_depth)
y_train = pd.DataFrame(y)

# guess lower bound
start_time = time.perf_counter()
clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train.values.flatten())
warm_labels = clf.predict(X_train)

elapsed_time = time.perf_counter() - start_time

lb_time = elapsed_time

# save the labels as a tmp file and return the path to it.
labelsdir = pathlib.Path('/tmp/warm_lb_labels')
labelsdir.mkdir(exist_ok=True, parents=True)

labelpath = labelsdir / 'warm_label.tmp'
labelpath = str(labelpath)
pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels",index=None)


# train GOSDT model
config = {
            "regularization": 0.001,
            "depth_budget": 5,
            "time_limit": 60,
            "warm_LB": True,
            "path_to_labels": labelpath,
            "similar_support": False,
        }

model = GOSDT(config)

model.fit(X_train, y_train)

print("evaluate the model, extracting tree and scores", flush=True)

# get the results
train_acc = model.score(X_train, y_train)
n_leaves = model.leaves()
n_nodes = model.nodes()
time = model.utime

print("Model training time: {}".format(time))
print("Training accuracy: {}".format(train_acc))
print("# of leaves: {}".format(n_leaves))
print(model.tree)



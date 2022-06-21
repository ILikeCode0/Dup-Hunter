import os
import re
import sys
import random
import copy
import numpy as np
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from numpy import array
from multiprocessing import Pool
from util import localfile
from repo_name import *
from comp import *
from git import *

from tqdm import tqdm

# ----------------INPUT & CONFIG------------------------------------
data_for = "GMN"

train_data_folder = 'data/train-dataset/'
test_data_folder = 'data/test-dataset/'
# path label type
dataset = [
    [train_data_folder + 'first_msr_pairs.txt', 1, 'train'],
    [train_data_folder + 'first_nondup.txt', 0, 'train']
]

for i in range(0, len(repos)):
    prefix = repo_br[i][0] + "/"
    dup_file = prefix + "dup_" + repo_br[i][0] + ".txt"
    nondup_file = prefix + "non_dup_" + repo_br[i][0] + ".txt"
    dataset.append([test_data_folder + dup_file, 1, "test"])
    dataset.append([test_data_folder + nondup_file, 0, "test"])

# for i in range(0 ,  len(dataset)):
#     print(dataset[i])


part_params = None
model_data_random_shuffle_flag = False
model_data_renew_flag = False

# ------------------------------------------------------------

# init NLP model
def init_model_with_pulls(pulls, save_id=None):
    t = [str(pull["title"]) for pull in pulls]
    b = []
    for pull in pulls:
        if pull["body"] and (len(pull["body"]) <= 2000):
            b.append(pull["body"])
    init_model_from_raw_docs(t + b, save_id)
    if code_sim_type == 'tfidf':
        c = []
        for pull in pulls:  # only added code
            try:
                if not check_large(pull):
                    p = copy.deepcopy(pull)
                    p["file_list"] = fetch_pr_info(p)
                    c.append(get_code_tokens(p)[0])
            except Exception as e:
                print('Error on get', pull['url'])

        init_code_model_from_tokens(c, save_id + '_code' if save_id is not None else None)


def init_model_with_repo(repo, save_id=None):
    print('init nlp model with %s data!' % repo)

    if save_id is None:
        save_id = repo.replace('/', '_') + '_allpr'
    else:
        save_id = repo.replace('/', '_') + '_' + save_id
    try:
        init_model_with_pulls([], save_id)
    except Exception as e:
        init_model_with_pulls(get_repo_info(repo, 'pull'), save_id)


# Calculate feature vector.
def get_sim(repo, num1, num2):
    if data_for == "GMN":
        return get_pull_info(repo, num1, num2)

def get_sim_wrap(args):
    return get_sim(*args)

def get_feature_vector(data, label, renew=False):
    print('Model Data Input=', data)
    if data_for == "GMN":
        default_path = data.replace('.txt', '') + '_pull_info'
        X_path, y_path = default_path + '_X.txt', default_path + '_y.txt'
        print("X_path", X_path, "y_path", y_path)

    if os.path.exists(X_path) and os.path.exists(y_path) and (not renew):
        print('warning: feature vector already exists!')
        X = localfile.get_file(X_path)
        y = localfile.get_file(y_path)
        return X, y

    X, y = [], []

    # run with all PR's info model
    p = {}
    with open(data) as f:
        all_pr = f.readlines()

    for l in all_pr:
        r, n1, n2 = l.strip().split()
        #
        # if 'msr_pairs' not in data:
        #     if check_large(get_pull(r, n1)) or check_large(get_pull(r, n2)):
        #         continue

        if r not in p:
            p[r] = []
        p[r].append((n1, n2, label))

    print('all=', len(all_pr))


    for r in p:
        init_model_with_repo(r)

    for r in p:
        print('Start running on', r)

        # init NLP model
        init_model_with_repo(r)

        print('pairs num=', len(p[r]))

        # run parallel
        for label in [0, 1]:
            pairs = []
            result = []
            for z in tqdm(p[r]):
                if z[2] == label:
                    pairs.append((r, z[0], z[1]))
                    temp_result = get_sim_wrap((r, z[0], z[1]))
                    result.append(temp_result)

            X.extend(result)
            y.extend([label for i in range(len(result))])


    # save to local
    localfile.write_to_file(X_path, X)
    localfile.write_to_file(y_path, y)
    return (X, y)


# Build classification model
def getData():
    def model_data_prepare(dataset):
        X_train, y_train = [], []
        X_test, y_test = [], []

        for s in dataset:
            new_X, new_y = get_feature_vector(s[0], s[1], model_data_renew_flag)
            if s[2] == 'train':
                X_train += new_X
                y_train += new_y
            elif s[2] == 'test':
                X_test += new_X
                y_test += new_y

        # random sorts
        def get_ran_shuffle(X, y, train_percent=0.5):
            X, y = shuffle(X, y, random_state=12345)
            num = len(X)
            train_num = int(num * train_percent)
            X_train, X_test = X[:train_num], X[train_num:]
            y_train, y_test = y[:train_num], y[train_num:]
            return (X_train, y_train, X_test, y_test)

        # ran shuffle with train set and test set
        if model_data_random_shuffle_flag:
            X_train, y_train, X_test, y_test = get_ran_shuffle(X_train + X_test, y_train + y_test)

        return (X_train, y_train, X_test, y_test)

    print('--------------------------')
    print('Loading Data')
    X_train, y_train, X_test, y_test = model_data_prepare(dataset)
    if part_params:
        def extract_col(a, c):
            for i in range(len(a)):
                t = []
                for j in range(len(c)):
                    if c[j] == 1:
                        t.append(a[i][j])
                a[i] = t

        extract_col(X_train, part_params)
        extract_col(X_test, part_params)
        print('extract=', part_params)

    print('--------------------------')
    print('Size of Dataset: training_set', len(X_train), 'testing_set', len(X_test), 'feature_length=', len(X_train[0]))


if __name__ == "__main__":
    getData()

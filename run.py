###############################################
# Used for run Dup-Hunter. Create by LiYulong #
###############################################
import subprocess
import threading
import multiprocessing
import argparse
import sys

# >>>>>>!!! 【 DO Not Change templates】!!!<<<<<<
template1 = 'python getGraph_repo_test.py {0} {1}'      # > test_rid_{0}_ws_{1}.txt'
template2 = 'python getGraph_train_data.py {0}'         # > train_ws_{0}.txt'
template3 = 'python gmn/train.py {0}'                   # > train_model_{0}.txt'
template4 = 'python gmn/getResult.py {0} {1} {2}'       # > test_model_{0}_{1}_{2}.txt'

#    Some mix options
# 1. For build graph, -b -i -w are needed
# 2. For train,       -t -f    are needed
# 3. For test,        -r -f -m are needed
parser = argparse.ArgumentParser()
# Just for test graph, means how many repositories will be built.
parser.add_argument('-i', '--repoid', default=14)
parser.add_argument('-w', '--windowsize', default=9)
parser.add_argument('-f', '--filename',   default="")
parser.add_argument('-m', '--modelname',  default="")
parser.add_argument('-b', '--buildgraph', default=False)
parser.add_argument('-t', '--train',      default=False)
parser.add_argument('-r', '--test',       default=False)
args = parser.parse_args()

# changed for build graph for repo 0 - REPO_ID
REPO_ID = int(args.repoid)
# window_size
ws = int(args.windowsize)
# filename for train
filename = args.filename
# modelname for test
modelname = args.modelname
# get Graph ?
buildgraph = args.buildgraph
# train ?
train = args.train
# test ?
test = args.test

threads = []
semaphore = threading.Semaphore(multiprocessing.cpu_count())

def getTestGraph(rw, semaphore):
    semaphore.acquire()
    print(template1.format(*rw))
    subprocess.call(template1.format(*rw), shell=True)
    semaphore.release()

def getTrainGraph(w, semaphore):
    semaphore.acquire()
    print(template2.format(w))
    subprocess.call(template2.format(w), shell=True)
    semaphore.release()

def trainModel(fn, semaphore):
    semaphore.acquire()
    print(template3.format(fn))
    subprocess.call(template3.format(fn), shell=True)
    semaphore.release()

def testModel(fn, semaphore):
    semaphore.acquire()
    print(template4.format(*fn))
    subprocess.call(template4.format(*fn), shell=True)
    semaphore.release()

# Get repo test graph
if buildgraph:
    for id in range(0, REPO_ID):
        threads.append(threading.Thread(target=getTestGraph, args=((id, ws), semaphore)))

# Get train data
if buildgraph:
    threads.append(threading.Thread(target=getTrainGraph, args=((ws), semaphore)))

# For train
if train:
    if filename == "":
        print("train with filename ?")
        exit(0)
    threads.append(threading.Thread(target=trainModel, args=((filename), semaphore)))

# For test
if test:
    if filename == "" or modelname == "":
        print("test with filename or modelname?")
        exit(0)
    threads.append(threading.Thread(target=testModel, args=((filename, modelname, REPO_ID), semaphore)))

[t.start() for t in threads]
[t.join() for t in threads]
print('finished', len(threads), 'experiments')

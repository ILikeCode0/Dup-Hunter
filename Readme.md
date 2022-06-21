# Dup-Hunter

## Dup-Hunter: Detecting Duplicate Contributions in Fork-Based Development

Python library dependencies:
+ tensorflow -v : 1.13.1
+ numpy -v : 1.18.5
+ nltk -v : 3.4.5
+ flask -v : 1.1.1
+ GitHub-Flask -v : 3.2.0
+ gensim -v : 3.8.3
+ scipy -v : 1.4.1 
+ others: sklearn, bs4,

---

Dataset:

[dupPR]: Reference paper: Yu, Yue, et al. "A dataset of duplicate pull-requests in github." Proceedings of the 15th International Conference on Mining Software Repositories. ACM, 2018. (link: http://yuyue.github.io/res/paper/DupPR-msr2017.pdf)
<including: 2323 Duplicate PR pairs in 26 repos>

---
#### **If you want to use our model quickly, three steps need to be done.**

First, run `python run.py -b True -i 14 -w 9` to get all data graph (training data graphs and testing data graphs);

Second, run `python run.py -t True -f remove_9` to train the model;

Third, run `python run.py -r True -f remove_9 -m 0 -i 14` to get the result.

```
# -i/--repoid means how many testing data sets will be processed.
parser.add_argument('-i', '--repoid',     default=14)
# -w/--windowsize means the window size of sliding window.
parser.add_argument('-w', '--windowsize', default=9)
# -f/--filename means the filename of training or testing graph file.
parser.add_argument('-f', '--filename',   default="")
# -m/--modelname means which model used for testing.
parser.add_argument('-m', '--modelname',  default="")
```
---
#### **More clearly, the following 5 steps are required**
+ getData.py
  
    `python getData.py`
    
    It will generate `xxx_pull_info_X.txt` and `xxx_pull_info_y.txt` files which include all data needed to build graph. 
    
+ getGraph_train_data.py.py
    
    `python getGraph_train_data.py N`
    
    It will take `xxx_pull_info_X.txt` and `xxx_pull_info_y.txt` files from `data/train-dataset/` directory to build the training data graph. The N1 indicates the sliding window size used to build the graph, then two graph files will be generated in the same directory, as follows:
      
    `remove_XXX_N.train_graphs`, `remove_XXX_N.train_val_graphs`
    
    >Note: XXX can be ` `, `title`, `body`. Where N represents the sliding window size.

+ getGraph_repo_test.py
   
    `python getGraph_repo_test_data.py N1 N2`
    
    It will take `xxx_pull_info_X.txt` and `xxx_pull_info_y.txt` files from each repository directory to build the testing data graph. The N1 indicates the sliding window size used to build the graph. The N2 indicates the graph building of which repository. Then `remove_XXX_N.test_graphs` will be generated in the each repository directory.

+ gmn/train.py  

    `python gmn/train.py filename`
    
    It will take `filename.train_graphs` and `filename.train_val_graphs` in the `data/train-dataset/` directory to train the model.
     
+ gmn/getResult.py

     `python gmn/getResult.py filename modelname repoid`
    
    It will take `filename.test_graphs` in each repository directory to test the model in the `modelname` directory. Repoid indicates how many testing data sets will be processed.
    
---

nlp.py: Natural Language Processing model for calculating the text similarity.


```
m = Model(texts)
text_sim = query_sim_tfidf(tokens1, tokens2)
``` 


comp.py: Calculate the similarity for feature extraction.

``` 
# Set up the params of compare (different metrics).
# Check for init NLP model.
feature_vector = get_pr_sim_vector(pull1, pull2)
```


git.py: About GitHub API setting and fetching.

``` python
get_repo_info('repositories',
              'fork' / 'pull' / 'issue' / 'commit' / 'branch',
              renew_flag)

get_pull(repo, num, renew)
get_pull_commit(pull, renew)
fetch_file_list(pull, renew)
get_another_pull(pull, renew)
check_too_big(pull)
```


fetch_raw_diff.py: Get data from API, parse the raw diff.

```
parse_diff(file_name, diff) # parse raw diff
fetch_raw_diff(url) # parse raw diff from GitHub API
```



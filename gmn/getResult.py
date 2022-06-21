import tensorflow as tf
import sys
import pickle as pkl
from utils import *
from models import *
from layers import *
import collections
import time
import random
import copy
from tqdm import tqdm
import numpy as np
import os
from repo_name import *

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GraphData = collections.namedtuple(
    "GraphData",
    ["from_idx", "to_idx", "node_features", "edge_features", "graph_idx", "n_graphs"],
)

f_result = open("result.txt", "w", encoding="UTF-8")

fileName = ""
modelName = ""
REPO_ID = 0
test_graphs = []

def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = tf.cast(tf.equal(x > 0, y > 0), dtype=tf.float32)
    return tf.reduce_mean(match, axis=1)

def compute_similarity(config, x, y):
    """Compute the distance between x and y vectors.

  The distance will be computed based on the training loss type.

  Args:
    config: a config dict.
    x: [n_examples, feature_dim] float tensor.
    y: [n_examples, feature_dim] float tensor.

  Returns:
    dist: [n_examples] float tensor.

  Raises:
    ValueError: if loss type is not supported.
  """
    if config["training"]["loss"] == "margin":
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif config["training"]["loss"] == "hamming":
        return exact_hamming_similarity(x, y)
    else:
        raise ValueError("Unknown loss type %s" % config["training"]["loss"])

def auc(scores, labels, **auc_args):
    """Compute the AUC for pair classification.

  See `tf.metrics.auc` for more details about this metric.

  Args:
    scores: [n_examples] float.  Higher scores mean higher preference of being
      assigned the label of +1.
    labels: [n_examples] int.  Labels are either +1 or -1.
    **auc_args: other arguments that can be used by `tf.metrics.auc`.

  Returns:
    auc: the area under the ROC curve.
  """
    scores_max = tf.reduce_max(scores)
    scores_min = tf.reduce_min(scores)
    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2
    # The following code should be used according to the tensorflow official
    # documentation:
    # value, _ = tf.metrics.auc(labels, scores, **auc_args)

    # However `tf.metrics.auc` is currently (as of July 23, 2019) buggy so we have
    # to use the following:
    _, value = tf.metrics.auc(labels, scores, **auc_args)
    return value

"""Build the model"""
def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

  Args:
    tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
      multiple of `n_splits`.
    n_splits: int, number of splits to split the tensor into.

  Returns:
    splits: a list of `n_splits` tensors.  The first split is [tensor[0],
      tensor[n_splits], tensor[n_splits * 2], ...], the second split is
      [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
  """
    feature_dim = tensor.shape.as_list()[-1]
    # feature dim must be known, otherwise you can provide that as an input
    assert isinstance(feature_dim, int)
    tensor = tf.reshape(tensor, [-1, feature_dim * n_splits])
    return tf.split(tensor, n_splits, axis=-1)

def build_placeholders(node_feature_dim, edge_feature_dim):
    """Build the placeholders needed for the model.

  Args:
    node_feature_dim: int.
    edge_feature_dim: int.

  Returns:
    placeholders: a placeholder name -> placeholder tensor dict.
  """
    # `n_graphs` must be specified as an integer, as `tf.dynamic_partition`
    # requires so.
    return {
        "node_features": tf.placeholder(tf.float32, [None, node_feature_dim]),
        "edge_features": tf.placeholder(tf.float32, [None, edge_feature_dim]),
        "from_idx": tf.placeholder(tf.int32, [None]),
        "to_idx": tf.placeholder(tf.int32, [None]),
        "graph_idx": tf.placeholder(tf.int32, [None]),
        # only used for pairwise training and evaluation
        "labels": tf.placeholder(tf.int32, [None]),
    }

def fill_feed_dict(placeholders, batch):
    """Create a feed dict for the given batch of data.

  Args:
    placeholders: a dict of placeholders.
    batch: a batch of data, should be either a single `GraphData` instance for
      triplet training, or a tuple of (graphs, labels) for pairwise training.

  Returns:
    feed_dict: a feed_dict that can be used in a session run call.
  """
    if isinstance(batch, GraphData):
        graphs = batch
        labels = None
    else:
        graphs, labels = batch
    #labels = [-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1]
    # print(graphs)
    # print(labels)

    feed_dict = {
        placeholders["node_features"]: graphs.node_features,
        placeholders["edge_features"]: graphs.edge_features,
        placeholders["from_idx"]: graphs.from_idx,
        placeholders["to_idx"]: graphs.to_idx,
        placeholders["graph_idx"]: graphs.graph_idx,
    }
    if labels is not None:
        feed_dict[placeholders["labels"]] = labels
    return feed_dict


def build_model(config, node_feature_dim, edge_feature_dim):
    encoder = GraphEncoder(**config["encoder"])
    aggregator = GraphAggregator(**config["aggregator"])
    if config["model_type"] == "embedding":
        model = GraphEmbeddingNet(encoder, aggregator, **config["graph_embedding_net"])
    elif config["model_type"] == "matching":
        model = GraphMatchingNet(encoder, aggregator, **config["graph_matching_net"])
    else:
        raise ValueError("Unknown model type: %s" % config["model_type"])

    training_n_graphs_in_batch = config["training"]["batch_size"]
    if config["training"]["mode"] == "pair":
        training_n_graphs_in_batch *= 2
    elif config["training"]["mode"] == "triplet":
        training_n_graphs_in_batch *= 4
    else:
        raise ValueError("Unknown training mode: %s" % config["training"]["mode"])

    placeholders = build_placeholders(node_feature_dim, edge_feature_dim)

    # training
    model_inputs = placeholders.copy()
    del model_inputs["labels"]
    model_inputs["n_graphs"] = training_n_graphs_in_batch
    graph_vectors = model(**model_inputs)

    if config["training"]["mode"] == "pair":
        x, y = reshape_and_split_tensor(graph_vectors, 2)
        labels = placeholders["labels"]
        sim = compute_similarity(config, x, y)

    return (
        {
            "metrics": {
                "training": {
                    "x": x,
                    "y": y,
                    "sim": sim,
                    "label": labels,
                }
            }
        },
        placeholders,
        model,
    )

def get_test_graphs():
    for i in range(0, REPO_ID):
        prefix = "data/test-dataset/" + repo_br[i][0] + "/"
        filename = prefix + fileName + ".test_graphs"
        test_graphs.append(filename)
    print(test_graphs)
    # print(len(test_graphs))

dup_in_repo = [199, 152, 112, 104, 103, 74, 62, 61, 57, 55, 52, 46, 42, 30]

"""Main run process"""
if __name__ == "__main__":

    if len(sys.argv) == 4:
        fileName = sys.argv[1]
        modelName = sys.argv[2]
        REPO_ID = int(sys.argv[3])
    get_test_graphs()
    config = get_default_config()
    config["training"]["n_training_steps"] = 2
    tf.reset_default_graph()

    # Set random seeds
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)

    batch_size = config["training"]["batch_size"]
    print("batch_size", batch_size)
    tensors, placeholders, model = build_model(config, 300, 1)
    accumulated_metrics = collections.defaultdict(list)
    t_start = time.time()
    init_ops = (tf.global_variables_initializer(), tf.local_variables_initializer())

    # If we already have a session instance, close it and start a new one
    if "sess" in globals():
        sess.close()
    saver = tf.train.Saver()
    cfg = tf.ConfigProto()
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.1
    model_path = "Model/" + modelName + "/"
    with tf.Session(config=cfg) as sess:
        if os.path.exists(model_path + 'checkpoint'):
            print("yes")
            saver.restore(sess, model_path + 'lyl.GMN-9')
        else:
            print("no")
            init = tf.global_variables_initializer()
            sess.run(init)
        test_graph = []

        aa = 0

        p1_all, p2_all, p3_all, p4_all, p5_all = 0.0, 0.0, 0.0, 0.0, 0.0
        r1_all, r2_all, r3_all, r4_all, r5_all = 0.0, 0.0, 0.0, 0.0, 0.0
        map1_all, map2_all, map3_all, map4_all, map5_all = 0.0, 0.0, 0.0, 0.0, 0.0
        mrr_all = 0.0
        for graph in test_graphs:
            print(aa)
            aa += 1

            with open(graph, 'rb') as f:
                test_graph = pkl.load(f)

            size = len(test_graph)
            rate1, rate2, rate3, rate4, rate5 = int(size * 0.01), int(size * 0.02), int(size * 0.03), int(size * 0.04), int(size * 0.05)
            dup_all = dup_in_repo[aa - 1]
            print("size", size, "dup_all", dup_all,
                  "rate1", rate1, "rate2", rate2, "rate3", rate3, "rate4", rate4, "rate5", rate5)

            sim_list = {}
            label_list = {}
            for i in tqdm(range(len(test_graph))):
                batch = test_graph[i]
                # print(batch)
                sim, label = sess.run(
                    [tensors["metrics"]["training"]["sim"], tensors["metrics"]["training"]["label"]],
                    feed_dict=fill_feed_dict(placeholders, batch)
                )

                sim_list[i] = 1 / (1 + ((-sim[0]) ** 0.5))
                label_list[i] = label

                # if sim_list[i] > 0.7 and label == -1:
                #     print("find one in ", graph, ", index = ", i, ", sim = ", sim_list[i])
                # elif sim_list[i] < 0.3 and label == 1:
                #     print("find two in ", graph, ", index = ", i, ", sim = ", sim_list[i])
            print(len(sim_list))

            sim_list_sorted = [(x, y) for x, y in sorted(sim_list.items(), key=lambda x: x[1], reverse=True)][:rate5]

            label_list_sorted = []
            for x, y in sim_list_sorted:
                # print(label_list[x])
                label_list_sorted.extend(label_list[x])
            print(label_list_sorted)
            f_0, f_1, f_5, f_10, f_20 = 0, 0, 0, 0, 0
            mrr = 0
            map_0, map_1, map_5, map_10, map_20 = 0.0, 0.0, 0.0, 0.0, 0.0
            notFind = True
            for i in range(0, len(label_list_sorted)):
                if label_list_sorted[i] == 1:
                    if notFind:
                        mrr = 1 / (i + 1)
                        notFind = False

                    if i < rate1:  # 1%
                        f_0 += 1
                        map_0 += f_0 / (i + 1)
                    if i < rate2:
                        f_1 += 1
                        map_1 += f_1 / (i + 1)
                    if i < rate3:
                        f_5 += 1
                        map_5 += f_5 / (i + 1)
                    if i < rate4:
                        f_10 += 1
                        map_10 += f_10 / (i + 1)
                    if i < rate5:
                        f_20 += 1
                        map_20 += f_20 / (i + 1)
            p1, p2, p3, p4, p5 = f_0 / rate1, f_1 / rate2, f_5 / rate3, f_10 / rate4, f_20 / rate5
            r1, r2, r3, r4, r5 = f_0 / dup_in_repo[aa - 1], f_1 / dup_in_repo[aa - 1], f_5 / dup_in_repo[aa - 1], \
                                 f_10 / dup_in_repo[aa - 1], f_20 / dup_in_repo[aa - 1]
            map1, map2, map3, map4, map5 = map_0 / rate1, map_1 / rate2, map_5 / rate3, map_10 / rate4, map_20 / rate5

            p1_all, p2_all, p3_all, p4_all, p5_all = p1_all + p1, p2_all + p2, p3_all + p3, p4_all + p4, p5_all + p5
            r1_all, r2_all, r3_all, r4_all, r5_all = r1_all + r1, r2_all + r2, r3_all + r3, r4_all + r4, r5_all + r5
            map1_all, map2_all, map3_all, map4_all, map5_all = map1_all + map1, map2_all + map2, map3_all + map3, map4_all + map4, map5_all + map5
            mrr_all += mrr

            print("%s: \n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n" %
                  (graph, p1, p2, p3, p4, p5, r1, r2, r3, r4, r5, mrr, map1, map2, map3, map4, map5))

            print("%s: \n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n" %
                  (graph, p1, p2, p3, p4, p5, r1, r2, r3, r4, r5, mrr, map1, map2, map3, map4, map5), file=f_result)

        p1_all, p2_all, p3_all, p4_all, p5_all = \
                        p1_all / REPO_ID, p2_all / REPO_ID, p3_all / REPO_ID, p4_all / REPO_ID, p5_all / REPO_ID
        r1_all, r2_all, r3_all, r4_all, r5_all = \
                        r1_all / REPO_ID, r2_all / REPO_ID, r3_all / REPO_ID, r4_all / REPO_ID, r5_all / REPO_ID
        map1_all, map2_all, map3_all, map4_all, map5_all = map1_all / REPO_ID, map2_all / REPO_ID, map3_all / REPO_ID, map4_all / REPO_ID, map5_all / REPO_ID

        mrr_all = mrr_all / REPO_ID
        print("%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n" %
                                                            (p1_all, p2_all, p3_all, p4_all, p5_all,
                                                            r1_all, r2_all, r3_all, r4_all, r5_all,
                                                            mrr_all, map1_all, map2_all, map3_all, map4_all, map5_all))
        print("%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n" %
                                                            (p1_all, p2_all, p3_all, p4_all, p5_all,
                                                            r1_all, r2_all, r3_all, r4_all, r5_all,
                                                            mrr_all, map1_all, map2_all, map3_all, map4_all, map5_all), file=f_result)

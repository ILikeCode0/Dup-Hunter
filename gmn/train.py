import sys
import pickle as pkl
from utils import *
from models import *
from layers import *
import collections
import time
import random
import copy

import tensorflow.compat.v1 as tf
import numpy as np

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] = "0， 1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
GraphData = collections.namedtuple(
    "GraphData",
    ["from_idx", "to_idx", "node_features", "edge_features", "graph_idx", "n_graphs"],
)

f_train = open("gmn/result/train.txt","w",encoding="UTF-8")
f_valid = open("gmn/result/valid.txt","w",encoding="UTF-8")
f_test = open("gmn/result/test.txt","w",encoding="UTF-8")
"""Evaluation"""


def batch_graph(data, fro, batch):
    '''
    from_ids = data[for][0][0]
    from_ida = g1[0][0]
    print("from_ida", from_ida.shape)
    from_idb = g2[0][0] + g1[0][4].shape[0]
    print("from_idb", from_idb.shape)
    from_ids = np.hstack((from_ida, from_idb))
    print("from_ids", from_ids)
    print(from_ids.shape)
    '''
    # print(data[fro])
    from_ids = data[fro][0][0]
    to_ids = data[fro][0][1]
    num = data[fro][0][4].shape[0]
    # print(num)
    for i in range(fro + 1, fro + batch):
        from_idx = data[i][0][0] + num
        to_idx = data[i][0][1] + num
        num = num + data[i][0][4].shape[0]
        from_ids = np.hstack((from_ids, from_idx))
        to_ids = np.hstack((to_ids, to_idx))
    # print(from_ids)
    # print(to_ids)

    '''
    graph_ida = g1[0][4]
    print(graph_ida)
    graph_idb = g2[0][4] + 2
    print(graph_idb)
    graph_ids = np.hstack((graph_ida, graph_idb))
    print(graph_ids)
    '''
    graph_ids = data[fro][0][4]
    for i in range(fro+1, fro + batch):
        graph_idx = data[i][0][4] + 2*(i-fro)
        graph_ids = np.hstack((graph_ids, graph_idx))
    graph_ns = 2 * batch
    labels = data[fro][1]
    for i in range(fro+1, fro + batch):
        labels = np.hstack((labels, data[i][1]))
    # print(labels)

    # node_features
    node_fs = data[fro][0][2]
    for i in range(fro + 1, fro + batch):
        node_fx = data[i][0][2]
        node_fs = np.append(node_fs, node_fx, axis=0)
    # edge_features
    edge_fs = data[fro][0][3]
    empty = False
    if edge_fs.shape[0] == 0 or edge_fs.shape[1] == 0:
        empty = True
    for i in range(fro + 1, fro + batch):
        edge_fx = data[i][0][3]
        if edge_fx.shape[0] == 0 or edge_fx.shape[1] == 0:
            continue
        if empty:
            edge_fs = edge_fx
            empty = False
            continue
        edge_fs = np.append(edge_fs, edge_fx, axis=0)
    graph = GraphData(
            from_idx=np.array(from_ids),
            to_idx=np.array(to_ids),
            node_features=np.array(node_fs, dtype="float32"),
            edge_features=np.array(edge_fs, dtype="float32"),
            graph_idx=np.array(graph_ids), n_graphs=graph_ns)
    return graph, labels


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

def build_model(config, node_feature_dim, edge_feature_dim):
    """Create model for training and evaluation.

  Args:
    config: a dictionary of configs, like the one created by the
      `get_default_config` function.
    node_feature_dim: int, dimensionality of node features.
    edge_feature_dim: int, dimensionality of edge features.

  Returns:
    tensors: a (potentially nested) name => tensor dict.
    placeholders: a (potentially nested) name => tensor dict.
    model: a GraphEmbeddingNet or GraphMatchingNet instance.

  Raises:
    ValueError: if the specified model or training settings are not supported.
  """
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

        loss = pairwise_loss(
            x,
            y,
            placeholders["labels"],
            loss_type=config["training"]["loss"],
            margin=config["training"]["margin"],
        )

        # optionally monitor the similarity between positive and negative pairs
        is_pos = tf.cast(tf.equal(placeholders["labels"], 1), tf.float32)
        is_neg = 1 - is_pos
        n_pos = tf.reduce_sum(is_pos)
        n_neg = tf.reduce_sum(is_neg)
        sim = compute_similarity(config, x, y)
        sim_pos = tf.reduce_sum(sim * is_pos) / (n_pos + 1e-8)
        sim_neg = tf.reduce_sum(sim * is_neg) / (n_neg + 1e-8)
    else:
        x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
        loss = triplet_loss(
            x_1,
            y,
            x_2,
            z,
            loss_type=config["training"]["loss"],
            margin=config["training"]["margin"],
        )

        sim_pos = tf.reduce_mean(compute_similarity(config, x_1, y))
        sim_neg = tf.reduce_mean(compute_similarity(config, x_2, z))

    graph_vec_scale = tf.reduce_mean(graph_vectors ** 2)
    if config["training"]["graph_vec_regularizer_weight"] > 0:
        loss += (
            config["training"]["graph_vec_regularizer_weight"] * 0.5 * graph_vec_scale
        )

    # monitor scale of the parameters and gradients, these are typically helpful
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config["training"]["learning_rate"]
    )
    grads_and_params = optimizer.compute_gradients(loss)
    grads, params = zip(*grads_and_params)
    grads, _ = tf.clip_by_global_norm(grads, config["training"]["clip_value"])
    train_step = optimizer.apply_gradients(zip(grads, params))

    grad_scale = tf.global_norm(grads)
    param_scale = tf.global_norm(params)

    # evaluation
    model_inputs["n_graphs"] = config["evaluation"]["batch_size"] * 2
    eval_pairs = model(**model_inputs)
    x, y = reshape_and_split_tensor(eval_pairs, 2)
    similarity = compute_similarity(config, x, y)
    pair_auc = auc(similarity, placeholders["labels"])

    model_inputs["n_graphs"] = config["evaluation"]["batch_size"] * 4
    eval_triplets = model(**model_inputs)
    x_1, y, x_2, z = reshape_and_split_tensor(eval_triplets, 4)
    sim_1 = compute_similarity(config, x_1, y)
    sim_2 = compute_similarity(config, x_2, z)
    triplet_acc = tf.reduce_mean(tf.cast(sim_1 > sim_2, dtype=tf.float32))

    return (
        {
            "train_step": train_step,
            "metrics": {
                "training": {
                    "x": x,
                    "y": y,
                    "sim": similarity,
                    "loss": loss,
                    "grad_scale": grad_scale,
                    "param_scale": param_scale,
                    "graph_vec_scale": graph_vec_scale,
                    "sim_pos": sim_pos,
                    "sim_neg": sim_neg,
                    "sim_diff": sim_pos - sim_neg,
                    "labels":  placeholders["labels"],
                },
                "validation": {
                    "pair_auc": pair_auc,
                    "triplet_acc": triplet_acc,
                },
            },
        },
        placeholders,
        model,
    )

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

def evaluate(sess, eval_metrics, placeholders, validation_set, batch_size ,nums):
    """Evaluate model performance on the given validation set.

  Args:
    sess: a `tf.Session` instance used to run the computation.
    eval_metrics: a dict containing two tensors 'pair_auc' and 'triplet_acc'.
    placeholders: a placeholder dict.
    validation_set: a `GraphSimilarityDataset` instance, calling `pairs` and
      `triplets` functions with `batch_size` creates iterators over a finite
      sequence of batches to evaluate on.
    batch_size: number of batches to use for each session run call.

  Returns:
    metrics: a dict of metric name => value mapping.
  """
    accumulated_pair_auc = []

    for i in range(nums):
        batch = batch_graph(validation_set, i * batch_size, batch_size)
        feed_dict = fill_feed_dict(placeholders, batch)
        pair_auc = sess.run(eval_metrics["pair_auc"], feed_dict=feed_dict)
        accumulated_pair_auc.append(pair_auc)

    # accumulated_triplet_acc = []
    # for batch in validation_set.triplets(batch_size):
    #     feed_dict = fill_feed_dict(placeholders, batch)
    #     triplet_acc = sess.run(eval_metrics["triplet_acc"], feed_dict=feed_dict)
    #     accumulated_triplet_acc.append(triplet_acc)

    return {
        "pair_auc": np.mean(accumulated_pair_auc),
        # "triplet_acc": np.mean(accumulated_triplet_acc),
    }

def load_data(filename):
    names = ['train_graphs', 'train_val_graphs']
    objects = []
    for i in range(len(names)):
        with open("data/train-dataset/{}.{}".format(filename, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    # train_graphs,train_val_graphs,test_graphs = tuple(objects)
    train_graphs, train_val_graphs = tuple(objects)
    # return train_graphs, train_val_graphs, test_graphs
    return train_graphs, train_val_graphs


"""Main run process"""

# get filename
fn = ""
try:
    fn = sys.argv[1]
except:
    print("no filename provide")
    exit(0)



config = get_default_config()
config["training"]["n_training_steps"] = 100
tf.reset_default_graph()

# Set random seeds
seed = config["seed"]
random.seed(seed)
np.random.seed(seed + 1)
tf.set_random_seed(seed + 2)


# train_graphs,train_val_graphs, test_graphs = load_data(fn)
train_graphs, train_val_graphs = load_data(fn)
print("len(train_graphs)", len(train_graphs))
print("len(train_val_graphs)", len(train_val_graphs))
# print("len(test_graphs)", len(test_graphs))

batch_size = config["training"]["batch_size"]
print("batch_size", batch_size)

tensors, placeholders, model = build_model(config, 300, 1)
accumulated_metrics = collections.defaultdict(list)
t_start = time.time()
init_ops = (tf.global_variables_initializer(), tf.local_variables_initializer())

# If we already have a session instance, close it and start a new one
if "sess" in globals():
    sess.close()

# We will need to keep this session instance around for e.g. visualization.
# But you should probably wrap it in a `with tf.Session() sess:` context if you
# want to use the code elsewhere.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
cfg = tf.ConfigProto()
cfg.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=cfg)
# sess = tf.Session()
sess.run(init_ops)
saver = tf.train.Saver()

if os.path.exists('gmn/Model/checkpoint'):
    saver.restore(sess, 'gmn/Model/lyl.GMN-107')
else:
    init = tf.global_variables_initializer()
    sess.run(init)

for i_iter in range(config["training"]["n_training_steps"]):
    j = 0
    print("epoch", i_iter + 1)
    if i_iter % 5 == 0:
        saver.save(sess, "gmn/Model/%d/lyl.GMN" % (i_iter), global_step = 9)
    for i in range(0, len(train_graphs), batch_size):
        j = j+1
        end = i + batch_size
        if end > len(train_graphs):
            break
        batch = batch_graph(train_graphs, i, batch_size)
        # print(batch)
        _, train_metrics, sim = sess.run(
            [tensors["train_step"], tensors["metrics"]["training"], tensors["metrics"]["training"]["sim"]],
            feed_dict=fill_feed_dict(placeholders, batch),
        )
        # print("sim", sim)
        # sim_list = []
        # for s in sim:
        #     s[0]
        # print("label", train_metrics["labels"])
        for k, v in train_metrics.items():
            accumulated_metrics[k].append(v)
        metrics_to_print = {k: np.mean(v) for k, v in accumulated_metrics.items()}
        info_str = ", ".join(["%s %.4f" % (k, v) for k, v in metrics_to_print.items()])
        print("epoch:%d,batch:%d, %s, time %.2fs" % (i_iter + 1, j, info_str, time.time() - t_start))
        print("epoch:%d,batch:%d, %s, time %.2fs" % (i_iter + 1, j, info_str, time.time() - t_start),file=f_train)
        # print(train_metrics)
        # accumulate over minibatches to reduce variance in the training metrics

        if j % config["training"]["print_after"] == 0:
            # metrics_to_print = {k: np.mean(v) for k, v in accumulated_metrics.items()}
            # info_str = ", ".join(["%s %.4f" % (k, v) for k, v in metrics_to_print.items()])
            # # reset the metrics
            accumulated_metrics = collections.defaultdict(list)
            if j // config["training"]["print_after"] % config["training"][
                "eval_after"
            ] == 0:
                eval_metrics = evaluate(
                    sess,
                    tensors["metrics"]["validation"],
                    placeholders,
                    train_val_graphs,
                    config["evaluation"]["batch_size"],
                    len(train_val_graphs) // config["evaluation"]["batch_size"],
                )
                info_str += ", " + ", ".join(
                    ["%s %.4f" % ("val/" + k, v) for k, v in eval_metrics.items()]
                )
                print("epoch:%d,batch:%d, %s, time %.2fs" % (i_iter + 1, j, info_str, time.time() - t_start))
                print("epoch:%d,batch:%d, %s, time %.2fs" % (i_iter + 1, j, info_str, time.time() - t_start),file=f_valid)
                # print(train_metrics)
                t_start = time.time()


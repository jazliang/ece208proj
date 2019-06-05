
import treeswift
from math import floor,ceil
import random

DEBUG_MODE = True


def score(node, time):
    return 0


def algorithm1(tree, k=10):
    """
    :param tree:
    :param k: scaling factor, # samples in interval Δt = floor
    :return:
    """
    # Maintain a set of branches that are currently being processed
    # The 'cur_proc_branches' set actually store the nodes associated with the edges
    cur_proc_branches = set()

    # Distances from root of all nodes in the tree
    def dfs(node):
        node.br_history = node.get_parent().br_history + [node.get_parent().dist_from_root]
        node.dist_from_root = node.get_parent().dist_from_root + node.edge

    for node, dist in tree.distances_from_root(unlabeled=True):
        node.dist_from_root = dist

    # Nodes in the tree: attribute, stores info such as scores
    # edge (connecting to its parent) info stored as attributes of node
    # node.count
    # node.score

    # Running count of predictions (sampled times)
    n_pred = 0

    # Running count of correct predictions
    n_correct_pred = 0

    # Next node that is going to branch
    # Initially, it is the root node
    next_branching_node = tree.root
    next_branching_node_dist = float('inf')


    # Number of leaf nodes in the 'cur_proc_branches' set
    # If n_leaves_in_set = size of 'cur_proc_branches', then the algorithm stops.
    n_leaves_in_set = 0

    if tree.root.is_leaf():
        n_leaves_in_set += 1

    while n_leaves_in_set != len(cur_proc_branches):
        # === Update the 'cur_proc_branches' set ===
        cur_proc_branches.remove(next_branching_node)
        cur_dist_from_root = next_branching_node_dist
        next_branching_node_dist = float('inf')

        for child in next_branching_node.child_nodes():
            if child.dist_from_root < next_branching_node_dist:
                next_branching_node_dist = child.dist_from_root
                next_branching_node = child

        # === Sampling ===
        if DEBUG_MODE:
            assert next_branching_node_dist > cur_dist_from_root

        n_samples = ceil((next_branching_node_dist - cur_dist_from_root) * k)

        for i in range(n_samples):
            # Make a prediction at time...
            time = random.uniform(cur_dist_from_root, next_branching_node_dist)

            best_score = 0
            best_node = None

            for node in cur_proc_branches:
                _score = score(node, time)
                if _score > best_score:
                    best_score = _score
                    best_node = node

            # Prediction
            n_pred += 1
            if best_node == next_branching_node:
                n_correct_pred += 1

    accuracy = n_pred / n_correct_pred


if __name__ == '__main__':
    filename = 'datasets/small.tre'
    tree = treeswift.read_tree_newick(filename)
    # print(dict(tree.distances_from_root(unlabeled=True)))
    print(tree, k=10)


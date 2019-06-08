
import treeswift
from math import floor, ceil, exp
import random

# Enable debug mode to assert or print debugging information
debug_mode = {
    'assert': True,
    'print_score': False
}


class Score:
    def __init__(self, type='exp_aging'):
        self.type = type

        if type == 'random':
            self.__score = self.__score_random
        elif type == 'counting':
            self.__score = self.__score_counting
        elif type == 'exp_aging':
            self.__score = self.__score_exp_aging
        else:
            raise ValueError('Unknown scoring function type.')

    def __call__(self, node, time):
        return self.__score(node, time)

    @staticmethod
    def __score_random(node, time):
        return random.uniform(0, 1)

    @staticmethod
    def __score_exp_aging(node, time):
        if node.is_root():
            return 1
        else:
            return (node.parent.score + 1) * exp(-time)

    @staticmethod
    def __score_counting(node, time):
        return len(node.br_history)


def algorithm1(tree, score, k=10):
    """
    :param tree:
    :param score: a scoring function object
    :param k: scaling factor, # samples in interval Δt = floor
    :return:
    """

    # === Branching history & Distance to root ===
    # Distances from root of all nodes in the tree
    def br_hist_dfs(node):
        if node.is_root():
            node.br_history = []
            node.dist_from_root = 0
        else:
            node.br_history = node.get_parent().br_history + [node.get_parent().dist_from_root]
            node.dist_from_root = node.get_parent().dist_from_root + node.edge_length

        for c in node.child_nodes():
            br_hist_dfs(c)

    br_hist_dfs(tree.root)

    # Running count of predictions (sampled times)
    n_pred = 0

    # Running count of correct predictions
    n_correct_pred = 0

    # Next node that is going to branch
    # Initially, it is the root node
    next_branching_node = tree.root
    next_branching_node_dist_from_root = 0

    # Maintain a set of branches that are currently being processed
    # The 'cur_proc_branches' set actually store the nodes associated with the edges
    cur_proc_branches = set()
    cur_proc_branches.add(tree.root)

    # Number of leaf nodes in the 'cur_proc_branches' set
    # If n_leaves_in_set = size of 'cur_proc_branches', then the algorithm stops.
    n_leaves_in_set = 0

    if tree.root.is_leaf():
        n_leaves_in_set += 1

    while True:
        # === Update the 'cur_proc_branches' set ===
        # Upon removing a node from the set, its score is fixed.
        next_branching_node.score = score(next_branching_node,
                                          next_branching_node.edge_length)
        # Remove this node from the set
        cur_proc_branches.remove(next_branching_node)

        cur_dist_from_root = next_branching_node_dist_from_root

        # Add the children of the node being removed to the set
        for child in next_branching_node.child_nodes():
            cur_proc_branches.add(child)
            # if child.dist_from_root < next_branching_node_dist_from_root:
            #     next_branching_node_dist_from_root = child.dist_from_root
            #     next_branching_node = child

        # Update 'next_branching_node'
        next_branching_node_dist_from_root = float('inf')
        for n in cur_proc_branches:
            if n.dist_from_root < next_branching_node_dist_from_root:
                next_branching_node = n
                next_branching_node_dist_from_root = n.dist_from_root


        # Check whether it is time to terminate
        # All nodes are leaves
        if n_leaves_in_set == len(cur_proc_branches):
            break

        # === Sampling ===
        if debug_mode['assert']:
            # TODO: Note that it is possible that multiple nodes branch at the same time.
            assert next_branching_node_dist_from_root >= cur_dist_from_root

        n_samples = ceil((next_branching_node_dist_from_root - cur_dist_from_root) * k)

        for i in range(n_samples):
            # Make a prediction at time...
            time = random.uniform(cur_dist_from_root, next_branching_node_dist_from_root)

            best_score = 0
            best_node = None

            if debug_mode['print_score']:
                print('------')

            for node in cur_proc_branches:
                _score = score(node, time - cur_dist_from_root)

                if debug_mode['print_score']:
                    print(_score)

                if _score > best_score:
                    best_score = _score
                    best_node = node

            # Prediction
            n_pred += 1
            if best_node == next_branching_node:
                n_correct_pred += 1

    # accuracy = n_correct_pred / n_pred
    # return accuracy
    return n_pred, n_correct_pred


def experiment(tree, repeat=100,
               algorithm='algorithm1', k=10, score=Score(type='exp_aging')):

    running_n_pred = 0
    running_n_correct_pred = 0

    for i in range(repeat):
        n_pred, n_correct_pred = eval(algorithm)(tree, score, k=k)
        running_n_pred += n_pred
        running_n_correct_pred += n_correct_pred

    print('=== Summary ===')
    print('Algorithm:', 1)
    print('Scoring function:', score.type)
    print('Acc:', running_n_correct_pred / running_n_pred)
    print()


if __name__ == '__main__':
    # filename = 'datasets/big.tre'
    # tree = treeswift.read_tree_newick(filename)
    # # print(tree)
    # # print(dict(tree.distances_from_root(unlabeled=True)))
    # print(algorithm1(tree, k=10))

    import os

    for filename in os.listdir('datasets'):
        if filename == 'big.tre':
            tree = treeswift.read_tree_newick('datasets/' + filename)
            experiment(tree, repeat=1)

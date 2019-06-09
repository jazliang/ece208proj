
import treeswift
from math import floor, ceil, exp
from queue import PriorityQueue
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
        elif type == 'ave_time':
            self.__score = self.__score_by_average
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
            return 3
        else:
            return (node.parent.score + 3) * exp(-time)

    @staticmethod
    def __score_counting(node, time):
        return len(node.br_history)

    @staticmethod
    def __score_by_average(node, time):
        if node.is_root():
            return int('inf')
        return len(node.br_history) / (node.dist_from_root * 1.0)


def algorithm1(tree, score, k=10, eval_start_time=0):
    """
    :param tree:
    :param score: a scoring function object
    :param k: scaling factor, # samples in interval Δt = floor
    :param eval_start_time: the time when the evaluation (i.e. computing accuracy) starts

    :return:
    """

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

        # Hasn't reached the set evaluation start time. Ignored.
        if next_branching_node_dist_from_root < eval_start_time:
            continue

        n_samples = ceil((next_branching_node_dist_from_root - cur_dist_from_root) * k)

        for i in range(n_samples):
            # Make a prediction at time...
            time = random.uniform(cur_dist_from_root, next_branching_node_dist_from_root)

            # Hasn't reached the set evaluation start time. Ignored.
            if time < eval_start_time:
                continue

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


class Experiment:
    def __init__(self, tree, repeat=100,
                 algorithm='algorithm1',
                 k=10, score='exp_aging', eval_ratio=1.0):
        self.tree = tree
        self.max_leaf_to_root_dist = self._extract_tree_info()
        self.eval_ratio = eval_ratio
        self.eval_start_time = self.max_leaf_to_root_dist * (1 - self.eval_ratio)

        self.repeat = repeat

        self.algorithm = eval(algorithm)
        self.k = k
        self.score = Score(type=score)

        self.running_n_pred = 0
        self.running_n_correct_pred = 0

    def _extract_tree_info(self):
        # Information to be extracted:
        # - Branching history of all nodes
        # - Distance to root of all nodes
        # - Maximum leaf-to-root distance

        def br_hist_dfs(node):
            if node.is_root():
                node.br_history = []
                node.dist_from_root = 0
            else:
                node.br_history = node.get_parent().br_history + [node.get_parent().dist_from_root]
                node.dist_from_root = node.get_parent().dist_from_root + node.edge_length

            _max_leaf_to_root_dist = 0
            if node.is_leaf():
                _max_leaf_to_root_dist = node.dist_from_root

            for c in node.child_nodes():
                _max_leaf_to_root_dist = max(br_hist_dfs(c), _max_leaf_to_root_dist)

            return _max_leaf_to_root_dist

        return br_hist_dfs(self.tree.root)

    def run(self):
        for i in range(self.repeat):
            _n_pred, _n_correct_pred = self.algorithm(tree=self.tree, k=self.k, score=self.score, eval_start_time=self.eval_start_time)
            self.running_n_pred += _n_pred
            self.running_n_correct_pred += _n_correct_pred

        self.report()

    def report(self):
        print('=== Summary ===')
        print('Algorithm:', self.algorithm.__name__)
        print('Scoring function:', self.score.type)
        print('Acc:', self.running_n_correct_pred / self.running_n_pred)
        print()


if __name__ == '__main__':
    # filename = 'datasets/big.tre'
    # tree = treeswift.read_tree_newick(filename)
    # # print(tree)
    # # print(dict(tree.distances_from_root(unlabeled=True)))
    # print(algorithm1(tree, k=10))

    pass


import treeswift
from math import floor, ceil, exp
from queue import PriorityQueue
import random
from queue import PriorityQueue


# Enable debug mode to assert or print debugging information
debug_mode = {
    'assert': False,
    'print_score': False
}


class Score:
    def __init__(self, type='exp_aging'):
        self.type = type
        self.sampling_time_variant = False

        if type == 'random':
            self.__score = self.__score_random
        elif type == 'counting':
            self.__score = self.__score_counting
        elif type == 'exp_aging':
            self.__score = self.__score_exp_aging
            self.sampling_time_variant = True
        elif type == 'ave_time':
            self.__score = self.__score_by_average
        elif type == 'ave_time_last_n_ancestors':
            self.__score = self.__score_ave_time_with_n_ancestors
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
        if node.is_root() or node.parent.is_root():
            return float('inf')
        return (len(node.br_history) - 1) * 1.0 / node.parent.dist_from_root

    @staticmethod
    def __score_ave_time_with_n_ancestors(node, time):
        # calculate the score by averaging the edge length of last_n_ancestors
        if node.is_root() or node.parent.is_root():
            return float('inf')
        return 1.0 / node.parent.ave_edge_last_n_ancestor


def algorithm1(tree, score, sample_scale=10, eval_start_time=0, top_k=5):
    """
    :param tree:
    :param score: a scoring function object
    :param sample_scale: scaling factor, # samples in interval Δt = floor
    :param eval_start_time: the time when the evaluation (i.e. computing accuracy) starts
    :param top_k: see if the correct answer is in the top k predictions

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

    j = 0
    while True:
        # === Debug print ===
        j += 1
        if j % 5000 == 0:
            print("***** Processing Node #{} is leaf? {}".format(j, next_branching_node.is_leaf()))

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

        n_samples = max(ceil((next_branching_node_dist_from_root - cur_dist_from_root) * sample_scale), 1)


        tmp_n_pre, tmp_n_correct_pred = run_sampling(n_samples, cur_dist_from_root, next_branching_node_dist_from_root,
                                                     cur_proc_branches, next_branching_node, score,
                                                    top_k=top_k, eval_start_time=eval_start_time)
        n_pred += tmp_n_pre
        n_correct_pred += tmp_n_correct_pred

    # accuracy = n_correct_pred / n_pred
    # return accuracy
    return n_pred, n_correct_pred


def run_sampling(n_samples, start_time, end_time, predict_branches, real_next_branch, score,
                 top_k=5, eval_start_time=0):
    """

    :param n_samples: sample numbers
    :param start_time: sampling start time
    :param end_time: sampling end time
    :param predict_branches: the current branch set that we need to predict on
    :param real_next_branch: the real next branch by which we calculate the correct prediction number
    :param score: the score function, ie. __score_counting, __score_exp_aging
    :param top_k: top_k branch we will select by their scores
    :param eval_start_time: the time when the evaluation (i.e. computing accuracy) starts
    :return:
    """
    n_pred = 0
    n_correct_pred = 0

    for i in range(n_samples):
        # Make a prediction at time...
        time = random.uniform(start_time, end_time)

        # best_score = 0
        # best_node = None

        if debug_mode['print_score']:
            print('------')

        pq = PriorityQueue()

        for node in predict_branches:
            _score = score(node, time - start_time)

            if debug_mode['print_score']:
                print(_score)

            # if _score > best_score:
            #     best_score = _score
            #     best_node = node

            if pq.qsize() < top_k:
                pq.put((_score, node))
            elif pq.queue[0][0] < _score:
                pq.get()
                pq.put((_score, node))

        # Prediction
        # Only predict for time >= eval_start_time
        if time >= eval_start_time:
            n_pred += 1
            if real_next_branch in list(zip(*pq.queue))[1]:
                n_correct_pred += 1

        # if time_variant is False, then just run once and multiply the results
        if not score.sampling_time_variant:
            break

    if score.sampling_time_variant:
        n_pred = n_pred * n_samples
        n_correct_pred  = n_correct_pred * n_samples

    return n_pred, n_correct_pred


class Experiment:
    def __init__(self, tree, repeat=100,
                 algorithm='algorithm1',
                 sample_scale=10, score='exp_aging', eval_ratio=1.0, top_k=10, last_n_ancestors=5):
        self.tree = tree

        self.algorithm = eval(algorithm)
        self.sample_scale = sample_scale
        self.repeat = repeat

        self.score = Score(type=score)
        self.top_k = top_k
        self.last_n_ancestors = last_n_ancestors

        self.max_leaf_to_root_dist = self._extract_tree_info()
        self.eval_ratio = eval_ratio
        self.eval_start_time = self.max_leaf_to_root_dist * (1 - self.eval_ratio)

        self.running_n_pred = 0
        self.running_n_correct_pred = 0

    def _extract_tree_info(self):
        # Information to be extracted:
        # - Branching history of all nodes
        # - Distance to root of all nodes
        # - Maximum leaf-to-root distance
        # - Average edge length of last_n_ancestors

        def br_hist_dfs(node):
            if node.is_root():
                node.br_history = []
                node.dist_from_root = 0
                node.ave_edge_last_n_ancestor = 0
            else:
                node.br_history = node.get_parent().br_history + [node.get_parent().dist_from_root]
                node.dist_from_root = node.get_parent().dist_from_root + node.edge_length

                start_index = 0 if len(node.br_history) < self.last_n_ancestors else -self.last_n_ancestors
                branch_num = len(node.br_history) if len(node.br_history) < self.last_n_ancestors else self.last_n_ancestors
                node.ave_edge_last_n_ancestor = (node.dist_from_root - node.br_history[start_index]) \
                                                * 1.0 / branch_num

            _max_leaf_to_root_dist = 0
            if node.is_leaf():
                _max_leaf_to_root_dist = node.dist_from_root

            for c in node.child_nodes():
                _max_leaf_to_root_dist = max(br_hist_dfs(c), _max_leaf_to_root_dist)

            return _max_leaf_to_root_dist

        return br_hist_dfs(self.tree.root)

    def run(self):
        for i in range(self.repeat):
            print("Repeat at {}th".format(i))
            _n_pred, _n_correct_pred = self.algorithm(tree=self.tree,
                                                      sample_scale=self.sample_scale,
                                                      score=self.score,
                                                      eval_start_time=self.eval_start_time,
                                                      top_k=self.top_k)
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
    # print(algorithm1(tree, sample_scale=10))

    pass

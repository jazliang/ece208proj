
from config import *
import treeswift
from math import floor, ceil, exp
import random
from queue import PriorityQueue
import os
import json
import datetime
import time
from tqdm import tqdm


def timing(func):
    """
    A decorator for measuring the running time of a function call

    :param func: the function to be measured
    :return: a wrapper function object
    """
    def wrapper(*arg):
        start = time.clock()
        res = func(*arg)
        stop = time.clock()

        if debug_mode['timing']:
            print('Time elapsed: %.04f second(s).' % (stop - start))

        return res

    return wrapper


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


def algorithm1(tree, score, sample_scale=10, eval_start_time=0.0, top_k=5):
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

    if debug_mode['print_progress']:
        pbar = tqdm(total=tree.n_nodes, unit='nodes')

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
            if debug_mode['print_progress']:
                pbar.update(1)
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

        for i in range(n_samples):
            # Make a prediction at time...
            time = random.uniform(cur_dist_from_root, next_branching_node_dist_from_root)

            # best_score = 0
            # best_node = None

            if debug_mode['print_score']:
                print('------')

            pq = PriorityQueue()

            for node in cur_proc_branches:
                _score = score(node, time - cur_dist_from_root)

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
                if next_branching_node in list(zip(*pq.queue))[1]:
                    n_correct_pred += 1

    # accuracy = n_correct_pred / n_pred
    # return accuracy
    if debug_mode['print_progress']:
        pbar.close()

    return n_pred, n_correct_pred


class Experiment:
    def __init__(self, tree_filename, repeat=100,
                 algorithm='algorithm1',
                 sample_scale=10, score='exp_aging', eval_ratio=1.0, top_k=5, last_n_ancestors=5,
                 use_prev_results=True):

        self.tree_filename = tree_filename
        self.tree = treeswift.read_tree_newick(os.path.join(*tree_filename))

        self.algorithm = eval(algorithm)
        self.sample_scale = sample_scale
        self.repeat = repeat

        self.score = Score(type=score)
        self.top_k = top_k
        self.last_n_ancestors = last_n_ancestors

        self.tree.n_nodes, self.tree.max_leaf_to_root_dist = self._extract_tree_info()

        self.eval_ratio = eval_ratio
        self.eval_start_time = self.tree.max_leaf_to_root_dist * (1.0 - self.eval_ratio)

        self.running_n_pred = 0
        self.running_n_correct_pred = 0

        # Use the results of previous experiments if available
        self.use_prev_results = use_prev_results

        self.summary = {
            'Settings': {
                'tree': tree_filename,  # Tree filename
                'alg': algorithm,  # Algorithm
                'sc': score,  # Scoring function
                'k': self.sample_scale,  # Sample scaling factor k
                'rpt': self.repeat,  # Repeat
                'top': self.top_k,  # Top k
                'lna': self.last_n_ancestors,  # Last n ancestors
                'evalr': self.eval_ratio  # Eval ratio
            },
            'Repeats': {}
        }

        self._settings_list = [k + '-' + str(v) for k, v in self.summary['Settings'].items()]

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

            if node.is_leaf():
                return 1, node.dist_from_root
            else:
                _max_leaf_to_root_dist = 0
                _n_nodes = 1
                for c in node.child_nodes():
                    _n, _d = br_hist_dfs(c)
                    _n_nodes += _n
                    _max_leaf_to_root_dist = max(_d, _max_leaf_to_root_dist)

                return _n_nodes, _max_leaf_to_root_dist

        return br_hist_dfs(self.tree.root)

    @timing
    def run(self):
        print('-----', 'Tree:', self.tree_filename, '-----')

        self._print_settings()

        if self.use_prev_results:
            _summary = self._check_prev_experiments()
            if _summary is not None:
                self.summary = _summary
                self._print_summary()
                return

        for i in range(self.repeat):
            _n_pred, _n_correct_pred = self.algorithm(tree=self.tree,
                                                      sample_scale=self.sample_scale,
                                                      score=self.score,
                                                      eval_start_time=self.eval_start_time,
                                                      top_k=self.top_k)
            self.running_n_pred += _n_pred
            self.running_n_correct_pred += _n_correct_pred

            self.summary['Repeats']['Repeat %d' % i] = {
                '# Predictions': _n_pred,
                '# Correct Predictions': _n_correct_pred,
                'Accuracy': _n_correct_pred / _n_pred
            }

            if debug_mode['print_repeat_results']:
                print('Repeat:', i)
                for k, v in self.summary['Repeats']['Repeat %d' % i].items():
                    print(k + ':', v)
                print()

        self.summary['Final Results'] = {
            'Total # Predictions': self.running_n_pred,
            'Total # Correct Predictions': self.running_n_correct_pred,
            'Accuracy': self.running_n_correct_pred / self.running_n_pred
        }

        # Summary of the experiment
        self._print_summary()
        self._save_results()

    def _print_settings(self):
        print(json.dumps(self.summary['Settings'], indent=4))
        print()

    def _print_summary(self):
        print('Summary for tree:', self.tree_filename)
        for k, v in self.summary['Final Results'].items():
            print(k + ':', v)
        print()

    def _save_results(self):
        # invalid filename using the time str format
        # _datetime = str(datetime.datetime.now().strftime("%m:%d:%H:%M:%S"))
        # _filename = '__'.join(self._settings_list) + '__' + _datetime
        _filename = '__'.join(self._settings_list)
        if not os.path.exists('log'):
            os.makedirs('log')

        fn = os.path.join('log', _filename)
        print(fn)
        with open(fn, 'w') as f:
            json.dump(self.summary, f, indent=4)

    def _check_prev_experiments(self):
        _this_part1 = '__'.join(self._settings_list[:4])
        _this_repeat = self._settings_list[4]
        _this_part2 = '__'.join(self._settings_list[5:])

        for log_filename in os.listdir('log'):
            if not log_filename.startswith('tree-'):
                continue

            _settings = log_filename.split('__')[:-1]  # Don't include date and time
            _part1 = '__'.join(_settings[:4])
            _repeat = _settings[4]
            _part2 = '__'.join(_settings[5:])

            if _this_part1 == _part1 and _this_part2 == _part2:
                _this_repeat = int(_this_repeat.split('-')[1])
                _repeat = int(_repeat.split('-')[1])

                if _this_repeat == _repeat:
                    print('INFO: Found previous experiment with same settings.\n')
                    with open(os.path.join('log', log_filename)) as f:
                        _summary = eval(f.read())
                    return _summary

        return None


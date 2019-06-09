from utils import *
import os
import treeswift

# '01.ft.mv.time9.tre'
# 'big.tre'
# '02.sub25.ft.mv.time9.tre'
# '01.sub25.ft.mv.time9.tre'
# '02.ft.mv.time9.tre'
# '02.true.time9.tre'
# 'merged_tree_0.tre'
# '04.ft.mv.tre'

filename = os.path.join('datasets', 'big.tre')
tree = treeswift.read_tree_newick(filename)
repeat = 10

Experiment(tree, repeat=repeat, algorithm='algorithm1', score='random', eval_ratio=1).run()
Experiment(tree, repeat=repeat, algorithm='algorithm1', score='ave_time', eval_ratio=1).run()
Experiment(tree, repeat=repeat, algorithm='algorithm1', score='ave_time_last_n_ancestors', eval_ratio=1).run()

Experiment(tree, repeat=repeat, algorithm='algorithm1', score='counting', eval_ratio=1).run()
Experiment(tree, repeat=repeat, algorithm='algorithm1', score='exp_aging', eval_ratio=1).run()



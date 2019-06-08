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
repeat = 1

experiment(tree, repeat=repeat, score=Score(type='random'))
experiment(tree, repeat=repeat, score=Score(type='counting'))
experiment(tree, repeat=repeat, score=Score(type='exp_aging'))


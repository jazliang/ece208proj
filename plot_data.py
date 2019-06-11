import os
import treeswift
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_hist(file_name):
    data = get_edge_length(file_name)
    print(data)
    plt.hist(data, bins=100)
    plt.title("Edge length distribution for tree {}".format(file_name))
    plt.xlabel("Edge length")
    plt.ylabel("Distribution")
    plt.show()
    # sns.distplot(data)


def plot_edge_length_with_distance_from_root(file_name):
    distance_from_root, edge_length = get_edge_length_with_distance_from_root(file_name)
    plt.plot(distance_from_root, edge_length, 'ro', markersize=1)
    plt.title("Edge length evolvement for tree {}".format(file_name))
    plt.xlabel("Time")
    plt.ylabel("Edge length")
    plt.show()

def read_tree(file_name):
    filename = os.path.join('datasets', file_name)
    tree = treeswift.read_tree_newick(filename)
    return tree

def get_edge_length_with_distance_from_root(file_name):
    tree = read_tree(file_name)
    distance_from_root = list()
    edge_length = list()

    for node in tree.traverse_levelorder():
        if node.is_root():
            node.distance_from_root = 0
            continue

        node.distance_from_root = node.parent.distance_from_root + node.edge_length
        distance_from_root.append(node.distance_from_root)
        edge_length.append(node.edge_length)
    return distance_from_root, edge_length


def get_edge_length(file_name):
    tree = read_tree(file_name)

    edge_lengthes = list()
    for node in tree.traverse_preorder():
        if node.is_root():
            continue
        edge_lengthes.append(node.edge_length)
    return edge_lengthes


## plot examples
plot_hist('merged_tree_0.tre')
plot_edge_length_with_distance_from_root('merged_tree_0.tre')
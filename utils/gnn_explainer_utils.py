# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:16:15 2023

@author: yu
"""

import random
from itertools import product
from scipy.special import comb, softmax
import numpy as np
from sklearn.linear_model import LinearRegression


class HGNNExplainer:

    def __init__(self,
                 node_attributes, node_embeddings,
                 max_nodes_known=20, max_attributes=10):
        self.node_attributes = node_attributes  # Node attribute matrix [n_nodes, n_attributes]
        self.node_embeddings = node_embeddings  # node embedding matrix [n_nodes, emb_size], where emb_size=n_attributes
        self.max_nodes_known = max_nodes_known  # Maximum number of known nodes
        self.max_attributes = max_attributes  # Maximum number of knowable attributes
        self.num_nodes = self.node_embeddings.shape[0]  # num of nodes

        # Save the relationship between node numbers and node attributes in a dictionary for easy querying
        self.node2attr = {}
        for i, attr in enumerate(self.node_attributes):
            self.node2attr[i] = [k for k, a in enumerate(attr) if a != 0]

    def readout(self, nodes):
        if len(nodes) == 0:
            return np.zeros(self.node_attributes.shape[1])
        else:
            # Average the attributes of the specified nodes as a summary description of those nodes
            return np.mean(self.node_attributes[nodes, :], axis=0)

    def predict(self, nodes_known, rank=0):
        # Average over known nodes as a vector representation
        # then dot product with the embedding vector of all nodes to get the score of all nodes
        nodes_readout = self.readout(nodes_known)
        score = np.matmul(nodes_readout, self.node_embeddings.T)  # [n_nodes]
        indices = sorted(range(self.num_nodes), key=lambda i: score[i], reverse=True)
        indices = [i for i in indices if i not in nodes_known]
        return indices[rank]  # 返回得分排名为rank的节点编号

    def generate_samples_by_masking_links(self, nodes_known):
        # Sample generation by masking links
        num_nodes = len(nodes_known)
        # num of combinations calculated
        nchoosek = [comb(num_nodes, i, exact=True) for i in range(num_nodes + 1)]
        # Calculate the total number of known attributes of the node
        n_attr_all = sum(len(self.node2attr[node]) for node in nodes_known)
        # If the number of known nodes is less than or equal to the threshold
        if num_nodes <= self.max_nodes_known:
            # Iterate through the masks of all nodes
            for t in product(range(2), repeat=len(nodes_known)):
                # Get unmasked nodes
                nodes = [n for i, n in zip(t, nodes_known) if i != 0]
                # Get readout results for unmasked nodes
                nodes_readout = self.readout(nodes)
                # Create a sample
                x = list(t) + [1] * n_attr_all
                weight = 1 / nchoosek[sum(t)]
                yield nodes_readout, x, weight
        # If the number of known nodes is greater than the threshold
        else:
            # Calculate the maximum number of samples
            max_sample_size = 2 ** self.max_nodes_known
            total_combinations = 2 ** num_nodes
            # Iterate over all possible known node numbers
            for n_node in range(1, num_nodes + 1):
                # 计算选择n_node个节点的所有可能组合数
                nchk = self.nchoosek[n_node]
                # Calculate the number of samples to be generated
                n_samples = round(max_sample_size / total_combinations * nchk)
                # If sample size is 0, continue to next round
                if n_samples == 0:
                    continue
                # Calculate the weights of the sample
                weight = 1 / n_samples
                samples = set()
                # Generate samples until the specified number of samples is reached
                while len(samples) < n_samples:
                    # Randomly select n_node of known nodes
                    nodes = tuple(sorted(random.sample(nodes_known, n_node)))
                    # If the selected node combination is not in the generated sample, it is added to the sample
                    if nodes not in samples:
                        samples.add(nodes)
                        # Get readout results for unmasked nodes
                        nodes_readout = self.readout(nodes)
                        # Create a sample
                        t = [1 if node in nodes else 0 for node in nodes_known]
                        x = t + [1] * n_attr_all
                        yield nodes_readout, x, weight
        pass

    def generate_samples_by_masking_attributes(self, nodes_known):
        # Calculate the total number of attributes for a known node
        n_attr_all = sum(len(self.node2attr[node]) for node in nodes_known)
        prev_attr = 0
        for i, node in enumerate(nodes_known):
            # Output the node being processed
            print('Mask attributes of node {}'.format(node))
            # Get the attributes of this node and make a copy
            node_attr = self.node_attributes[nodes_known, :].copy()
            attr = self.node2attr[node]
            n_attr = len(attr)
            # of schemes with k attributes selected
            nchoosek = [comb(n_attr, k, exact=True) for k in range(n_attr + 1)]
            # If the number of attributes does not exceed max_attributes, then all possible subsets of attributes are exhausted
            if n_attr <= self.max_attributes:
                for t in product(range(2), repeat=n_attr):
                    selected_attr = [n for i, n in zip(t, attr) if i != 0]
                    # Set unchecked attributes to 0
                    node_attr[i] = 0
                    for a in selected_attr:
                        node_attr[i, a] = self.node_attributes[node, a]
                    # Calculate the average attribute vector of all nodes
                    nodes_readout = np.mean(node_attr, axis=0)
                    # Construct a sample vector where the positions of known nodes and known attributes are 1 and the rest are 0
                    x = [1] * (len(nodes_known) + n_attr_all)
                    for ii, tt in enumerate(t):
                        if tt == 0:
                            x[prev_attr + ii] = 0
                            # Calculate sample weights
                    weight = 1 / nchoosek[sum(t)]
                    # Return to sample
                    yield nodes_readout, x, weight
            # If the number of attributes exceeds max_attributes, then max_attributes will be selected at random each time
            else:
                max_sample_size = 2 ** self.max_attributes
                total_combinations = 2 ** n_attr
                for n_a in range(1, n_attr + 1):
                    nchk = nchoosek[n_a]
                    n_samples = round(max_sample_size / total_combinations * nchk)
                    if n_samples == 0:
                        continue
                    weight = 1 / n_samples
                    samples = set()
                    while len(samples) < n_samples:
                        selected_attr = tuple(sorted(random.sample(attr, n_a)))
                        if selected_attr not in samples:
                            samples.add(selected_attr)
                            node_attr[i] = 0
                            for a in selected_attr:
                                node_attr[i, a] = self.node_attributes[node, a]
                            nodes_readout = np.mean(node_attr, axis=0)
                            x = [1] * (len(nodes_known) + n_attr_all)
                            for ii, a in enumerate(attr):
                                if a not in selected_attr:
                                    x[prev_attr + ii] = 0
                            yield nodes_readout, x, weight
            # Update prev_attr
            prev_attr += n_attr
        # End function
        pass

    def explained_factors(self, nodes_known):
        # Return to factors requiring explanation
        factors = ['link_{}'.format(node) for node in nodes_known]
        for node in nodes_known:
            factors.extend(['attr_{}_{}'.format(node, a) for a in self.node2attr[node]])
        return factors

    def explain_link(self, nodes_known, rank=0):
        # Predict links between specified nodes and interpret the predictions

        # Predict links and obtain probability distributions
        prediction = self.predict(nodes_known, rank)
        gen_readout_all = []
        xs = []
        ws = []

        # Generate samples by blocking links
        print('generate samples by masking links')
        for gen_readout, x, w in self.generate_samples_by_masking_links(nodes_known):
            gen_readout_all.append(gen_readout)
            xs.append(x)
            ws.append(w)

        # Sample generation by masking properties
        print('generate samples by masking attributes')
        for gen_readout, x, w in self.generate_samples_by_masking_attributes(nodes_known):
            gen_readout_all.append(gen_readout)
            xs.append(x)
            ws.append(w)

        gen_readout_all = np.vstack(gen_readout_all)  # [n_samples, n_attributes]
        xs = np.vstack(xs)

        # Predict the probability of the target node of the sample and softmax the probability
        ys = softmax(np.matmul(gen_readout_all, self.node_embeddings.T), axis=1)  # [n_samples, n_nodes]
        ys = ys[:, prediction]

        # Interpreting predictions using weighted linear regression
        reg = LinearRegression().fit(xs, ys, ws)
        weights = reg.coef_

        # Interpret the predicted results and return
        factors = self.explained_factors(nodes_known)
        sorted_factors_with_weights = sorted(zip(factors, weights), key=lambda t: t[1], reverse=True)
        return prediction, sorted_factors_with_weights


def softmax_for_weights(x: list):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return (exp_x / np.sum(exp_x, axis=0)).tolist()


def round_float_number(number):
    number = round(number, 4)
    return number


# if __name__ == '__main__':
#     #    num_nodes = 100
#     #    num_attr = 1000
#     #
#
#     #    node_attributes = np.where(np.random.random((num_nodes, num_attr)) > 0.99, 1, 0)
#     #    node_embeddings = np.ones((num_nodes, num_attr))
#     # np.random.random((num_nodes, num_attr))
#
#     node_attributes = np.array([[1, 2, 3], [0.01, -0.02, 0.05], [-0.01, 0.05, -0.02], [0.02, 0.05, -0.01]])
#     node_embeddings = np.array([[0.01, -0.02, 0.05], [-0.01, 0.05, -0.02], [0.02, 0.05, -0.01], [1, 2, 3]])
#
#     explainer = HGNNExplainer(node_attributes, node_embeddings)

#     # hyper edge level
#     nodes_known = [0, 1, 2]
#     prediction, explain = explainer.explain_link(nodes_known, rank=0)
#
#     print('Predicted link: {}'.format(prediction))
#     print('Explain:')
#     print(explain)
#
#     print('Done')

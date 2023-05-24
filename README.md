# MLHAT: Multi-Label Hoeffding Adaptive Trees for classification in multi-label data streams

Associated repository with complementary material to the manuscript *Hoeffding adaptive trees for multi-label classification on data streams* [submitted to *IEEE Transactions on Knowledge and Data Engineering*]:

* Source code of the MLHAT proposal
* Datasets used in the experimentation
* Complete tables of results
* Source code for a reproductible experimentation

## Source code

The purpose of this repository is to make public and accessible the source code of MLHAT, a proposal for multi-instance classification in data streams based on incremental decision trees, specifically using the principle of Hoeffding adaptive trees. The code is available under the [src](https://github.com/aestebant/mlhat/blob/main/src) folder with the following structure:
```
src
│   ml_hoeffding_tree.py > Source code of the main algorithm of MLHAT.
│   requeriments.yml
│   tutorial.ipynb
│
└───leaf_classifiers > Implementations of the multi-label classifiers used in the leaves of the tree.
│
└───nodes
│   │   ml_node.py > Abstract classes for multi-label adaptive nodes (leaves and branches)
│   │   ml_leaves.py > Implementation of the MLHAT leaf.
│   │   ml_branches.py > Implementations of the MLHAT branches.
│
└───split_criterion > Implementation of the multi-label split criterion used in MLHAT, based on combining Bernoully process with information gain.
```

The development environment is based on Python >=3.10, with a special mention to the library [`River`](https://riverml.xyz/0.16.0/) as the base for development of MLHAT. The complete list of libraries to replicate the environment is available in [requeriments.yml](https://github.com/aestebant/mlhat/blob/main/src/requeriments.yml).

The Jupyter notebook [tutorial.ipnb](https://github.com/aestebant/mlhat/blob/main/src/tutorial.ipynb) describes a complete tutorial for using the presented library, including loading data, building the tree in a streaming environment, evaluating and visualizing results.

## Datasets

The performance of MLHAT has been validated on a large selection of multi-label datasets that have been adapted to work in a streaming environment. The datasets belong to different sources and contexts, but all are publicly available in this [repository](https://www.uco.es/kdis/mllresources/).

## Results

The results associated to the complete experimentation carried out in this work are available in the [results](https://github.com/aestebant/mlhat/blob/main/results) folder. They are organized in #TODO


## Reproductible experimentation

All the experimentation has been executed in Python, using for the comparative analysis the implementations available in River of the main classification algorithms in multi-label data streams, with the default parameters proposed by the authors. #TODO
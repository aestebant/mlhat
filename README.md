# MLHAT: Multi-Label Hoeffding Adaptive Trees for classification in multi-label data streams

Associated repository with complementary material to the manuscript *Hoeffding adaptive trees for multi-label classification on data streams* [submitted to *IEEE Transactions on Knowledge and Data Engineering*]:

* Source code of the MLHAT proposal
* Datasets used in the experimentation
* Complete tables of results
* Source code for a reproductible experimentation

## Source code

The purpose of this repository is to make public and accessible the source code of MLHAT, a proposal for multi-instance classification in data streams based on incremental decision trees, specifically using the principle of Hoeffding adaptive trees. The code is available under the [src](src/) folder with the following structure:
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

The development environment is based on Python >=3.10, with a special mention to the library [`River`](https://riverml.xyz/0.16.0/) as the base for development of MLHAT. The complete list of libraries to replicate the environment is available in [requirements.yml](src/requirements.yml).

The Jupyter notebook [tutorial.ipnb](src/tutorial.ipynb) describes a complete tutorial for using the presented library, including loading data, building the tree in a streaming environment, evaluating and visualizing results.

## Datasets

The performance of MLHAT has been validated on a large selection of multi-label datasets that have been adapted to work in a streaming environment. The datasets belong to different sources and contexts, but all are publicly available in the [Multi-Label Classification Dataset Repository](https://www.uco.es/kdis/mllresources/). Moreover, all of then are available ready to use in this framework under the folder [src/ml_datasets](src/ml_datasets). The datasets employed are the following, and more information about how to load them is presented in the [tutorial.ipnb](src/tutorial.ipynb).

| **Dataset**      | **Instances** | **Features** | **Labels** | **Cardinality** | **Density** |
|------------------|---------------:|--------------:|------------:|-----------------:|-------------:|
| **Flags**        | 194           | 19           | 7          | 3.39            | 0.48        |
| **Chd**          | 555           | 49           | 6          | 2.58            | 0.43        |
| **WaterQuality** | 1060          | 16           | 14         | 5.07            | 0.36        |
| **Emotions**     | 593           | 72           | 6          | 1.87            | 0.31        |
| **VirusGO**      | 207           | 749          | 6          | 1.22            | 0.20        |
| **Birds**        | 645           | 260          | 19         | 1.01            | 0.05        |
| **Yeast**        | 2417          | 103          | 14         | 4.24            | 0.30        |
| **Scene**        | 2407          | 294          | 6          | 1.07            | 0.18        |
| **Gnegative**    | 1392          | 440          | 8          | 1.05            | 0.13        |
| **Plant**        | 978           | 440          | 12         | 1.08            | 0.09        |
| **Cal500**       | 502           | 68           | 174        | 26.04           | 0.15        |
| **Human**        | 3106          | 440          | 14         | 1.19            | 0.08        |
| **Genbase**      | 662           | 1186         | 27         | 1.25            | 0.05        |
| **Yelp**         | 10806         | 671          | 5          | 1.64            | 0.33        |
| **Medical**      | 978           | 1449         | 45         | 1.25            | 0.03        |
| **Eukaryote**    | 7766          | 440          | 22         | 1.15            | 0.05        |
| **Slashdot**     | 3782          | 1079         | 22         | 1.18            | 0.05        |
| **Enron**        | 1702          | 1001         | 53         | 4.27            | 0.08        |
| **Hypercube**    | 100000        | 100          | 10         | 1.00            | 0.10        |
| **Hypersphere**  | 100000        | 100          | 10         | 2.31            | 0.23        |
| **Langlog**      | 1460          | 1004         | 75         | 15.94           | 0.21        |
| **Stackex**      | 1675          | 585          | 227        | 2.41            | 0.01        |
| **Tmc**          | 28596         | 500          | 22         | 2.22            | 0.10        |
| **Ohsumed**      | 13929         | 1002         | 23         | 0.81            | 0.04        |
| **D20ng**        | 19300         | 1006         | 20         | 1.42            | 0.07        |
| **Mediamill**    | 43907         | 120          | 101        | 4.38            | 0.04        |
| **Corel5k**      | 5000          | 499          | 374        | 3.52            | 0.01        |
| **Corel16k**     | 13766         | 500          | 153        | 2.86            | 0.02        |
| **Bibtex**       | 7395          | 1836         | 159        | 2.40            | 0.02        |
| **Nuswidec**     | 269648        | 129          | 81         | 1.87            | 0.02        |
| **YahooSociety** | 14512         | 31802        | 27         | 1.67            | 0.06        |
| **Eurlex**       | 19348         | 5000         | 201        | 2.21            | 0.01        |


## Results

The results associated to the complete experimentation carried out in this work are available in the [results](results/) folder. The following shows the average result of all the datasets and algorithms included in the experimentation with respect to 12 multi-label classification evaluation metrics following the prequential evaluation, with our MLHAT obtaining the best result in 11 of them. 

| **Algorithm** | **Su. Acc** | **H. Loss** | **Ex. Acc** | **Ex. Pre** | **Ex. Rec** | **Ex. F1** | **Mi. Pre** | **Mi. Rec** | **Mi. F1** | **Ma. Pre** | **Ma. Rec** | **Ma. F1** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kNN | 0.2549 | 0.0962 | 0.8989 | 0.4353 | 0.3937 | 0.3977 | 0.5327 | 0.3857 | 0.4275 | 0.3910 | 0.2714 | 0.2870 |
| NB | 0.1279 | 0.1449 | 0.8502 | 0.2893 | 0.3562 | 0.2858 | 0.3736 | 0.3582 | 0.2983 | 0.2048 | 0.2728 | 0.2036 |
| AMR | 0.2733 | 0.0927 | 0.9024 | 0.4747 | 0.4301 | 0.4315 | 0.6819 | 0.4176 | 0.4558 | 0.4084 | 0.3073 | 0.3234 |
| HT | 0.1895 | 0.0983 | 0.8967 | 0.3941 | 0.3422 | 0.3459 | 0.5460 | 0.3349 | 0.3885 | 0.3047 | 0.2295 | 0.2389 |
| HAT | 0.2224 | 0.0934 | 0.9016 | 0.4331 | 0.3816 | 0.3848 | 0.5584 | 0.3716 | 0.4243 | 0.3300 | 0.2555 | 0.2663 |
| EFDT | 0.2120 | 0.0945 | 0.9006 | 0.4262 | 0.3829 | 0.3820 | 0.5482 | 0.3761 | 0.4260 | 0.3294 | 0.2621 | 0.2741 |
| SGT | 0.0466 | 0.2792 | 0.7150 | 0.2006 | 0.3162 | 0.1859 | 0.2116 | 0.3170 | 0.2040 | 0.1549 | 0.2967 | 0.1494 |
| MT | 0.1584 | 0.1028 | 0.8923 | 0.3418 | 0.2612 | 0.2772 | 0.5928 | 0.2553 | 0.3143 | 0.2934 | 0.1823 | 0.1927 |
| OBA | 0.2896 | 0.0894 | 0.9057 | 0.4861 | 0.4326 | 0.4390 | **0.7152** | 0.4188 | 0.4653 | 0.4214 | 0.3068 | 0.3290 |
| OBOA | 0.2856 | 0.0904 | 0.9046 | 0.4827 | 0.4319 | 0.4369 | 0.7034 | 0.4183 | 0.4612 | 0.4193 | 0.3085 | 0.3290 |
| ARF | 0.2496 | **0.0847** | 0.9104 | 0.4595 | 0.3847 | 0.4001 | 0.7097 | 0.3743 | 0.4408 | 0.4173 | 0.2597 | 0.2801 |
| AMF | 0.2373 | 0.0874 | 0.9074 | 0.4515 | 0.3709 | 0.3868 | 0.6731 | 0.3613 | 0.4282 | 0.4022 | 0.2619 | 0.2865 |
| MLHT | 0.1736 | 0.1359 | 0.8591 | 0.3052 | 0.2712 | 0.2743 | 0.3196 | 0.2589 | 0.2761 | 0.1435 | 0.1562 | 0.1295 |
| MLHTPS | 0.1582 | 0.1105 | 0.8845 | 0.3314 | 0.2668 | 0.2759 | 0.4715 | 0.2621 | 0.2958 | 0.2270 | 0.1859 | 0.1821 |
| iSOUPT | 0.2710 | 0.0945 | 0.9006 | 0.4752 | 0.4253 | 0.4289 | 0.6960 | 0.4132 | 0.4578 | 0.3904 | 0.2907 | 0.3074 |
| MLHAT | **0.3158** | **0.0847** | **0.9104** | **0.5343** | **0.4820** | **0.4863** | 0.6939 | **0.4701** | **0.5211** | **0.4949** | **0.3546** | **0.3786** |




## Reproductible experimentation

All the experimentation has been executed in Python, using for the comparative analysis the implementations available in River of the main classification algorithms in multi-label data streams, with the default parameters proposed by the authors. #TODO
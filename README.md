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

The performance of MLHAT has been validated on a large selection of multi-label datasets that have been adapted to work in a streaming environment. The datasets belong to different sources and contexts, but all are publicly available in the [Multi-Label Classification Dataset Repository](https://www.uco.es/kdis/mllresources/). Moreover, all of then are available ready to use in this framework under the folder [src/ml_datasets](https://github.com/aestebant/mlhat/blob/main/src/ml_datasets). The datasets employed are the following, and more information about how to load them is presented in the [tutorial.ipnb](https://github.com/aestebant/mlhat/blob/main/src/tutorial.ipynb).

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
| **Imdb**         | 120919        | 1001         | 28         | 1.00            | 0.04        |
| **YahooSociety** | 14512         | 31802        | 27         | 1.67            | 0.06        |
| **Eurlex**       | 19348         | 5000         | 201        | 2.21            | 0.01        |


## Results

The results associated to the complete experimentation carried out in this work are available in the [results](https://github.com/aestebant/mlhat/blob/main/results) folder. The following shows the average result of all the datasets and algorithms included in the experimentation with respect to 12 multi-label classification evaluation metrics following the prequential evaluation, with our MLHAT obtaining the best result in 11 of them.

| Algorithm | Subset Acc | Hamming Loss | Example Acc | Example Precision | Example Recall | Example F1 | Micro Precision | Micro Recall | Micro F1 | Macro Precision | Macro Recall | Macro F1 |
|-----------|------------|--------------|-------------|-------------------|----------------|------------|-----------------|--------------|----------|-----------------|--------------|----------|
| kNN       | 0.2486     | 0.0957       | 0.8994      | 0.4278            | 0.3856         | 0.3899     | 0.5261          | 0.3775       | 0.4197   | 0.3858          | 0.2644       | 0.2801   |
| NB        | 0.1241     | 0.1510       | 0.8440      | 0.2840            | 0.3625         | 0.2824     | 0.3657          | 0.3645       | 0.2949   | 0.2014          | 0.2767       | 0.2011   |
| AMR       | 0.2653     | 0.0920       | 0.9030      | 0.4611            | 0.4175         | 0.4189     | 0.6755          | 0.4053       | 0.4428   | 0.4014          | 0.2980       | 0.3137   |
| HT        | 0.1840     | 0.0976       | 0.8974      | 0.3833            | 0.3326         | 0.3363     | 0.5357          | 0.3255       | 0.3781   | 0.2986          | 0.2230       | 0.2325   |
| HAT       | 0.2159     | 0.0930       | 0.9021      | 0.4215            | 0.3710         | 0.3742     | 0.5477          | 0.3613       | 0.4131   | 0.3233          | 0.2483       | 0.2591   |
| EFDT      | 0.2061     | 0.0941       | 0.9009      | 0.4157            | 0.3730         | 0.3723     | 0.5374          | 0.3664       | 0.4156   | 0.3227          | 0.2552       | 0.2672   |
| SGT       | 0.0426     | 0.2874       | 0.7077      | 0.2025            | 0.3202         | 0.1854     | 0.1975          | 0.3231       | 0.2043   | 0.1527          | 0.3022       | 0.1501   |
| MT        | 0.1566     | 0.1031       | 0.8920      | 0.3387            | 0.2588         | 0.2747     | 0.5862          | 0.2534       | 0.3109   | 0.2958          | 0.1819       | 0.1920   |
| OBA       | 0.2813     | 0.0888       | 0.9062      | 0.4731            | 0.4205         | 0.4269     | 0.7083          | 0.4070       | 0.4529   | 0.4163          | 0.2978       | 0.3195   |
| OBOA      | 0.2774     | 0.0898       | 0.9052      | 0.4699            | 0.4199         | 0.4249     | 0.6966          | 0.4065       | 0.4490   | 0.4149          | 0.2994       | 0.3195   |
| ARF       | 0.2423     | 0.0843       | 0.9108      | 0.4466            | 0.3736         | 0.3886     | 0.7001          | 0.3634       | 0.4284   | 0.4140          | 0.2520       | 0.2719   |
| AMF       | 0.2442     | 0.0936       | 0.9015      | 0.4736            | 0.3903         | 0.4066     | 0.6804          | 0.3839       | 0.4475   | 0.4227          | 0.2820       | 0.3037   |
| MLHT      | 0.1717     | 0.1343       | 0.8608      | 0.3069            | 0.2694         | 0.2737     | 0.3209          | 0.2565       | 0.2751   | 0.1429          | 0.1526       | 0.1265   |
| MLHTPS    | 0.1535     | 0.1093       | 0.8857      | 0.3215            | 0.2589         | 0.2677     | 0.4747          | 0.2544       | 0.2872   | 0.2226          | 0.1804       | 0.1769   |
| iSOUPT    | 0.2631     | 0.0938       | 0.9013      | 0.4619            | 0.4131         | 0.4167     | 0.6892          | 0.4012       | 0.4451   | 0.3851          | 0.2820       | 0.2983   |
| **MLHAT**     | **0.3073**     | **0.0843**       | **0.9108**      | **0.5221**            | **0.4699**         | **0.4744**     | 0.6870          | **0.4581**       | **0.5092**   | **0.4914**          | **0.3447**       | **0.3685**   |



## Reproductible experimentation

All the experimentation has been executed in Python, using for the comparative analysis the implementations available in River of the main classification algorithms in multi-label data streams, with the default parameters proposed by the authors. #TODO
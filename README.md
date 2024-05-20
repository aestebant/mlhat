# MLHAT: Multi-Label Hoeffding Adaptive Trees for classification in multi-label data streams

Associated repository with complementary material to the manuscript *Hoeffding adaptive trees for multi-label classification on data streams* [submitted to the *Artificial Intelligence Journal*]:

* Source code of the MLHAT proposal
* Datasets used in the experimentation
* Complete tables of results
* Source code for a reproductible experimentation

## Source code

The purpose of this repository is to make public and accessible the source code of MLHAT, a proposal for multi-instance classification in data streams based on incremental decision trees, specifically using the principle of Hoeffding adaptive trees. The code is available under the [src](src/) folder with the following structure:

```text
src
│   ml_hoeffding_tree.py > Source code of the main algorithm of MLHAT.
│   requirements.yml
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

The performance of MLHAT has been validated on a large selection of multi-label datasets that have been adapted to work in a streaming environment. The datasets belong to different sources and contexts, but all are publicly available in the [Multi-Label Classification Dataset Repository](https://www.uco.es/kdis/mllresources/).

| **Dataset**      | **Instances** | **Features** | **Labels** | **Cardinality** | **Density** |
|------------------|--------------:|-------------:|-----------:|----------------:|------------:|
| **Flags**        | 194           | 19           | 7          | 3.39            | 0.48        |
| **WaterQuality** | 1060          | 16           | 14         | 5.07            | 0.36        |
| **Emotions**     | 593           | 72           | 6          | 1.87            | 0.31        |
| **VirusGO**      | 207           | 749          | 6          | 1.22            | 0.20        |
| **Birds**        | 645           | 260          | 19         | 1.01            | 0.05        |
| **Yeast**        | 2417          | 103          | 14         | 4.24            | 0.30        |
| **Scene**        | 2407          | 294          | 6          | 1.07            | 0.18        |
| **Gnegative**    | 1392          | 440          | 8          | 1.05            | 0.13        |
| **Plant**        | 978           | 440          | 12         | 1.08            | 0.09        |
| **Human**        | 3106          | 440          | 14         | 1.19            | 0.08        |
| **Yelp**         | 10806         | 671          | 5          | 1.64            | 0.33        |
| **Medical**      | 978           | 1449         | 45         | 1.25            | 0.03        |
| **Eukaryote**    | 7766          | 440          | 22         | 1.15            | 0.05        |
| **Slashdot**     | 3782          | 1079         | 22         | 1.18            | 0.05        |
| **Enron**        | 1702          | 1001         | 53         | 4.27            | 0.08        |
| **Hypercube**    | 100000        | 100          | 10         | 1.00            | 0.10        |
| **Langlog**      | 1460          | 1004         | 75         | 15.94           | 0.21        |
| **Stackex**      | 1675          | 585          | 227        | 2.41            | 0.01        |
| **Tmc**          | 28596         | 500          | 22         | 2.22            | 0.10        |
| **Ohsumed**      | 13929         | 1002         | 23         | 0.81            | 0.04        |
| **D20ng**        | 19300         | 1006         | 20         | 1.42            | 0.07        |
| **Mediamill**    | 43907         | 120          | 101        | 4.38            | 0.04        |
| **Corel16k**     | 13766         | 500          | 153        | 2.86            | 0.02        |
| **Bibtex**       | 7395          | 1836         | 159        | 2.40            | 0.02        |
| **Nuswidec**     | 269648        | 129          | 81         | 1.87            | 0.02        |
| **Nuswideb**     | 269648        | 501          | 81         | 1.87            | 0.02        |
| **Imdb**         | 120919        | 1001         | 28         | 1.00            | 0.04        |
| **YahooSociety** | 14512         | 31802        | 27         | 1.67            | 0.06        |
| **Eurlex**       | 19348         | 5000         | 201        | 2.21            | 0.01        |

Moreover, a set of synthetic datasets have been generated to test the performance of MLHAT under specific concept drifts and other different conditions. The datasets have been generated using the multi-label meta-generator provided by the [MOA framewok](https://moa.cms.waikato.ac.nz) and made publicly available in our repository under the folder [synthdata](synthdata)

| **Dataset**     | **Instances** | **Features**    | **Labels** | **Generator** | **Drift type** | **Drift width** |
|----------------|--------------:|:----------------:|-----------:|:--------------:|:---------------:|----------------:|
| **SynTreeSud**  | 50000        | 20 num.+10 cat. | 8          | Random Tree   | sudden         | 1               |
| **SynRBFSud**   | 50000        | 80 numeric      | 25         | Random RBF    | sudden         | 1               |
| **SynHPSud**    | 50000        | 30 numeric      | 8          | Hyper Plane   | sudden         | 1               |
| **SynTreeGrad** | 50000        | 20 num.+10 cat. | 8          | Random Tree   | gradual        | 500             |
| **SynRBFGrad**  | 50000        | 80 numeric      | 25         | Random RBF    | gradual        | 500             |
| **SynHPGrad**   | 50000        | 30 numeric      | 8          | Hyper Plane   | gradual        | 500             |
| **SynTreeInc**  | 50000        | 20 num.+10 cat. | 8          | Random Tree   | incremental    | 275             |
| **SynRBFInc**   | 50000        | 80 numeric      | 25         | Random RBF    | incremental    | 275             |
| **SynHPInc**    | 50000        | 30 numeric      | 8          | Hyper Plane   | incremental    | 275             |
| **SynTreeRec**  | 50000        | 20 num.+10 cat. | 8          | Random Tree   | recurrent      | 1               |
| **SynRBFRec**   | 50000        | 80 numeric      | 25         | Random RBF    | recurrent      | 1               |
| **SynHPRec**    | 50000        | 30 numeric      | 8          | Hyper Plane   | recurrent      | 1               |

Moreover, all of then are available ready to use in this framework under the folder [src/ml_datasets](src/ml_datasets). The datasets employed are the following, and more information about how to load them is presented in the [tutorial.ipnb](src/tutorial.ipynb).

## Results

The results associated to the complete experimentation carried out in this work are available in the [results](results/) folder. The following shows the average result of all the datasets and algorithms included in the experimentation with respect to 12 multi-label classification evaluation metrics following the prequential evaluation, with our MLHAT obtaining the best result in 11 of them.

| Algorithm | Subset Acc.   | Hamming loss        | Example-based F1     | Example-based Precision   | Example-based Recall    | Micro F1       | Micro Precision     | Micro Recall      | Macro F1       | Macro Precision     | Macro Recall      | Time (s)       |
|----------|---------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|---------------|
| **MLHAT**    | **0.2646** | 0.1301          | **0.4755** | 0.6195          | 0.4892          | **0.4855** | 0.5309          | 0.4556          | **0.3871** | 0.4691          | **0.3661** | 14994         |
| **KNN**      | 0.2018        | 0.1273          | 0.3321          | 0.4781          | 0.3234          | 0.3744          | 0.5170          | 0.3220          | 0.2609          | 0.4035          | 0.2309          | 102362        |
| **NB**       | 0.0993        | 0.1842          | 0.2338          | 0.3794          | 0.3157          | 0.2479          | 0.3392          | 0.3211          | 0.1754          | 0.2198          | 0.2582          | 7578          |
| **AMR**      | 0.2111        | 0.1177          | 0.3343          | 0.5716          | 0.3263          | 0.3728          | 0.6513          | 0.3207          | 0.2626          | 0.4467          | 0.2356          | 29882         |
| **HT**       | 0.1512        | 0.1232          | 0.2687          | 0.5127          | 0.2608          | 0.3176          | 0.5336          | 0.2574          | 0.2003          | 0.3424          | 0.1770          | 42975         |
| **HAT**      | 0.1746        | 0.1195          | 0.3076          | 0.5352          | 0.3008          | 0.3612          | 0.5494          | 0.2972          | 0.2319          | 0.3659          | 0.2056          | 43651         |
| **EFDT**     | 0.1620        | 0.1236          | 0.3066          | 0.4996          | 0.3028          | 0.3603          | 0.5095          | 0.3015          | 0.2413          | 0.3465          | 0.2138          | 36942         |
| **SGT***     | 0.0337        | 0.2562          | 0.1221          | 0.4381          | 0.2103          | 0.1392          | 0.2115          | 0.2122          | 0.1016          | 0.1638          | 0.1991          | 298466        |
| **MT***      | 0.1224        | 0.1285          | 0.1920          | 0.5126          | 0.1789          | 0.2223          | 0.5606          | 0.1750          | 0.1335          | 0.2922          | 0.1232          | 186185        |
| **ARF**      | 0.1967        | **0.1126** | 0.3178          | 0.5507          | 0.3008          | 0.3730          | 0.6886          | 0.2972          | 0.2440          | 0.4710          | 0.2081          | 29308         |
| **AMF***     | 0.1935        | 0.1247          | 0.2987          | **0.6457** | 0.2834          | 0.3367          | 0.6681          | 0.2790          | 0.2267          | 0.4547          | 0.2029          | 131491        |
| **ABALR**    | 0.2189        | 0.1156          | 0.3283          | 0.5958          | 0.3178          | 0.3638          | **0.7057** | 0.3113          | 0.2509          | **0.4808** | 0.2254          | 12584         |
| **ABOLR**    | 0.2162        | 0.1164          | 0.3276          | 0.5931          | 0.3182          | 0.3616          | 0.6959          | 0.3115          | 0.2512          | 0.4682          | 0.2270          | 13868         |
| **MLBELS***  | 0.2046        | 0.1858          | 0.4595          | 0.4854          | **0.5452** | 0.4703          | 0.4579          | **0.5173** | 0.3090          | 0.4518          | 0.3650          | 77813         |
| **MLHT**     | 0.1419        | 0.1723          | 0.2495          | 0.2792          | 0.2419          | 0.2523          | 0.2992          | 0.2343          | 0.1139          | 0.2497          | 0.1522          | **1458** |
| **MLHTPS**   | 0.1211        | 0.1343          | 0.1900          | 0.4596          | 0.1856          | 0.2096          | 0.4164          | 0.1823          | 0.1337          | 0.3058          | 0.1327          | 29517         |
| **iSOUPT**   | 0.2154        | 0.1175          | 0.3284          | 0.5938          | 0.3202          | 0.3710          | 0.6908          | 0.3149          | 0.2507          | 0.4608          | 0.2247          | 6486          |

## Reproductible experimentation

All the experimentation has been run in Python, using for the comparative analysis the implementations available in River of the main classification algorithms in multi-label data streams, with the default parameters proposed by the authors. The methods used, their parameters and the reference implementation used are detailed below.

| Algorithm | Family | Parameters | Implementation reference |
|:-|:-|:-|:-|
| MLHAT | Multilabel tree | `grace_period=200, delta=1e-05, tau=0.05, split_criterion=MLInfoGain, leaf_prediction=MLNBA, splitter=MLGaussian, nb_threshold=0, drift_detector=ADWIN, drift_window_threshold=50, switch_significance=0.05, poisson_rate=1.0` | This repository |
| MLHT | Multilabel tree | `grace_period=200, delta=1e-05, tau=0.05, split_criterion=MLInfoGain, leaf_prediction=MLNBA, splitter=MLGaussian, nb_threshold=0, leaf_classifier=MajorityLabelset` | <https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/multilabel/MultilabelHoeffdingTree.java>|
|MLHTPS | Multilabel tree | `grace_period=200, delta=1e-05, tau=0.05, split_criterion=MLInfoGain, leaf_prediction=MLNBA, splitter=MLGaussian, nb_threshold=0, leaf_classifier=PrunedSet(HoeffdingTree)` | <https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/multilabel/MultilabelHoeffdingTree.java>|
| iSOUPT | Multilabel tree | `grace_period=200, delta=1e-5, tau=0.05, leaf_prediction=LogisticRegression, model_selector_decay=0.95, splitter=TEBST, min_samples_split=5`| <https://riverml.xyz/0.15.0/api/tree/iSOUPTreeRegressor/> |
| MLBELS | Multilabel neural network | `N1=3, N2=25, N3=1, max_learners=100, shrink_coef=0.8, regularization_coef=2**-30, batch_size=50, preprocess=True, tau=1.5` | <https://github.com/sepehrbakhshi/ML-BELS> |
| BR+HT | Tree | `grace_period=200, delta=1e-05, tau=0.05, split_criterion=InfoGain, leaf_prediction=NBA, splitter=Gaussian, nb_threshold=0` | <https://riverml.xyz/0.15.0/api/tree/HoeffdingTreeClassifier/> |
| BR+EFDT | Tree | `grace_period=200, delta=1e-05, tau=0.05, split_criterion=InfoGain, leaf_prediction=NBA, splitter=Gaussian, nb_threshold=0, min_samples_reevaluate=20` | <https://riverml.xyz/0.15.0/api/tree/ExtremelyFastDecisionTreeClassifier/> |
| BR+HAT | Tree | `grace_period=200, delta=1e-05, tau=0.05, split_criterion=InfoGain, leaf_prediction=NBA, splitter=Gaussian, nb_threshold=0, drift_detector=ADWIN, drift_window_threshold=50, switch_significance=0.05`| <https://riverml.xyz/0.15.0/api/tree/HoeffdingAdaptiveTreeClassifier/> |
| BR+MT | Tree | `step=0.1, loss=log, use_aggregator=True, dirichlet=0.5, split_pure=False` | <https://riverml.xyz/0.15.0/api/forest/AMFClassifier/> |
| BR+SGT | Tree | `grace_period=200, delta=1e-5, init_pred=0.0, lambda_value=0.1, gamma=1.0` | <https://riverml.xyz/0.15.0/api/tree/SGTClassifier/> |
| BR+ARF | Forest | `n_models=10, max_features=sqrt, grace_period=50, delta=0.01, tau=0.05, split_criterion=InfoGain, leaf_prediction=NBA, splitter=Gaussian, nb_threshold=0, lambda_value=6, drift_detector=ADWIN, warning_detector=ADWIN` | <https://riverml.xyz/0.15.0/api/forest/ARFClassifier/> |
| BR+AMF | Forest | `n_models=10, step=1.0, use_aggregation=True, dirchlet=0.5, split_pure=False` | <https://riverml.xyz/0.15.0/api/forest/AMFClassifier/> |
| BR+kNN | Distance-based | `n_neighbors=5, window_size=200, min_distance_keep=0.0, weighted=True, cleanup_every=0, distance_func=Euclidean, softmax=False` | <https://riverml.xyz/0.15.0/api/neighbors/KNNClassifier/> |
| BR+NB | Bayesian | - | <https://riverml.xyz/0.15.0/api/naive-bayes/GaussianNB/> |
| BR+AMR | Rules | `n_min=200, delta=1e-5, tau=0.05, pred_model=LogisticRegression, splitter=TEBST, drift_detector=ADWIN, fading_factor=0.99, anomaly_threshold=-0.75, m_min=30, min_samples_split=5` | <https://riverml.xyz/0.15.0/api/rules/AMRules/> |
| BR+OBA | Ensemble | `model=LogisticRegression, n_models=10, drift_detector=ADWIN` |  <https://riverml.xyz/0.15.0/api/ensemble/ADWINBaggingClassifier/> |
| BR+OBOA | Ensemble | `model=LogisticRegression, n_models=10, drift_detector=ADWIN` | <https://riverml.xyz/0.15.0/api/ensemble/ADWINBoostingClassifier/> |

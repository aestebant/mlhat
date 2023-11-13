# MLHAT: Multi-Label Hoeffding Adaptive Trees for classification in multi-label data streams

Associated repository with complementary material to the manuscript *Hoeffding adaptive trees for multi-label classification on data streams* [submitted to the *Artificial Intelligence Journal*]:

* Source code of the MLHAT proposal
* Datasets used in the experimentation
* Complete tables of results
* Source code for a reproductible experimentation

## Source code

The purpose of this repository is to make public and accessible the source code of MLHAT, a proposal for multi-instance classification in data streams based on incremental decision trees, specifically using the principle of Hoeffding adaptive trees. The code is available under the [src](src/) folder with the following structure:
```
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

| **Algorithm** | **Su. Acc**     | **H. Loss**   | **Ex. Pre**     | **Ex. Rec**     | **Ex. F1**      | **Mi. Pre**     | **Mi. Rec**     | **Mi. F1**      | **Ma. Pre**     | **Ma. Rec**     | **Ma. F1**      | **Time (s)**   |
|--------------:|----------------:|--------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|----------------:|---------------:|
| **KNN**       | 0.2060          | 0.1299        | 0.4754          | 0.3315          | 0.3408          | 0.5203          | 0.3296          | 0.3827          | 0.4069          | 0.2379          | 0.2686          | 102274        |
| **NB**        | 0.1024          | 0.1850        | 0.3879          | 0.3194          | 0.2398          | 0.3470          | 0.3244          | 0.2543          | 0.2268          | 0.2611          | 0.1811          | 7567          |
| **AMR**       | 0.2130          | 0.1207        | 0.5686          | 0.3310          | 0.3396          | 0.6490          | 0.3252          | 0.3784          | 0.4456          | 0.2394          | 0.2673          | 29808         |
| **HT**        | 0.1550          | 0.1257        | 0.5116          | 0.2678          | 0.2761          | 0.5367          | 0.2639          | 0.3247          | 0.3453          | 0.1829          | 0.2066          | 42935         |
| **HAT**       | 0.1781          | 0.1219        | 0.5343          | 0.3071          | 0.3144          | 0.5522          | 0.3031          | 0.3675          | 0.3687          | 0.2106          | 0.2373          | 43600         |
| **EFDT**      | 0.1652          | 0.1264        | 0.4992          | 0.3092          | 0.3135          | 0.5138          | 0.3074          | 0.3668          | 0.3512          | 0.2190          | 0.2472          | 36901         |
| **SGT***      | 0.0356          | 0.2604        | 0.4323          | 0.2149          | 0.1275          | 0.2200          | 0.2163          | 0.1452          | 0.1682          | 0.2023          | 0.1052          | 294527        |
| **MT***       | 0.1244          | 0.1319        | 0.5104          | 0.1831          | 0.1967          | 0.5581          | 0.1786          | 0.2281          | 0.2994          | 0.1264          | 0.1384          | 185499        |
| **ARF**       | 0.2025          | **0.1142** | 0.5486          | 0.3092          | 0.3268          | 0.6862          | 0.3053          | 0.3813          | 0.4691          | 0.2152          | 0.2516          | 29001         |
| **AMF***      | 0.2029          | 0.1259        | 0.6441          | 0.2967          | 0.3131          | 0.6668          | 0.2918          | 0.3520          | 0.4589          | 0.2142          | 0.2405          | 122084        |
| **ABA**       | 0.2210          | 0.1188        | 0.5920          | 0.3228          | 0.3342          | **0.7002** | 0.3162          | 0.3712          | 0.4775          | 0.2296          | 0.2570          | 12570         |
| **ABO**       | 0.2180          | 0.1197        | 0.5894          | 0.3228          | 0.3330          | 0.6905          | 0.3161          | 0.3686          | 0.4651          | 0.2309          | 0.2569          | 13851         |
| **MLHT**      | 0.1431          | 0.1755        | 0.2835          | 0.2430          | 0.2518          | 0.3041          | 0.2351          | 0.2549          | 0.2516          | 0.1531          | 0.1160          | **1455** |
| **MLHTPS**    | 0.1242          | 0.1374        | 0.4613          | 0.1907          | 0.1957          | 0.4267          | 0.1868          | 0.2159          | 0.3138          | 0.1365          | 0.1387          | 29508         |
| **iSOUPT**    | 0.2184          | 0.1203        | 0.5908          | 0.3260          | 0.3351          | 0.6888          | 0.3203          | 0.3779          | 0.4585          | 0.2291          | 0.2562          | 6472          |
| **MLHAT**     | **0.2670** | 0.1295        | **0.6499** | **0.4900** | **0.4802** | 0.5490          | **0.4551** | **0.4929** | **0.4828** | **0.3630** | **0.3886** | 16119         |


## Reproductible experimentation

All the experimentation has been run in Python, using for the comparative analysis the implementations available in River of the main classification algorithms in multi-label data streams, with the default parameters proposed by the authors. The methods used, their parameters and the reference implementation used are detailed below.

| Algorithm | Family | Parameters | Implementation reference |
|:-|:-|:-|:-|
| MLHAT | Multilabel tree | `grace_period=200 delta=1e-05 tau=0.05 split_criterion=MLInfoGain leaf_prediction=MLNBA splitter=MLGaussian nb_threshold=0 drift_detector=ADWIN drift_window_threshold=50 switch_significance=0.05 poisson_rate=1.0 ` | This repository |
| MLHT | Multilabel tree | `grace_period=200 delta=1e-05 tau=0.05 split_criterion=MLInfoGain leaf_prediction=MLNBA splitter=MLGaussian nb_threshold=0 leaf_classifier=MajorityLabelset` | https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/multilabel/MultilabelHoeffdingTree.java|
|MLHTPS | Multilabel tree | `grace_period=200 delta=1e-05 tau=0.05 split_criterion=MLInfoGain leaf_prediction=MLNBA splitter=MLGaussian nb_threshold=0 leaf_classifier=PrunedSet(HoeffdingTree)` | https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/classifiers/multilabel/MultilabelHoeffdingTree.java|
| iSOUPT | Multilabel tree | `grace_period=200 delta=1e-5 tau=0.05 leaf_prediction=LogisticRegression model_selector_decay=0.95 splitter=TEBST min_samples_split=5`| https://riverml.xyz/0.15.0/api/tree/iSOUPTreeRegressor/ |
| BR+HT | Tree | `grace_period=200 delta=1e-05 tau=0.05 split_criterion=InfoGain leaf_prediction=NBA splitter=Gaussian nb_threshold=0` | https://riverml.xyz/0.15.0/api/tree/HoeffdingTreeClassifier/ |
| BR+EFDT | Tree | `grace_period=200 delta=1e-05 tau=0.05 split_criterion=InfoGain leaf_prediction=NBA splitter=Gaussian nb_threshold=0 min_samples_reevaluate=20` | https://riverml.xyz/0.15.0/api/tree/ExtremelyFastDecisionTreeClassifier/ |
| BR+HAT | Tree | `grace_period=200 delta=1e-05 tau=0.05 split_criterion=InfoGain leaf_prediction=NBA splitter=Gaussian nb_threshold=0 drift_detector=ADWIN drift_window_threshold=50 switch_significance=0.05`| https://riverml.xyz/0.15.0/api/tree/HoeffdingAdaptiveTreeClassifier/ |
| BR+MT | Tree | `step=0.1 loss=log use_aggregator=True dirichlet=0.5 split_pure=False` | https://riverml.xyz/0.15.0/api/forest/AMFClassifier/ |
| BR+SGT | Tree | `grace_period=200 delta=1e-5 init_pred=0.0 lambda_value=0.1 gamma=1.0` | https://riverml.xyz/0.15.0/api/tree/SGTClassifier/ |
| BR+ARF | Forest | `n_models=10 max_features=sqrt grace_period=50 delta=0.01 tau=0.05 split_criterion=InfoGain leaf_prediction=NBA splitter=Gaussian nb_threshold=0 lambda_value=6 drift_detector=ADWIN warning_detector=ADWIN ` | https://riverml.xyz/0.15.0/api/forest/ARFClassifier/ |
| BR+AMF | Forest | `n_models=10 step=1.0 use_aggregation=True dirchlet=0.5 split_pure=False` | https://riverml.xyz/0.15.0/api/forest/AMFClassifier/ |
| BR+kNN | Distance-based | `n_neighbors=5 window_size=200 min_distance_keep=0.0 weighted=True cleanup_every=0 distance_func=Euclidean softmax=False` | https://riverml.xyz/0.15.0/api/neighbors/KNNClassifier/ |
| BR+NB | Bayesian | - | https://riverml.xyz/0.15.0/api/naive-bayes/GaussianNB/ |
| BR+AMR | Rules | `n_min=200 delta=1e-5 tau=0.05 pred_model=LogisticRegression splitter=TEBST drift_detector=ADWIN fading_factor=0.99 anomaly_threshold=-0.75 m_min=30 min_samples_split=5` | https://riverml.xyz/0.15.0/api/rules/AMRules/ |
| BR+OBA | Ensemble | `model=LogisticRegression n_models=10 drift_detector=ADWIN` |  https://riverml.xyz/0.15.0/api/ensemble/ADWINBaggingClassifier/ |
| BR+OBOA | Ensemble | `model=LogisticRegression n_models=10 drift_detector=ADWIN` | https://riverml.xyz/0.15.0/api/ensemble/ADWINBoostingClassifier/ |

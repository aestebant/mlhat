# Experimental results

This folder organizes all the materials generated during the experimentation presented in the article following the prequential evaluation framework.

The [per_datatet](per_dataset/)  subfolder breaks down the result obtained by each algorithm in each dataset. Thus, there is a CSV file for each of the evaluated metrics with the following names:

* `subset_acc.csv`: Subset accuracy as  $\frac{1}{n}\sum^n_{i=0} 1 | Y_i = Z_i$
* `hamming.csv`: Hamming loss as $\frac{1}{n L} \sum^n_{i=0} \sum^L_{l=0} 1 | y_{il} \neq z_{il}$
* `example_prec.csv`: Example-based precision as $\frac{1}{n} \sum^n_{i=0} \frac{|Y_i \cup Z_i|}{|Z_i|}$
* `example_rec.csv`: Example-based recall as $\frac{1}{n} \sum^n_{i=0} \frac{|Y_i \cup Z_i|}{|Y_i|}$
* `example_f1.csv`: Example-based F1 as $\frac{1}{n} \sum^n_{i=0} \frac{2 |Y_i \cup Z_i|}{|Y_i| + |Z_i|}$
* `micro_prec.csv`: Micro-averaged precision as $\frac{\sum^L_{l=0} tp_l}{\sum^L_{l=0} tp_l + \sum^L_{l=0} fp_l}$
* `micro_rec.csv`: Micro-averaged recall as $\frac{\sum^L_{l=0} tp_l}{\sum^L_{l=0} tp_l + \sum^L_{l=0} fn_l}$
* `micro_f1.csv`: Micro-averaged F1 as the harmonic mean of the precision and recall in an equivalent way to the example-based F1 defined above.
* `macro_prec.csv`: Macro-averaged precision as $\frac{1}{L} \sum^L_{l=0} \frac{tp_l}{tp_l + fp_l}$
* `macro_rec.csv`: Macro-averaged recall as $\frac{1}{L} \sum^L_{l=0} \frac{tp_l}{tp_l + fn_l}$
* `macro_f1.csv`: Macro-averaged F1 as the harmonic mean of the precision and recall in an equivalent way to the example-based F1 defined above.
* `time_s.csv`: Time in seconds needed to process the whole data stream with prequential evaluation.

The rows show the datasets included in the experimentation and the columns show the algorithms. The intersection is the result for the given metric following the prequential evaluation scheme. For example, for the micro-averaged F1 score the following results are obtained:

| Dataset      | MLHAT          | KNN            | NB             | AMR            | HT           | HAT            | EFDT           | SGT   | MT    | ARF            | AMF            | ABALR          | ABOLR | MLBELS         | MLHT  | MLHTPS | iSOUPT         |
|--------------|----------------|----------------|----------------|----------------|--------------|----------------|----------------|-------|-------|----------------|----------------|----------------|-------|----------------|-------|--------|----------------|
| Flags        | 0.586          | 0.642          | 0.642          | 0.575          | **0.710** | 0.695          | **0.710** | 0.650 | 0.622 | 0.703          | 0.641          | 0.587          | 0.581 | 0.533          | 0.617 | 0.630  | 0.575          |
| WaterQuality | 0.551          | **0.561** | 0.549          | 0.552          | 0.420        | 0.436          | 0.459          | 0.352 | 0.357 | 0.533          | 0.509          | 0.549          | 0.553 | 0.465          | 0.258 | 0.382  | 0.547          |
| Emotions     | 0.445          | 0.484          | **0.629** | 0.319          | 0.573        | 0.575          | 0.546          | 0.389 | 0.033 | 0.547          | 0.413          | 0.300          | 0.309 | 0.558          | 0.342 | 0.537  | 0.319          |
| VirusGO      | 0.694          | 0.708          | 0.120          | 0.613          | 0.266        | 0.533          | 0.265          | 0.344 | 0.516 | 0.425          | **0.849** | 0.601          | 0.618 | 0.584          | 0.344 | 0.228  | 0.603          |
| Birds        | 0.133          | 0.164          | 0.000          | **0.176** | 0.000        | 0.000          | 0.122          | 0.081 | 0.065 | 0.115          | 0.108          | 0.174          | 0.154 | 0.156          | 0.000 | 0.000  | 0.175          |
| Yeast        | 0.598          | 0.590          | 0.545          | 0.488          | 0.599        | 0.597          | 0.589          | 0.452 | 0.477 | 0.574          | 0.485          | 0.486          | 0.489 | **0.613** | 0.536 | 0.583  | 0.503          |
| Scene        | **0.936** | 0.803          | 0.579          | 0.908          | 0.502        | 0.591          | 0.715          | 0.269 | 0.651 | 0.817          | 0.845          | 0.905          | 0.904 | 0.783          | 0.187 | 0.604  | 0.879          |
| Gnegative    | 0.937          | 0.730          | 0.639          | 0.943          | 0.656        | 0.663          | 0.670          | 0.190 | 0.737 | 0.763          | 0.879          | 0.942          | 0.941 | 0.699          | 0.389 | 0.640  | **0.945** |
| Plant        | 0.770          | 0.319          | 0.309          | **0.875** | 0.291        | 0.305          | 0.364          | 0.127 | 0.459 | 0.489          | 0.678          | 0.859          | 0.856 | 0.413          | 0.282 | 0.221  | **0.875** |
| Human        | 0.862          | 0.596          | 0.296          | **0.873** | 0.416        | 0.515          | 0.466          | 0.165 | 0.320 | 0.709          | 0.570          | 0.870          | 0.872 | 0.663          | 0.302 | 0.256  | 0.849          |
| Yelp         | **0.825** | 0.653          | 0.485          | 0.755          | 0.614        | 0.663          | 0.701          | 0.458 | 0.480 | 0.797          | 0.669          | 0.784          | 0.785 | 0.697          | 0.512 | 0.587  | 0.773          |
| Medical      | 0.584   | 0.502          | 0.003          | 0.315          | 0.520        | 0.534          | 0.636          | 0.087 | 0.287 | 0.217          | 0.574          | 0.215          | 0.200 | **0.684** | 0.240 | 0.003  | 0.324          |
| Eukaryote    | 0.892          | 0.731          | 0.239          | 0.891          | 0.544        | 0.685          | 0.646          | 0.073 | 0.486 | 0.811          | 0.655          | **0.898** | 0.897 | 0.776          | 0.270 | 0.242  | 0.876          |
| Slashdot     | **0.458** | 0.156          | 0.000          | 0.030          | 0.106        | 0.172          | 0.244          | 0.064 | 0.012 | 0.140          | 0.260          | 0.013          | 0.010 | 0.417          | 0.142 | 0.009  | 0.074          |
| Enron        | 0.436          | 0.374          | 0.436          | 0.499          | 0.429        | 0.431          | 0.442          | 0.129 | 0.303 | 0.448          | 0.451          | 0.510          | 0.511 | **0.520** | 0.297 | 0.276  | 0.500          |
| Hypercube    | 0.993          | 0.994          | 0.963          | 0.982          | 0.917        | 0.962          | 0.931          | 0.028 | 0.995 | 0.993          | **0.995** | 0.988          | 0.989 | 0.987          | 0.785 | 0.806  | 0.933          |
| Langlog      | **0.133** | 0.074          | 0.000          | 0.054          | 0.000        | 0.000          | 0.099          | 0.015 | 0.011 | 0.003          | 0.012          | 0.035          | 0.036 | 0.127          | 0.002 | 0.048  | 0.061          |
| Stackex      | 0.257   | 0.127          | 0.001          | 0.111          | 0.011        | 0.025          | 0.162          | 0.018 | 0.021 | 0.043          | 0.050          | 0.129          | 0.129 | **0.310** | 0.146 | 0.000  | 0.133          |
| Tmc          | 0.638   | 0.487          | 0.322          | 0.618          | 0.529        | 0.525          | 0.463          | 0.325 | 0.424 | 0.516          | 0.503          | 0.629          | 0.621 | **0.641** | 0.197 | 0.439  | 0.607          |
| Ohsumed      | **0.445** | 0.065          | 0.301          | 0.196          | 0.359        | 0.385          | 0.325          | 0.186 | 0.026 | 0.147          | 0.006          | 0.257          | 0.247 | 0.382          | 0.137 | 0.308  | 0.229          |
| D20ng        | **0.662** | 0.146          | 0.001          | 0.147          | 0.449        | 0.425          | 0.471          | 0.191 | 0.121 | 0.190          | 0.121          | 0.338          | 0.306 | 0.531          | 0.083 | 0.289  | 0.273          |
| Mediamill    | **0.661** | 0.600          | 0.170          | 0.553          | 0.511        | 0.519          | 0.518          | 0.223 | 0.454 | 0.583          | 0.519          | 0.558          | 0.557 | 0.555          | 0.425 | 0.367  | 0.521          |
| Corel5k      | 0.284   | 0.138          | 0.018          | 0.025          | 0.078        | 0.141          | 0.134          | 0.010 | 0.153 | **0.316** | 0.221          | 0.020          | 0.022 | 0.214          | 0.043 | 0.019  | 0.021          |
| Bibtex       | **0.347** | 0.030          | 0.000          | 0.203          | 0.216        | 0.224          | 0.218          | 0.091 | 0.111 | 0.154          | 0.141          | 0.182          | 0.183 | 0.272          | 0.041 | 0.090  | 0.196          |
| Nuswidec     | **0.403** | 0.394          | 0.331          | 0.088          | 0.236        | 0.282          | 0.243          | -     | 0.017 | 0.261          | -              | 0.075          | 0.075 | -              | 0.011 | 0.214  | 0.121          |
| Imdb         | 0.226          | 0.169          | 0.187          | 0.025          | 0.047        | 0.054          | 0.085          | 0.006 | 0.000 | 0.033          | -              | 0.054          | 0.058 | **0.353** | 0.242 | 0.013  | 0.038          |
| Nuswideb     | **0.423** | 0.373          | 0.105          | 0.111          | 0.175        | 0.275          | 0.250          | -     | -     | 0.278          | -              | 0.110          | 0.107 | -              | 0.010 | 0.109  | 0.119          |
| YahooSociety | 0.435          | 0.232          | 0.349          | 0.394          | 0.243        | 0.290          | 0.250          | -     | 0.108 | 0.248          | -              | **0.451** | 0.435 | 0.442          | 0.369 | 0.028  | 0.420          |
| Eurlex       | 0.551          | 0.283          | 0.123          | 0.478          | 0.405        | 0.406          | 0.262          | 0.037 | 0.122 | 0.376          | -              | **0.600** | 0.579 | -              | 0.074 | 0.021  | 0.546          |
| SynTreeSud   | 0.308          | 0.274          | 0.063          | 0.160          | 0.062        | 0.148          | 0.206          | 0.007 | 0.001 | 0.195          | 0.004          | 0.168          | 0.168 | **0.449** | 0.329 | 0.001  | 0.127          |
| SynRBFSud    | 0.292          | 0.251          | 0.209          | 0.222          | 0.294        | 0.351          | 0.257          | 0.008 | 0.040 | **0.368** | 0.166          | 0.041          | 0.040 | 0.214          | 0.107 | 0.085  | 0.190          |
| SynHPSud     | 0.364  | 0.339          | 0.011          | 0.293          | 0.111        | 0.290          | 0.275          | 0.009 | 0.000 | 0.291          | 0.001          | 0.292          | 0.293 | **0.460** | 0.295 | 0.000  | 0.250          |
| SynTreeGrad  | 0.333          | 0.297          | 0.130          | 0.138          | 0.067        | 0.103          | 0.180          | 0.010 | 0.000 | 0.192          | 0.052          | 0.165          | 0.171 | **0.469** | 0.348 | 0.000  | 0.109          |
| SynRBFGrad   | 0.219          | 0.180          | 0.196          | 0.171          | 0.246        | 0.271          | 0.207          | 0.008 | 0.010 | **0.285** | 0.083          | 0.013          | 0.013 | 0.194          | 0.103 | 0.068  | 0.131          |
| SynHPGrad    | 0.324   | 0.279          | 0.058          | 0.183          | 0.089        | 0.162          | 0.211          | 0.007 | 0.000 | 0.185          | 0.011          | 0.166          | 0.165 | **0.435** | 0.282 | 0.001  | 0.122          |
| SynTreeInc   | 0.311          | 0.270          | 0.190          | 0.143          | 0.086        | 0.139          | 0.182          | 0.010 | 0.001 | 0.173          | 0.005          | 0.164          | 0.165 | **0.443** | 0.316 | 0.000  | 0.105          |
| SynRBFInc    | 0.232          | 0.182          | 0.188          | 0.181          | 0.248        | 0.277          | 0.209          | 0.022 | 0.038 | **0.310** | 0.122          | 0.020          | 0.020 | 0.199          | 0.147 | 0.079  | 0.148          |
| SynHPInc     | 0.258          | 0.189          | 0.047          | 0.030          | 0.046        | 0.046          | 0.106          | 0.007 | 0.000 | 0.033          | 0.000          | 0.027          | 0.028 | **0.393** | 0.291 | 0.000  | 0.036          |
| SynTreeRec   | 0.387          | 0.389          | 0.214          | 0.407          | 0.283        | 0.402          | 0.368          | 0.007 | 0.108 | 0.383          | 0.108          | 0.410          | 0.409 | **0.480** | 0.334 | 0.000  | 0.372          |
| SynRBFRec    | 0.242          | 0.193          | 0.216          | 0.243          | 0.352        | **0.374** | 0.232          | 0.029 | 0.035 | 0.308          | 0.143          | 0.017          | 0.016 | 0.205          | 0.126 | 0.135  | 0.294          |
| SynHPRec     | 0.468  | 0.383          | 0.299          | 0.317          | 0.316        | 0.335          | 0.353          | 0.205 | 0.291 | 0.341          | 0.274          | 0.315          | 0.314 | **0.552** | 0.385 | 0.273  | 0.287          |
|||||||||||||||||||
| *Average* | **0.486**   | 0.374 | 0.248  | 0.373 | 0.318 | 0.361 | 0.360 | 0.139  | 0.222  | 0.373 | 0.337  | 0.364 | 0.362 | 0.470 | 0.252  | 0.210  | 0.371 |
| *Ranking* | **3.439** | 6.878 | 11.439 | 7.451 | 9.354 | 7.707 | 7.183 | 14.659 | 13.232 | 6.902 | 10.866 | 7.951 | 8.049 | 5.293 | 10.854 | 13.634 | 8.110 |

From these data, the distribution of average results per dataset for the subset accuracy, example-based F1, micro-averaged F1, and macro-averaged F1 metrics is obtained below. In these results it can be seen how MLHAT is better than the mean in 36 of the 41 datasets, with 30 in the first quartile.

![boxplot summary](paper_boxplot.jpg)

## Statistical tests

The raw measures per algorithm and dataset have been used to find statistically significant differences between the studied methods. Specifically we use the Friedman test of the ranks of the metrics and the post-hoc Bonferroni-Dumm test to find the pair of groups which are significantly different.

We have use R and its [scmamp](https://github.com/b0rxa/scmamp) library in the following way:

```R
library(scmamp)

# Load raw data
rd <- read.csv(csv_path)
nAlgorithms <- ncol(rd)-1
nDatasets <- nrow(rd)
rdm <- rd[, 2: (nAlgorithms+1)]
# Friedman test. Multiple comparison
alpha <- 0.01
friedman <- friedmanTest(data=rdm,alpha=alpha)
if(friedman$p.value < alpha) {
    # Post-Hoc test
    test <- postHocTest(data=rdm, test='friedman', correct='bonferroni', alpha=alpha, use.rank=FALSE, sum.fun=mean)
}
```

For each studied metric, the ranks, Friedman's $\chi^{2}$, and the *p*-value are the following:

| **Algorithm** | Su.Acc       | H.Loss         | Ex.F1          | Ex.Pre         | Ex.Rec         | Mi.F1          | Mi.Prec        | Mi.Rec         | Ma.F1          | Ma.Prec        | Ma.Rec         | Time (s)       |
|---------------|--------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| **MLHAT**     | **4.000** | 9.024          | **3.024** | 6.878          | **2.902** | **3.439** | 9.415          | **3.098** | **2.463** | 6.341          | **2.659** | 7.049          |
| **KNN**       | 5.463        | 9.439          | 7.049          | 9.585          | 7.585          | 6.878          | 9.780          | 7.512          | 6.415          | 7.268          | 7.293          | 13.488         |
| **NB**        | 12.512       | 12.573         | 10.695         | 12.305         | 8.988          | 11.439         | 13.000         | 8.671          | 10.683         | 12.976         | 8.841          | 3.366          |
| **AMR**       | 7.549        | 6.037          | 7.817          | 7.524          | 8.232          | 7.451          | 5.378          | 8.207          | 7.524          | 6.305          | 8.159          | 8.854          |
| **HT**        | 9.220        | 7.890          | 9.598          | 9.305          | 9.951          | 9.354          | 8.939          | 9.951          | 9.439          | 9.780          | 9.902          | 7.634          |
| **HAT**       | 7.720        | 7.622          | 7.805          | 8.829          | 8.341          | 7.707          | 8.707          | 8.366          | 7.805          | 8.634          | 8.195          | 9.049          |
| **EFDT**      | 8.366        | 10.207         | 7.305          | 9.354          | 7.463          | 7.183          | 10.061         | 7.488          | 6.439          | 9.146          | 7.293          | 10.927         |
| **SGT***      | 15.707       | 14.341         | 14.537         | 11.488         | 12.098         | 14.659         | 15.610         | 11.927         | 12.780         | 15.000         | 10.854         | 15.634         |
| **MT***       | 13.159       | 9.037          | 13.622         | 9.256          | 13.841         | 13.232         | 9.232          | 13.915         | 12.890         | 11.890         | 13.256         | 12.524         |
| **ARF**       | 6.902        | 5.293          | 7.195          | 8.049          | 7.927          | 6.902          | 5.024          | 7.854          | 7.122          | **5.244** | 8.293          | 11.024         |
| **AMF***      | 10.744       | 7.524          | 11.159         | 7.207          | 11.695         | 10.866         | 7.037          | 11.671         | 10.695         | 9.085          | 11.232         | 16.110         |
| **ABALR**     | 7.195        | **4.976** | 8.341          | **6.463** | 8.805          | 7.951          | **3.512** | 8.805          | 8.732          | 5.488          | 9.098          | 5.683          |
| **ABOLR**     | 7.366        | 5.756          | 8.293          | 6.780          | 8.634          | 8.049          | 4.146          | 8.732          | 8.463          | 5.976          | 8.732          | 6.195          |
| **MLBELS***   | 9.268        | 12.829         | 4.610          | 9.024          | 4.220          | 5.293          | 12.073         | 4.341          | 6.317          | 8.902          | 5.659          | 12.195         |
| **MLHT**      | 7.756        | 14.415         | 10.024         | 13.537         | 10.098         | 10.854         | 14.561         | 10.293         | 13.049         | 12.585         | 11.049         | **1.000** |
| **MLHTPS**    | 12.329       | 10.390         | 13.354         | 10.427         | 13.402         | 13.634         | 11.683         | 13.476         | 13.390         | 11.488         | 13.329         | 7.854          |
| **iSOUPT**    | 7.744        | 5.646          | 8.573          | 6.988          | 8.817          | 8.110          | 4.841          | 8.695          | 8.793          | 6.890          | 9.159          | 4.415          |
||||||||||||||
| **Friedman's  $\chi^{2}$** | 231.48  | 248.68  | 239.21  | 107.07    | 206.71  | 238.17  | 342.55  | 202.84  | 227.82  | 220.85  | 176.03  | 462.72  |
| $p$-value | 2.2e-16 | 2.2e-16 | 2.2e-16 | 1.665e-15 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 2.2e-16 |

And the critical distance plots are available under the [critical_distances](critical_distances/) subfolder. For example, the subset accuracy post-hoc test shows that there are five statistically different groups at 99% of confidence, with MLHAT in the first one:

![cd for subset acc](critical_distances/st_subset_acc.jpg)

## Performance evolution across the stream

Finally, the evolution of the main metrics during the prequential evaluation of MLHAT against the main state-of-the-art methods included in the experimentation is shown. For example the figure below shows the evolution on Yelp. For the rest of the datasets included in this work, see the [evolution](evolution/) subfolder.

![](evolution/paper_evolmicrof1_Yelp.jpg) ![](evolution/paper_evoltime_Yelp.jpg) ![](evolution/paper_evolmb_Yelp.jpg)
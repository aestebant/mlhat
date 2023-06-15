# Experimental results

This folder organizes all the materials generated during the experimentation presented in the article following the prequential framework.

The [per_datatet](per_dataset/)  subfolder breaks down the result obtained by each algorithm in each dataset. Thus, there is a CSV file for each of the evaluated metrics with the following names:
* `subset_acc.csv`: Subset accuracy as  $\frac{1}{n}\sum^n_{i=0} 1 | Y_i = Z_i$
* `hamming.csv`: Hamming loss as $\frac{1}{n L} \sum^n_{i=0} \sum^L_{l=0} 1 | y_{il} \neq z_{il}$
* `example_acc.csv`: Example-based accuracy as $\frac{1}{n} \sum^n_{i=0} \frac{|Y_i \cap Z_i|}{|Y_i \cup Z_i|}$
* `example_prec.csv`: Example-based precision as $\frac{1}{n} \sum^n_{i=0} \frac{|Y_i \cup Z_i|}{|Z_i|}$
* `example_rec.csv`: Example-based recall as $\frac{1}{n} \sum^n_{i=0} \frac{|Y_i \cup Z_i|}{|Y_i|}$
* `example_f1.csv`: Example-based F1 as $\frac{1}{n} \sum^n_{i=0} \frac{2 |Y_i \cup Z_i|}{|Y_i| + |Z_i|}$
* `micro_prec.csv`: Micro-averaged precision as $\frac{\sum^L_{l=0} tp_l}{\sum^L_{l=0} tp_l + \sum^L_{l=0} fp_l}$
* `micro_rec.csv`: Micro-averaged recall as $\frac{\sum^L_{l=0} tp_l}{\sum^L_{l=0} tp_l + \sum^L_{l=0} fn_l}$
* `micro_f1.csv`: Micro-averaged F1 as the harmonic mean of the precision and recall in an equivalent way to the example-based F1 defined above.
* `macro_prec.csv`: Macro-averaged precision as $\frac{1}{L} \sum^L_{l=0} \frac{tp_l}{tp_l + fp_l}$
* `macro_rec.csv`: Macro-averaged recall as $\frac{1}{L} \sum^L_{l=0} \frac{tp_l}{tp_l + fn_l}$
* `macro_f1.csv`: Macro-averaged F1 as the harmonic mean of the precision and recall in an equivalent way to the example-based F1 defined above.


The rows show the datasets included in the experimentation and the columns show the algorithms. The intersection is the result for the given metric following the prequential evaluation scheme. For example, for the subset accuracy metric the following results are obtained:

| **Dataset** | **kNN** | **NB** | **AMR** | **HT** | **HAT** | **EFDT** | **SGT** | **MT** | **OBA** | **OBOA** | **ARF** | **AMF** | **MLHT** | **MLHTPS** | **iSOUPT** | **MLHAT** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Flags | 0.072 | 0.077 | 0.021 | 0.114 | 0.093 | 0.114 | 0.010 | 0.031 | 0.026 | 0.031 | 0.124 | 0.072 | **0.129** | 0.026 | 0.021 | 0.036 |
| Chd | 0.133 | 0.128 | 0.171 | 0.156 | 0.129 | 0.156 | 0.072 | 0.118 | **0.178** | 0.174 | 0.110 | 0.131 | 0.144 | 0.124 | 0.165 | **0.178** |
| WaterQuality | 0.020 | 0.003 | **0.021** | 0.002 | 0.002 | 0.006 | 0.007 | 0.004 | 0.020 | **0.021** | **0.021** | **0.021** | 0.009 | 0.004 | **0.021** | 0.020 |
| Emotions | 0.151 | 0.178 | 0.059 | 0.195 | 0.193 | 0.181 | 0.042 | 0.002 | 0.062 | 0.064 | 0.190 | 0.102 | 0.124 | **0.222** | 0.059 | 0.086 |
| VirusGO | 0.551 | 0.077 | 0.445 | 0.159 | 0.392 | 0.159 | 0.034 | 0.314 | 0.425 | 0.435 | 0.295 | **0.730** | 0.256 | 0.130 | 0.430 | 0.459 |
| Birds | 0.453 | 0.448 | 0.294 | 0.450 | 0.450 | 0.454 | 0.304 | 0.451 | 0.397 | 0.349 | **0.465** | 0.437 | 0.453 | 0.451 | 0.300 | 0.352 |
| Yeast | 0.145 | 0.088 | 0.015 | 0.072 | 0.091 | 0.057 | 0.010 | 0.014 | 0.015 | 0.015 | 0.071 | 0.015 | 0.089 | **0.185** | 0.035 | 0.050 |
| Scene | 0.762 | 0.278 | 0.869 | 0.299 | 0.413 | 0.574 | 0.119 | 0.500 | 0.867 | 0.866 | 0.736 | 0.753 | 0.166 | 0.522 | 0.833 | **0.900** |
| Gnegative | 0.695 | 0.541 | 0.915 | 0.578 | 0.576 | 0.579 | 0.130 | 0.620 | 0.915 | 0.914 | 0.692 | 0.814 | 0.380 | 0.605 | 0.917 | **0.930** |
| Plant | 0.269 | 0.152 | **0.823** | 0.179 | 0.180 | 0.256 | 0.013 | 0.327 | 0.805 | 0.801 | 0.380 | 0.552 | 0.279 | 0.142 | **0.823** | 0.809 |
| Cal500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Human | 0.507 | 0.121 | 0.779 | 0.261 | 0.357 | 0.284 | 0.017 | 0.155 | 0.776 | **0.779** | 0.573 | 0.371 | 0.229 | 0.143 | 0.749 | 0.767 |
| Genbase | 0.098 | 0.023 | 0.035 | 0.023 | 0.011 | 0.023 | 0.000 | 0.023 | 0.054 | 0.035 | 0.024 | 0.023 | **0.250** | 0.024 | 0.035 | 0.027 |
| Yelp | 0.430 | 0.190 | 0.479 | 0.273 | 0.340 | 0.402 | 0.193 | 0.207 | 0.543 | 0.553 | 0.552 | 0.370 | 0.245 | 0.172 | 0.515 | **0.626** |
| Medical | 0.328 | 0.002 | 0.146 | 0.300 | 0.327 | **0.429** | 0.000 | 0.145 | 0.083 | 0.081 | 0.094 | 0.354 | 0.157 | 0.002 | 0.152 | 0.222 |
| Eukaryote | 0.640 | 0.111 | 0.798 | 0.389 | 0.526 | 0.482 | 0.021 | 0.275 | **0.813** | 0.811 | 0.695 | 0.490 | 0.202 | 0.130 | 0.784 | 0.810 |
| Slashdot | 0.139 | 0.000 | 0.017 | 0.045 | 0.086 | 0.119 | 0.000 | 0.007 | 0.007 | 0.005 | 0.076 | **0.160** | 0.140 | 0.005 | 0.043 | 0.074 |
| Enron | 0.105 | 0.027 | 0.096 | 0.090 | 0.087 | 0.072 | 0.000 | 0.064 | 0.099 | 0.101 | 0.076 | 0.071 | **0.112** | 0.061 | 0.095 | 0.106 |
| Hypercube | 0.994 | 0.930 | 0.977 | 0.855 | 0.946 | 0.888 | 0.001 | 0.994 | 0.986 | 0.987 | 0.993 | **0.995** | 0.785 | 0.806 | 0.901 | 0.970 |
| Hypersphere | 0.512 | 0.057 | 0.510 | 0.329 | 0.590 | 0.483 | 0.004 | 0.081 | 0.539 | 0.541 | 0.561 | 0.108 | 0.288 | 0.265 | 0.307 | **0.603** |
| Langlog | **0.180** | 0.141 | 0.164 | 0.141 | 0.141 | 0.174 | 0.000 | 0.140 | 0.154 | 0.157 | 0.136 | 0.140 | 0.137 | 0.165 | 0.164 | 0.177 |
| Stackex | 0.026 | 0.002 | 0.017 | 0.005 | 0.006 | 0.025 | 0.001 | 0.003 | 0.021 | 0.020 | 0.003 | 0.004 | **0.027** | 0.002 | 0.024 | **0.027** |
| Tmc | 0.134 | 0.061 | 0.227 | 0.152 | 0.139 | 0.096 | 0.091 | 0.104 | 0.247 | 0.242 | 0.158 | 0.137 | 0.086 | 0.155 | 0.230 | **0.272** |
| Ohsumed | 0.016 | 0.061 | 0.086 | 0.142 | 0.149 | 0.107 | 0.073 | 0.010 | 0.105 | 0.101 | 0.058 | 0.003 | 0.084 | **0.166** | 0.103 | 0.158 |
| D20ng | 0.089 | 0.000 | 0.075 | 0.274 | 0.253 | 0.280 | 0.111 | 0.064 | 0.197 | 0.177 | 0.104 | 0.065 | 0.083 | 0.253 | 0.156 | **0.361** |
| Mediamill | **0.169** | 0.000 | 0.102 | 0.058 | 0.064 | 0.046 | 0.029 | 0.048 | 0.110 | 0.110 | 0.126 | 0.086 | 0.053 | 0.024 | 0.078 | 0.155 |
| Corel5k | 0.014 | 0.000 | 0.000 | 0.001 | 0.007 | 0.003 | 0.000 | 0.004 | 0.000 | 0.000 | **0.025** | 0.012 | 0.009 | 0.000 | 0.000 | 0.003 |
| Corel16k | 0.048 | 0.001 | 0.007 | 0.003 | 0.022 | 0.012 | 0.000 | 0.009 | 0.002 | 0.001 | **0.078** | 0.020 | 0.018 | 0.006 | 0.002 | 0.022 |
| Bibtex | 0.003 | 0.000 | 0.057 | 0.053 | 0.057 | 0.036 | 0.032 | 0.052 | 0.059 | 0.059 | 0.046 | 0.044 | 0.063 | 0.043 | 0.067 | **0.093** |
| Nuswidec | 0.254 | 0.132 | 0.225 | 0.215 | 0.217 | 0.117 | 0.104 | 0.221 | 0.229 | 0.229 | 0.248 | 0.209 | 0.222 | 0.206 | 0.223 | **0.279** |
| YahooSociety | 0.144 | 0.258 | 0.202 | 0.133 | 0.156 | 0.123 | 0.068 | 0.057 | 0.285 | 0.264 | 0.160 | 0.190 | 0.281 | 0.013 | 0.240 | **0.288** |
| Eurlex | 0.074 | 0.006 | 0.117 | 0.120 | 0.117 | 0.049 | 0.003 | 0.023 | **0.250** | 0.216 | 0.117 | 0.113 | 0.057 | 0.009 | 0.200 | 0.244 |
||
| *Average* | 0.255 | 0.128 | 0.273 | 0.190 | 0.222 | 0.212 | 0.047 | 0.158 | 0.290 | 0.285 | 0.249 | 0.237 | 0.174 | 0.158 | 0.271 | **0.316** |
| *Ranking* | 6.047 | 12.734 | 7.250 | 9.063 | 8.000 | 8.500 | 14.719 | 11.219 | 6.328 | 6.484 | 6.953 | 7.250 | 8.266 | 10.844 | 7.234 | **4.016** |

From these data, the distribution of average results per dataset for the subset accuracy, example-based F1, micro-averaged F1, and macro-averaged F1 metrics is obtained below. In these results it can be seen how MLHAT is better than the mean in 26 of the 32 datasets, with 22 in the first quartile.

![subset-acc box plot](bpd3_subset_acc.jpg)

![example-f1 box plot](bpd3_example_f1.jpg)

![micro-f1 box plot](bpd3_micro_f1.jpg)

![macro-f1 box plot](bpd3_macro_f1.jpg)


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

| **Algorithm** | **Su. Acc** | **H. Loss** | **Ex. Acc** | **Ex. Pre** | **Ex. Rec** | **Ex. F1** | **Mi. Prec** | **Mi. Rec** | **Mi. F1** | **Ma. Prec** | **Ma. Rec** | **Ma. F1** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SGT | 14.719 | 15.594 | 15.625 | 13.375 | 8.750 | 12.781 | 15.125 | 8.625 | 13.016 | 12.563 | 6.875 | 9.375 |
| MLHT | 8.266 | 13.625 | 13.625 | 11.531 | 10.828 | 10.734 | 13.906 | 11.031 | 11.391 | 14.703 | 12.000 | 13.250 |
| MLHTPS | 10.844 | 11.047 | 11.031 | 10.688 | 11.984 | 11.719 | 10.781 | 12.172 | 12.188 | 12.578 | 12.281 | 12.469 |
| NB | 12.734 | 12.234 | 12.281 | 12.734 | 10.016 | 11.422 | 12.875 | 9.719 | 11.859 | 11.875 | 9.953 | 10.938 |
| MT | 11.219 | 9.547 | 9.516 | 10.281 | 12.516 | 12.094 | 8.313 | 12.578 | 11.625 | 10.000 | 11.672 | 11.078 |
| HT | 9.063 | 7.891 | 7.891 | 9.250 | 9.563 | 9.266 | 9.391 | 9.563 | 9.141 | 9.828 | 9.625 | 9.531 |
| HAT | 8.000 | 8.063 | 8.016 | 8.031 | 8.094 | 7.797 | 9.344 | 8.078 | 7.531 | 8.688 | 8.281 | 8.031 |
| EFDT | 8.500 | 9.234 | 9.250 | 7.594 | 7.219 | 7.203 | 9.656 | 7.125 | 7.203 | 8.578 | 7.281 | 7.063 |
| AMF | 8.344 | 6.172 | 6.453 | 7.313 | 9.219 | 8.406 | 5.641 | 9.172 | 8.156 | 6.375 | 8.438 | 7.750 |
| kNN | 6.047 | 8.969 | 8.953 | 7.594 | 7.453 | 6.719 | 9.594 | 7.547 | 6.656 | 5.594 | 7.219 | 6.625 |
| iSOUPT | 7.234 | 6.609 | 6.547 | 6.906 | 7.188 | 6.969 | 5.719 | 7.188 | 6.875 | 7.797 | 7.813 | 7.688 |
| AMR | 7.250 | 6.922 | 6.875 | 6.797 | 7.016 | 6.672 | 6.156 | 6.953 | 6.594 | 6.484 | 7.000 | 6.656 |
| ARF | 6.953 | 4.906 | 4.828 | 6.828 | 7.781 | 7.234 | 4.906 | 7.719 | 7.063 | 5.141 | 8.266 | 7.516 |
| OBOA | 6.484 | 5.766 | 5.625 | 6.547 | 7.172 | 6.938 | 4.844 | 7.203 | 6.781 | 6.563 | 7.422 | 7.188 |
| OBA | 6.328 | 5.141 | 5.172 | 6.438 | 7.172 | 6.531 | **4.531** | 7.125 | 6.453 | 6.453 | 7.656 | 7.344 |
| MLHAT | **4.016** | **4.281** | **4.313** | **4.094** | **4.031** | **3.516** | 5.219 | **4.203** | **3.469** | **2.781** | **4.219** | **3.500** |
||
| *Friedman's $\chi^{2}$* | 156.14 | 226.44 | 227.5 | 138.91 | 96.754 | 140.51 | 248.11 | 97.893 | 156.34 | 220.96 | 97.11 | 130.14 |
| *Friedman's p-value* | 2.2e-16 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 5.362e-14 | 2.2e-16 | 2.2e-16 | 3.264e-14 | 2.2e-16 | 2.2e-16 | 4.596e-14 | 2.2e-16 |

And the critical distance plots are available under the [critical_distances](critical_distances/) subfolder. For example, the subset accuracy post-hoc test shows that there are six statistically different groups at 99% of confidence, with MLHAT in the first one:

![cd for subset acc](critical_distances/t_subset_acc.jpg)


## Concept-drift adaptation

Finally, the evolution of the main multi-label classification metrics during the prequential evaluation of MLHAT against the main state-of-the-art methods included in the experimentation is shown. For example the figure below shows the evolution on Yelp. For the rest of the datasets included in this work, see the [evolution](evolution/) subfolder.

![prequential evol in yelp](evolution/p_Yelp.jpg)
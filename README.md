# NN-GLS-article-code
Available code for the [NN-GLS paper](https://arxiv.org/pdf/2304.09157.pdf)
=======
This is the vailable code for the NN-GLS paper. To run the code, \
Please download the folders in https://github.com/WentaoZhan1998/NN-GLS-article-code to the working directory.

## Folder description:
* Folder *est-pred-PI* contains code producing the majority of the simulation result,
including estimation, prediction (interval) results.

* Folder *large-sample* containts code running the large sample experiments, including the consistency 
and running time results.

* Folder *realdata* containts code for the realdata experiments, including the visualization, assumption check, 
and prediction results.

## File description:
* dim1.py/dim5.py/dim15.py: Run the estimation and prediction (interval) experiments with the true functions 
$f_0 = f_1, f_2, f_3$ respectively.

* running_time.py: Obtain the overall running time for different methods, as well as the corresponding 
estimation errors (to show consistency). The sample size range from 500 to 500000.
* running_time_details.py: Obtain the running time for each step in NN-GLS algorithm.

* heatmap.py: Generate the heatmaps for the PM_{2.5} level on the U.S. landscape using interpolation.
* PDP.py: Generate PDP of a NN-GLS estimation on June 18th, 2022's data.
* realdata.py: Run the prediction (interval) experiments on the real data.
* realdata_preprocess.py: Generate the PM_{2.5} and environmental covariates by filtering on the original data.
The file works on June 18th, 2022, but the workflow applies to the other days.
* realdata_stats.py: Check the assumptions on the realdata, including the Gaussianity and covariance pattern.

* confidence-interval.py: Confidence coverage and interval score over several simulation settings.
* GAM-GLS-vs-NN-GLS.py: Run the estimation experiment on Friedman's function with different interaction powers for the 
comparison between GAM-GLS and NN-GLS.
* NN-splines-vs-NNGLS.py: Run the prediction experiment with different sample sizes for the comparison between NN-splines 
and NN-GLS.
* NNGP-mat-property.py: Experiments on the properties of NNGP precision matrix $\mathbf{Q}$, include the KL-divergence between $\mathbf{I}$ and 
$\mathbf{\Sigma}^{\frac{T}{2}}\mathbf{Q}\mathbf{\Sigma}^{\frac{1}{2}}$, and $\mathbf{Q}$'s spectral width.
* parameter-estimation: Estimate spatial parameters over several simulation settings.
* utils.py: Include most functions used in the project. It has the following modules of code:
  + The *NNGP* module consists of functions directly related with NNGP, which is a procedure to build a sparse approximation of large-sample precision.
We introduce the SparseB class to store and operate on the NNGP precision in a efficient way.
  + The *Data* module consists of functions generate and preprocess the data for the training purpose.
  + The *Models*, *Stopping* modules contains code for necessary components of NN and its training.
  + The *Functions* module contains functions studied in the manuscript.
  + The *R* module contains functions introduced from R.
  + The *Decorrelation* module consists of functions doing decorrelation operation for different data types, 
  followed by the *resample* module doing the spatial resampling (bootstrap).
  + The *Training* module contains main functions to train the NN (NN-GLS).
  + The *Evaluation* module contains functions fo the kriging prediction.

* utils_PDP.py: Introduces functions for the partial dependency plot (PDP).
* utils_pygam.py: Modifies the GAM class from pyGAM to make it compatible with GAM-GLS.

## Figures vs functions

The following functions were run to produce the figures in the manuscript and the supplementary material.
* Figure 2(a): dim1.py; Figure 2(b): GAM-GLS-vs-NN-GLS.py; Figure 2(c): dim5.py; Figure 2(d): NN-splines-vs-NNGLS.py; 
Figure 2(e, f): running_time.py
* Figure 3(a): heatmap.py; Figure 3(b): realdata.py
* Figure 4: PDP.py
* Figure S5: parameter-estimation.py
* Figure S6: NNGP-mat-property.py
* Figure S7, S8, S9, S10, S11: dim1.py & dim5.py 
* Figure S12: GAM-GLS-vs-NN-GLS.py
* Figure S13: condifence-interval.py
* Figure S14: dim1.py & dim5.py 
* Figure S15: running_time_detail.py
* Figure S16: running_time.py
* Figure S17, S18: dim1.py & dim5.py 
* Figure S19: NN-splines-vs-NNGLS.py
* Figure S20:
* Figure S21: dim15.py
* Figure S22, S23: dim1.py & dim5.py
* Figure S24:
* Figure S25, S26: dim1.py & dim5.py 
* Figure S27: realdata.py
* Figure S28: heatmap.py
* Figure S29: realdata.py
* Figure S30: realdata.py
* Figure S31: realdata_stats.py
* Figure S32: PDP.py



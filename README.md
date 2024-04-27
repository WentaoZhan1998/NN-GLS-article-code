# NN-GLS-article-code
## Available code for the NN-GLS paper (<https://arxiv.org/pdf/2304.09157.pdf>)

This is the viable code for the NN-GLS paper. To run the code, 
please [download](https://github.com/WentaoZhan1998/NN-GLS-article-code/archive/refs/heads/main.zip) the code from https://github.com/WentaoZhan1998/NN-GLS-article-code to the working directory and unzip.

Note: we have published a beta version Python package [**geospaNN**](https://github.com/WentaoZhan1998/geospaNN) as an formal implementation of NN-GLS with the [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) library. For more details, see <https://github.com/WentaoZhan1998/geospaNN>.

## Folder description:
* Folder */code/est-pred-PI* contains code producing most of the simulation result,
including estimation, prediction (interval) results.

* Folder */code/large-sample* contains code for the large sample experiments, including the consistency 
and running time results.

* Folder */code/realdata* contains code for the real data experiments, including the visualization, assumption check, 
and prediction results.

* Folder */code/visualization* contains code for some complex visualizations, basically the grouped barplots and boxplots.
Note that some figures (for example Figure S6, S9, S12, S15, S19) can be easily generated from the experiment output with several lines of code, and we omit them here.
Some figures (for example Figure S24, S28) are generated directly from the code.

## Code description:
* dim1.py/dim5.py/dim15.py: Run the estimation and prediction (interval) experiments with the true functions 
$f_0 = f_1, f_2, f_3$ respectively.

* dim1-mis1.py/dim5-mis1.py: Run the estimation and prediction (interval) experiments with the true functions 
$f_0 = f_1, f_2$ respectively, with a misspecified Matern covariance (S4.12).

* dim1-mis2.py/dim5-mis2.py: Run the estimation and prediction (interval) experiments with the true functions 
$f_0 = f_1, f_2$ respectively, with a misspecified spatial effect (S4.13).

* running_time.py: Obtain the overall running time for different methods, as well as the corresponding 
estimation errors (to show consistency). The sample size range from 500 to 500000.
* running_time_details.py: Obtain the running time for each step in NN-GLS algorithm.

* heatmap.py: Generate the heatmaps for the PM_{2.5} level on the U.S. landscape using interpolation.
* PDP.py: Generate PDP of a NN-GLS estimation on June 18th, 2022's data.
* realdata.py: Run the prediction (interval) experiments on the real data.
* realdata_preprocess.py: Generate the PM_{2.5} and environmental covariates by filtering on the original data.
The file works on June 18th, 2022, but the workflow applies to the other days.
* realdata_stats.py: Check the assumptions on the real data, including the Gaussianity and covariance pattern.

* confidence-interval.py: Confidence coverage and interval score over several simulation settings.
* GAM-GLS-vs-NN-GLS.py: Run the estimation experiment on Friedman's function with different interaction powers for the 
comparison between GAM-GLS and NN-GLS.
* NN-splines-vs-NN-GLS.py: Run the prediction experiment with different sample sizes for the comparison between NN-splines 
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
  + The *R* module contains functions inported from R.
  + The *Decorrelation* module consists of functions doing decorrelation operation for different data types, 
  followed by the *resample* module doing the spatial resampling (bootstrap).
  + The *Training* module contains main functions to train the NN (NN-GLS).
  + The *Evaluation* module contains functions fo the kriging prediction.

* est-pred-PI.Rmd: Visualization for the simulation results, including estimation MISE, prediction RMSE, interval coverage and interval score.
* parameter-estimation.Rmd: Visualization for the parameter estimation result (Figure S5).
* real-data.Rmd: Visualization of boxplots of RMSE, barplots of prediction interval coverage and score, histograms of the fitting residuals (noise).

* utils_PDP.py: Introduces functions for the partial dependency plot (PDP).
* utils_pygam.py: Modifies the GAM class from pyGAM to make it compatible with GAM-GLS.

## Figures vs code
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
* Figure S17: dim1-dense.py 
* Figure S18: dim5-dense.py 
* Figure S19: NN-splines-vs-NNGLS.py
* Figure S20: Oracle-vs-NN-GLS.py
* Figure S21: dim15.py
* Figure S22, S23: dim1-mis1.py & dim5-mis1.py
* Figure S24: fixed-surface.py
* Figure S25, S26: dim1-mis2.py & dim5-mis2.py 
* Figure S27: realdata.py
* Figure S28: heatmap.py
* Figure S29: realdata.py
* Figure S30: realdata.py
* Figure S31: realdata_stats.py
* Figure S32: PDP.py

* Visualization of Figure S7, S8, S9, S10, S11, S14, S17, S18, S21, S22, S23, S25, S26 additionally uses est-pred-PI.Rmd.
* Visualization of Figure S5 additionally uses parameter-estimation.Rmd.
* Visualization of Figure S27, S28, S29, S30, S31(a, c, e) additionally uses real-data.Rmd.

## Data availability.
* The PM2.5 data is collected from the U.S. Environmental Protection Agency (<https://www.epa.gov/outdoor-air-quality-data/download-daily-data>)
datasets for each state are collected and binded together to obtain 'pm25_2022.csv'. daily PM2.5 files are subsets of 'pm25_2022.csv' produced by
'realdata_preprocess.py'. One can skip the preprocessing and use daily files directory. 
* The meteorologica data is collected from the National Centers for Environmental Prediction’s (NCEP) North American
Regional Reanalysis (NARR) product (<https://psl.noaa.gov/data/gridded/data.narr.html>). The '.nc' (netCDF) files should be downloaded from the website 
and saved in the root directory to run 'realdata_preprocess.py'. Otherwise, one may skip the preprocessing and use covariate files directly. 
* Note that for 06/05/2019, we use the flies provided by the DeepKriging project (<https://github.com/aleksada/DeepKriging>), which come from the same resource (EPA and NARR).
* File structure: The direct input to our model are the arrays including the $n\times p$ covariates matrix $\mathbf{X}$, the $n\times 1$ response vector $\mathbf{Y}$, and the $n\times 2$ coordinates for spatial locations $\mathbf{S}$. However, to better deliver the information, it's recommended to prepare the raw data in a data frame (in *.csv* or *.xlsx* format) which contains columns for covariates, spatial coordinates, and response. Each row of the data frame stands for an observation. In python, simple function like ".values" can be used to obtain array from data frame. 

## Workflow
* Folder structure: Keep the code and folders as the same structure in this repository to make sure data can be read properly.
* Preprocessing: All data necessary for training an NN-GLS model are the $n\times p$ covariates matrix $\mathbf{X}$, the $n\times 1$ response vector $\mathbf{Y}$, and the $n\times 2$ coordinates for spatial locations $\mathbf{S}$. We recommend user scale the spatial coordinates to a uniform range (for example $[0, 10]^2$) for a universal comparison of spatial parameters. Most of the simulation example generate properly scaled samples and no preprocessing is needed. For the real data, since the PM2.5 data and the meteorological data have unmatched spatial locations, nearest neighbor average of PM2.5 are computed at the meteorological data's locations to create matched dataset (see */code/realdata/realdata.py* for the code). In our new python package [**geospaNN**](https://github.com/WentaoZhan1998/geospaNN), we provide better way to implicitly put $\mathbf{X}$, $\mathbf{Y}$, $\mathbf{S}$ into the model.
* Running: Most of the files other than the visualization code in the "code" folder can run independently. To reproduce any figure, one can directly run the corresponding file indicated in
the "Figure' vs code" section. 

The runnning times are approximate and can vary significantly based on the machine configurations. Most experiments are conducted with Intel Xeon CPU, and 8 GB RAM. 
As the only exception, the running time experiment takes up to 100 GB memory.

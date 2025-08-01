# FICA: Faster Inner Convex Approximation

This repository contains codes for our paper ***FICA: Faster Inner Convex Approximation of Chance Constrained Grid Dispatch with Decision-Coupled Uncertainty*** https://arxiv.org/abs/2506.18806.

## Installation Guideline
1. git clone this repository.
2. navigate to the folder, and run the following command to install

        conda env create -f environment.yml

3. This will create a conda environment called `fica`, activate this by `conda activate fica` (or specify it in your IDE) and you are free to run all scripts.

    **Note:** If the conda installation is slow, we strongly recommend you to update your `anaconda` to the most recent version before installing our environment.

## Two Simple Runs to See the Speedup!
1. Open the `PD.py` file, scroll down to the code blocks after `if __name__ == '__main__':`.

2. Set `method = 'FICA'` (line 507), run this script `PD.py`, which will show 

    - A plot showing the generation power (day-ahead vs adjusted by AGC) for three wind scenarios.
    - After closing the plot, the terminal will print the optimal objective value, computing time, and other information similar to:

            the objective value is ...
            The method used is FICA
            The computing time for solving the dispatch is 5.67638897895813 seconds 

3. Set `method = 'CVAR'`, re-run this script, and ***compare the computing time and optimality!***  

\
\
\
\
\
If you want to dive deeper...

## File Description
It contains the following files and folders:

### Part I: Packages and Installation
- `environment.yml`: The file that contains all the package dependencies of this conda environment.
### Part II: Data
- `data/`: The folder containing the data files (load data, wind forecast and forecasting errors).
- `GEFC2012_wind.ipynb`: The notebook for processing and generating wind forecasting error scenarios, based on GEFC2012 Bronze Medal forecasting results and groud truth.
- `WT_error_gen.py`: Functions that generate non-Gaussian and time-coupled wind forecasting error samples for traning and testing the algorithms.
### Part III: Codes for Numerical Experiments
- `PD.py`: The code for solving chance-constrained power dispatch problem.
It can also be used for testing a specific parameter setting.
- `PD_all_param_eval.py`: This code is used for evaluating all parameter settings when comparing the computing time between the proposed `FICA` and the benchmark `CVAR` in our paper. Joblib is used for parallel computing. Be aware about your available CPU cores and memory usage when specifying the njobs.
- `PD_opt_all_param_eval.py`: This is similar to `PD_all_param_eval.py`, but for testing the results for comparing the optimality between the proposed `FICA` and the `ExactLHS` method. The codes are roughly the same as `PD_all_param_eval.py`. Having a seperate file is just a more convenient way to write down different parameter combnitions (because for `ExactLHS` we can only solve small-scale problems).

***Note that***, it is very important to ensure you are not exausting your computer memory when implementing `PD_all_param_eval.py` that takes a long time. Because **memory bottleneck can make CVAR even slower that leads to biased conclusion**. To facilitate memory monitoring, we provide the following script
```
bash record_memory.sh
```
This will create a `memory_usage.csv` file that records the computer memory usage every 5 seconds. This script works on a `Ubuntu 20.04.6 LTS` operation system.

### Part IV: Results
- `PD_results_bigM100000_thread4/`: The folder containing the results of the chance-constrained grid dispatch problem with `big-M=100000` and 4 threads per optimization problem. **Note that** `big-M=100000` is only used for `ExactLHS` instead of the proposed `FICA` and `CVAR`.

### Part V: Visualization
- `PD_result_vis.ipynb`: The notebook for visualizing the results of our paper, based on the result files stored in `PD_results_bigM100000_thread4/`.
- `figure/`: The folder containing the figures generated by the above notebook.

---
If you find this repository helpful, please cite our paper:
> Yihong Zhou, Hanbin Yang, Thomas Morstyn (2025). *FICA: Faster Inner Convex Approximation of Chance Constrained Grid Dispatch with Decision-Coupled Uncertainty*. Available at arXiv https://arxiv.org/abs/2506.18806.

# 2m2c

This repository contains the simulation code and analysis for our model of basal ganglia learning that includes both the DMS and DLS. The publication that describes this model can be found here: [Distinct cortico-striatal compartments drive competition between adaptive and automatized behavior - PubMed (nih.gov)](https://pubmed.ncbi.nlm.nih.gov/36943842/)

`Barnett WH, Kuznetsov A, Lapish CC. Distinct cortico-striatal compartments drive competition between adaptive and automatized behavior. PLoS One. 2023 Mar 21;18(3):e0279841.`

These codes require the `jax`, `numpy`, `scipy`, and `matplotlib` python packages. We performed these simulations using python version 3.8.10 and `jax` version 0.2.

All figures are saved in the `figs` subdirectory.

## Usage

The file `simulations.py` performs the first set of simulations with jax running in CPU mode. The initial learning simulations must be run first in order to acquire starting synaptic weights for the follow-up behavioral tasks.

```
python simulations.py learn
```

Once this command is complete, run `simulations.py` again for each of the arguments `devalue`, `reversal`, `punish`, `reversal_prat`, and `punish_prat` in any order or concurrently. This script will produce sample simulation figures `learn.pdf`, `devalue.pdf`, `reversal.pdf`, `punish.pdf`, `reversal_prat.pdf`, and `punish_prat.pdf` in the style of Figure 2A, Figure 4A, Figure 3A, Figure 5A, Figure S1, and Figure S2 respectively. These commands must all run to completion before proceeding to the next command.

The script `get_probabilities.py` will read the synaptic weights cached from each of the simulations performed by `simulations.py` and simulate each trial many times. This script will run simulations on the GPU if available. The results of these many simulations are used to determine the probability that each agent selects the first of two actions. The script `get_probabilities.py` takes two arguments. The first indicates the behavioral session (`learn`, `devalue`, `reversal`, `punish`, `reversal_prat`, and `punish_prat`) and the second indicates the manner in which simulations are organized. We recommend using the argument `batch` here, which will execute this script for all agents for the indicated behavioral session:

```
python get_probabilities.py reversal_prat batch
````

Alternatively, the second argument can be used to specify which single agent should be processed. For example

```
python get_probabilities.py reversal_prat 0
```
will read the cached synaptic weights from the reversal session with impaired PFC coding for the first agent (agent number 0). In this mode, the command must be run once for each session (`learn`, `devalue`, `reversal`, `punish`, `reversal_prat`, and `punish_prat`) and each agent (0 to 99). Each instance may be performed concurrently or in no particular order.

We provide this code configured to sample each trial 32 times. In the manuscript, we report our analysis after sampling each trial 1000 times. We recommend that this script is tried out first in the current configuration for evaluation since it is computationally intensive. The number of samples can be altered by changing the numerical value of the `numBatches` and `batchSize` parameters in the script `get_probabilities.py`.

To process these simulations, run the following commands
```
python process_probability.py
```
Once these commands have been completed, the results can be visualized:
```
python LLR.py
python p1_plots.py
python p1_comparison.py
python steady_state_whiskers.py
python ctx_fig.py
python plot_LLR.py
python weights.py
```
The figure `p1_learn.pdf` corresponds to Figure 3C.
These figures `compare_punish.pdf` and `compare_reversal.pdf` can be found in Figure 7B and 7C.
The figures `LLR_boxplots.pdf` and `steady_state_boxplot.pdf` are in Figure 8B nd 8C.
The figure `ctx_fig.pdf` corresonds to Figure 2C and 2D.
The script `plot_LLR.py` produces several examples of our change point detection in the style of Figure 8A for the follow-up behavioral sessions.
The script `weights.py` produces the figures for Figure S3.
The script `large_punish.py` produces a figure in the style of Figure S4.

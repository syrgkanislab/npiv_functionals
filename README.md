# Inference on Strongly Identified Functionals of Weakly Identified Functions

This repository contains code to reproduce the experiments in

Bennett, Kallus, Mao, Newey, Syrgkanis, and Uehara (2025) "Inference on Strongly Identified Functionals of Weakly Identified Functions"

In order to run experiments for all scenarios, can rune the two scripts:
- run-main-synthetic-experimens.sh (this runs the experiments for our main synthetic scenario)
- run-additional-experiments.sh (this runs experiments for additional scenarios from the literature)

Results can be parsed and latex tables generated using the make-results-tables.ipynb

Hyperparameters all experiments can be changed by updating the
config files in the directory experiment-configs
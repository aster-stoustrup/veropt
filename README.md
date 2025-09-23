
# _veropt_ - the versatile optimiser

_veropt_ is a Python package that aims to make Bayesian Optimisation easy to approach, inspect and adjust. It was developed for the Versatile Ocean Simulator ([VEROS](https://veros.readthedocs.io/en/latest/)) with the aim of providing a user-friendly optimisation tool to tune ocean simulations to real world data. 

_veropt_ can be used with any optimisation problem but has been developed for expensive optimisation problems with a small amount of evaluations (~100) and will probably be most relevant in such a context.

## Installation

_veropt_ is available on the Python Package Index (PyPI) and can be installed with pip.

```bash
pip install veropt
```

Please note that veropt relies on complex packages such as pytorch and will probably benefit from living in a conda (or other) environment. Furthermore, it may be recommendable to install pytorch separately first. See their website for their current recommendations.


## Usage

Below is a simple example of setting up an optimisation problem with _veropt_. 

```python
from veropt.optimiser.practice_objectives import VehicleSafety
from veropt import bayesian_optimiser

objective = VehicleSafety()

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective
)
```

Here we use a simple practice objective called 'VehicleSafety'. When setting up your own, real optimisation problem, we recommend looking in our 'interfaces' subpackage which will help run your expensive objective function, even when using slurm on a cluster. See more about this below.

With a simple practice objective like this, everything is already set up, and we can simply step forward the optimisation with,

```python
optimiser.run_optimisation_step()
```

and we'll receive a message like,

```
Optimisation running in initial mode at step 1 out of 12 
Best objective value(s): [1691.06, 10.02, 0.11] at variable values [2.77, 2.51, 2.11, 2.31, 2.46] 
Newest objective value(s): [1683.49, 8.93, 0.10] at variable values [2.23, 2.68, 1.55, 1.50, 2.92] 
```

Please note that in order to follow common Bayesian Optimisation convention, _veropt_ always maximises. If you need to minimise, simply put a negative sign on your objective.

## The Visualisation Tools

Once our optimisation has run for a few steps, we can visualise the surrogate model or other aspects of the optimisation to make sure everything is set up correctly.

For example, we can call,

```python
plot_prediction_grid_from_optimiser(
    optimiser=optimiser
)
```

and we'll get a figure like the one below.

<img width="10080" height="6480" alt="for_readme" src="https://github.com/user-attachments/assets/b43fafcc-d7f9-44ae-8bbe-7db3502b219e" />

For every objective function and variable combination, we see a cross section of the domain, where we can inspect the surrogate model (black line with grey area for uncertainty), acquisition function (grey lines), suggested points (red points with uncertainty bars) and evaluated points (colourful points).

These graphics are made with the library 'plotly', which offers modern, interactive plots. These can be saved and shared as html's, retaining the interactive features.

If you want to try out the library with a practice objective before setting up your own optimisation problem, we recommend looking through our examples.

## Interfaces

For optimization of computationally heavy, complex models, _veropt_ interfaces provide a framework to automatically submit, track and evaluate user-defined simulations. Below is an example of an experiment where a parameter of the ocean model [veros](https://veros.readthedocs.io/en/latest/) is optimised to simulate realistic current strength in an idealised setup.

```python
from veropt.interfaces.experiment import Experiment
from veropt.interfaces.local_simulation import LocalVerosRunner, LocalVerosConfig
from veropt.interfaces.result_processing import TestVerosResultProcessor

simulation_config = LocalVerosConfig.load("veropt/interfaces/configs/local_veros_config.json")
simulation_runner = LocalVerosRunner(config=simulation_config)

optimiser_config = "veropt/interfaces/configs/optimiser_config.json"
experiment_config = "veropt/interfaces/configs/veros_experiment_config.json"

result_processor = TestVerosResultProcessor(objective_names=["amoc"])

experiment = Experiment.from_the_beginning(
    simulation_runner=simulation_runner,
    result_processor=result_processor,
    experiment_config=experiment_config,
    optimiser_config=optimiser_config
)

experiment.run_experiment()
```

_veropt_ interfaces support the implementation of two types of experiments: local (for simulations running locally) and local slurm (for simulations running on a cluster).

For examples on how to use these features, see examples/interfaces.

## License

This project uses the [GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license.

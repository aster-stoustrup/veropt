
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
from veropt.optimiser.practice_objectives import Hartmann
from veropt import bayesian_optimiser

objective = Hartmann(
    n_variables=6
)

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective
)
```

## The Visualisation Tools

<img width="10080" height="6480" alt="for_readme" src="https://github.com/user-attachments/assets/86308763-5ca1-450d-ac31-17b4ea80d8f6" />

_veropt_ comes equipped with multiple visualisation tools that will help you inspect your optimisation problem and make sure everything looks correct.

In the figure above, we show a visualisation of the practice problem 'VehicleSafety' which features 3 objective functions and 5 variables.  

For every objective function and variable combination, we see a cross section of the domain, where we can inspect the surrogate model, acquisition function, suggested points and evaluated points.

These graphics are made with the library 'plotly', which offers modern, interactable plots that can be sent as html's.

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

experiment = Experiment(
    simulation_runner=simulation_runner,
    result_processor=result_processor,
    experiment_config=experiment_config,
    optimiser_config=optimiser_config
)

experiment.run_experiment()
```

_veropt_ interfaces support the implementation of two types of experiments: local (for simulations running locally) and local slurm (for simulations running on a cluster).

## License

This project uses the [GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license.

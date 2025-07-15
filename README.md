
# _veropt_, the versatile optimiser

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

_veropt_ comes equipped with multiple visualisation tools that will help you inspect your optimisation problem and make sure everything looks correct.

In the figure above, we show a visualisation of the practice problem 'VehicleSafety' which features 3 objective functions and 5 variables.  

For every objective function and variable combination, we see a cross section of the domain, where we can inspect the surrogate model, acquisition function, suggested points and evaluated points.

These graphics are made with the library 'plotly', which offers modern, interactable plots that can be sent as html's.

## License

This project uses the [GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license.

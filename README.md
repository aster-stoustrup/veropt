
# VerOpt - *the versatile optimiser*

VerOpt is a Python package that aims to make Bayesian Optimisation easy to approach, inspect and adjust. It was developed for the Versatile Ocean Simulator ([VEROS](https://veros.readthedocs.io/en/latest/)) with the aim of providing a user-friendly optimisation tool to tune ocean simulations to real world data. 

VerOpt can be used with any optimisation problem but has been developed for expensive optimisation problems with a small amount of evaluations (~100) and will probably be most relevant in such a context.

For more information about the package and the methods implemented in it, take a look at my [thesis report](https://nbi.ku.dk/english/theses/masters-theses/ida_lei_stoustrup/Ida_Stoustrup_MSc_Thesis.pdf). 

## Installation

To install veropt with the default dependencies and the package utilised by the GUI (PySide2), do the following:

```bash
pip install veropt[gui]
```

If you're installing veropt on a cluster and don't need the GUI you can simply do,

```bash
pip install veropt
```


## Usage

Below is a simple example of running an optimisation problem with veropt. 

```python
from veropt import BayesOptimiser
from veropt.obj_funcs.test_functions import *
from veropt.gui import veropt_gui

n_init_points = 24
n_bayes_points = 64
n_evals_per_step = 4

obj_func = PredefinedTestFunction("Hartmann")


optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)

veropt_gui.run(optimiser)
```

This example utilises one of the predefined test objective functions found in veropt.obj_funcs.test_functions. 

To use veropt with your own optimisation problem, you will need to create a class that uses the "ObjFunction" class from veropt/optimiser.py as a superclass. Your class must either include a method of running your objective function (YourClass.function()) or a method for both saving parameter values and loading new objective function values (YourClass.saver() and YourClass.loader()).

If you're using veropt with a veros simulation, take a look at veropt/obj_funcs/ocean_sims and the veros simulation examples under examples/ocean_examples.

## The GUI and the Visualisation Tools

<img width="1017" alt="Screenshot 2025-03-03 at 17 13 53" src="https://github.com/user-attachments/assets/f5fe7619-7e47-4746-a01e-2babbf3c7f89" />

After running the command,


```python
veropt_gui.run(optimiser)
```

You should see a window like the one above. From here, you can show the progress of the optimisation, visualise the predictions of the GP model, change essential parameters of the model or acquisition function and much more. 

##

If you press "Plot predictions" in the GUI, you will encounter a plot like the one below. 

<img width="936" alt="image" src="https://github.com/user-attachments/assets/2bceea4f-5d26-4707-8ed3-93d135c8642d" />

It shows a slice of the function domain, along the axis of a chosen optimisation parameter. You will be able to inspect the model, the acquisition function, as well as the suggested points for the next round of objective function evaluations. If any of this isn't as desired, you simply close the figure and go back to the GUI to modify the optimisation by changing the relevant parameters.

## License

This project uses the [GPLv3](https://choosealicense.com/licenses/gpl-3.0/) license.

from typing import Dict
from pydantic import BaseModel


class OptimizerObject(BaseModel):
    n_init_points: int
    n_Bayes_points: int
    n_evals_per_step: int
    

class MockOptimizer:
    def __init__(
            self,
            points = Dict
    ) -> None:
        self.points = points 

    def saver(
            self
    ) -> None:
        """Save optimizer object."""
        ...

    def loader(
            self
    ) -> None:
        """Load optimizer object."""
        ...

    def run_opt_step(
            self
    ) -> None:
        ...

    def edit_hyperparameters(
            self
    ) -> None:
        ...

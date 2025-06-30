from typing import Dict
from pydantic import BaseModel


class OptimiserObject(BaseModel):
    n_init_points: int
    n_Bayes_points: int
    n_evals_per_step: int
    

class MockOptimiser:
    def __init__(
            self,
            points: dict
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

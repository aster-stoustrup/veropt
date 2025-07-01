import pytest

from veropt.optimiser.model import GPyTorchSingleModel


def test_gpy_torch_single_model_init_mandatory_name() -> None:

    class TestModel(GPyTorchSingleModel):
        def __init__(self) -> None:
            super().__init__(
                n_variables=3
            )

    with pytest.raises(AssertionError):
        test_model = TestModel()

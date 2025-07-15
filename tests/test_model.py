import pytest

from veropt.optimiser.model import MaternSingleModel


def test_gpy_torch_single_model_init_mandatory_name() -> None:

    class TestModel(MaternSingleModel):
        pass

    with pytest.raises(AssertionError):
        _ = TestModel(n_variables=3)

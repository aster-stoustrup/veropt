import pytest
import torch

from veropt.optimiser.model import MaternSingleModel, DoubleMaternKernelSingleModel
from veropt.optimiser.utility import DataShape


def test_gpy_torch_single_model_init_mandatory_name() -> None:

    class TestModel(MaternSingleModel):
        pass

    with pytest.raises(AssertionError):
        _ = TestModel(n_variables=3)


def test_double_matern() -> None:

    var_1_tensor = torch.tensor([0.4, 0.1, 0.7, 0.3])
    var_2_tensor = torch.tensor([0.1, 0.5, 0.2, 0.6])
    var_3_tensor = torch.tensor([0.5, 0.2, 0.5, 0.9])

    obj_1_tensor = torch.tensor([2.1, 3.4, 5.2, 1.2])
    obj_2_tensor = torch.tensor([4.3, 6.3, 1.2, 4.2])

    var_tensor = torch.stack(
        tensors=(
            var_1_tensor,
            var_2_tensor,
            var_3_tensor,
        ),
        dim=DataShape.index_dimensions
    )

    obj_tensor = torch.stack(
        tensors=(
            obj_1_tensor,
            obj_2_tensor,
        ),
        dim=DataShape.index_dimensions
    )

    model = DoubleMaternKernelSingleModel(
        n_variables=3
    )

    model.initialise_model_with_data(
        train_inputs=var_tensor,
        train_targets=obj_tensor
    )

    # TODO: Finish test or maybe delete if not currently/urgently necessary
    pass

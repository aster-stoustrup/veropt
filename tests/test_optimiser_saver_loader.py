from veropt.optimiser.normalisation import Normaliser
from veropt.optimiser.saver_loader_utility import rehydrate_object


# TODO: Consider moving to new file now that the tested function was moved
def test_rehydrate_object_normaliser():

    rehydrate_object(
        superclass=Normaliser,
        name=''
    )

    assert False


def test_load_optimiser_from_json():

    # TODO: Load optimiser and test it's the same as the original
    #   - Need to do this at different stages
    #       - Before and after training the model

    assert False

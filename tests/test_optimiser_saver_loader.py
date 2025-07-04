from veropt.optimiser.normalisation import Normaliser
from veropt.optimiser.utility import rehydrate_object


def test_rehydrate_object_normaliser():

    rehydrate_object(
        superclass=Normaliser,
        name=''
    )

    assert False

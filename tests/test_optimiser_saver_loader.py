from veropt.optimiser.normalisation import Normaliser
from veropt.optimiser.optimiser_saver_loader import rehydrate_object

def test_rehydrate_object_normaliser():

    rehydrate_object(
        superclass=Normaliser,
        name=''
    )

    assert False

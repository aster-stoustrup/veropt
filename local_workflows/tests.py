import os

from utility import go_up_to_find_veropt_folder


def run_pytest():

    go_up_to_find_veropt_folder()

    # Note: Should be the same as in the github workflow. Please change both if you change this.
    os.system('pytest')


if __name__ == '__main__':

    # TODO: Test for coverage too?
    #   - Would require deciding on desired coverage rate
    #   - Should be added to github workflow ofc

    run_pytest()

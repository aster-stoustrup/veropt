import os

from utility import go_up_to_find_veropt_folder


def run_mypy():

    go_up_to_find_veropt_folder()

    # Note: Should be the same as in the github workflow. Please change both if you change this.
    os.system('mypy veropt tests')


if __name__ == '__main__':
    run_mypy()

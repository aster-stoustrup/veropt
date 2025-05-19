import os

from utility import go_up_to_find_veropt_folder


def run_flake8():

    go_up_to_find_veropt_folder()

    # Note: Should be the same as in the github workflow. Please change both if you change this.
    os.system('flake8 . --max-line-length=120')


if __name__ == '__main__':
    run_flake8()

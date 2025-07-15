import os

from utility import go_up_to_find_veropt_folder


def run_flake8():

    go_up_to_find_veropt_folder()

    os.system("flake8 .")


if __name__ == '__main__':
    run_flake8()

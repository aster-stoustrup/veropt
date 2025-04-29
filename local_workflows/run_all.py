from linting import run_flake8
from tests import run_pytest
from type_checking import run_mypy


def run_all():

    print("||| flake8 ||| \n \n")

    run_flake8()

    print("\n \n||| mypy ||| \n \n")

    run_mypy()

    print("\n \n||| pytest |||\n \n")

    run_pytest()


if __name__ == '__main__':
    run_all()

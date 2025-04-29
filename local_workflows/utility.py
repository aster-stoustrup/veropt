import os
from pathlib import Path


def go_up_to_find_veropt_folder():

    # Give a few tries to see if root directory is above us
    for i in range(3):

        current_folder_name = Path.cwd().name

        if current_folder_name == 'veropt':
            in_veropt_folder = True

        else:
            in_veropt_folder = False

        if in_veropt_folder is False:
            os.chdir('..')

        else:
            break

    if in_veropt_folder is False:
        raise RuntimeError("Couldn't find root directory")

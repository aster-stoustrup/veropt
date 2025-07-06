
def edit_veros_run_script(
        run_script: str,
        parameters: dict[str, float]
) -> None:
    
    with open(run_script, 'r') as file:
        data = file.readlines()

    # TODO: This is not robust. Need to figure out how to handle the indentation.
    #       Regular expression match /\w+settings\.([a-zA-Z0-9_]+)\w*=/, for all matches look up key 
    #       and set value; gather all keys in file, complement with assigned keys, add new lines.
    # TODO: How to introduce new parameters that are not already in the setup file?
    # TODO: Check if the parameters are already overwritten in the setup file.
    for i, line in enumerate(data):
        for key, value in parameters.items():
            if line.startswith(f"        settings.{key} ="):
                print(f"Overwriting {key} in setup file with value: {value}")
                old_line = data[i].strip()
                data[i] = f"        settings.{key} = {value}  # default {old_line}\n"
                break

    with open(run_script, 'w') as file:
        file.writelines(data)

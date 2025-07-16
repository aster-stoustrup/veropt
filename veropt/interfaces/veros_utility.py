import re


def edit_veros_run_script(
        run_script: str,
        parameters: dict[str, float]
) -> None:

    pattern = re.compile(r'^(?P<indent>\s*)settings\.(?P<key>[A-Za-z0-9_]+)\s*=\s*.*$')

    with open(run_script, 'r') as f:
        lines = f.readlines()

    new_lines: list[str] = []

    # TODO: Handle cases when key is not in setup file.
    for line in lines:
        m = pattern.match(line)

        if m:
            key = m.group('key')
            indent = m.group('indent')

            if key in parameters:
                val = parameters[key]
                old_assignment = line.strip()

                new_line = (
                    f"{indent}settings.{key} = {val}  "
                    f"# default: {old_assignment}\n"
                )

                print(f"Overwriting {key} with value: {val}")
                new_lines.append(new_line)
                continue

        new_lines.append(line)

    with open(run_script, 'w') as f:
        f.writelines(new_lines)

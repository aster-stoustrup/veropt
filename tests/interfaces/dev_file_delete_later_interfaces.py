import subprocess
import json 
import os
import torch

from veropt.interfaces.local_simulation import Conda


def activate_venv(venv_path, env_name):
    source = f'source {venv_path}/bin/activate {env_name}'
    dump = 'python -c "import os, json;print(json.dumps(dict(os.environ)))"'
    pipe = subprocess.Popen(['/bin/bash', '-c', '%s && %s' %(source,dump)], stdout=subprocess.PIPE)
    env = json.loads(pipe.stdout.read())
    os.environ = env


def run_veros(run_path, setup_script, setup_args, venv_path, device_id=1, float_type="float32", use_gpu=True):
    activate_venv(venv_path)
    os.chdir(run_path)
    HOME = os.environ['HOME']

    print(f"We are in {os.getcwd()}, veros={HOME}/.local/bin/veros")
    env_copy = os.environ.copy()
    env_copy["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # print(device_id)
    # print(env_copy)

    gpu_string = "--backend jax --device gpu" if use_gpu else ""
        
    command = f"{HOME}/.local/bin/veros run {gpu_string} --float-type {float_type} --setup-args {setup_args} {setup_script}".split(' ')
    print(command)
    pipe    = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env_copy)
    output, errors = pipe.communicate()
    print(f"output: {output}")
    print(f"errors: {errors}")

from veropt.optimiser.optimiser_utility import format_output_for_objective

vals = torch.FloatTensor([[0.0,1.1,2.2], [3.3,4.4,5.5], [5.5,6.6,7.7]])

print(vals)

format = format_output_for_objective(
    suggested_variables=vals,
    variable_names=["one", "two"]
)

print(format)
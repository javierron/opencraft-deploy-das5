import argparse
import glob
import os
import subprocess
import threading
from datetime import datetime
from enum import unique, Enum
from typing import List

import toml


@unique
class RunMode(Enum):
    OUTPUT = 1
    VOID = 2
    THREAD = 3
    FORGET = 4


@unique
class Output(Enum):
    NULL = 0
    FILE = 1
    STRING = 2


class Command(object):
    def __init__(self, command: str):
        self.command = command

    def build(self, address, wd=None, nohup=False, output: Output = Output.NULL, debug=False) -> List[str]:
        full_command = ["ssh", address]
        if wd is not None:
            full_command += ["cd", wd, ";"]
        if nohup:
            full_command += ["nohup"]
        if output == Output.STRING:
            output_str = ""
        else:
            if output == Output.FILE:
                out_file = f"{address}.log"
            elif output == Output.NULL:
                out_file = "/dev/null"
            else:
                raise RuntimeError(f"Not a valid command output mode: '{output}'")
            output_str = f">> {out_file} 2>&1"
        full_command += [f"{self.command} {output_str} < /dev/null"]
        if nohup:
            full_command += ["&", "echo", "$!"]
        if debug:
            print(" ".join(full_command))
        return full_command


def run_remotely(node: str, command: Command, wd=None, debug=False, mode: RunMode = RunMode.OUTPUT):
    if mode == mode.OUTPUT:
        output = Output.STRING
    elif debug:
        output = Output.FILE
    else:
        output = Output.NULL

    if mode == mode.FORGET:
        full_command = command.build(node, output=output, wd=wd, debug=debug, nohup=True)
        return subprocess.check_output(full_command).strip().decode()
    else:
        full_command = command.build(node, output=output, wd=wd, debug=debug, nohup=False)
        if mode == mode.OUTPUT:
            return subprocess.check_output(full_command).strip().decode()
        elif mode == mode.VOID:
            subprocess.call(full_command)
            return None
        elif mode == mode.THREAD:
            t = threading.Thread(target=subprocess.call, args=(full_command,))
            t.start()
            return t
        else:
            raise RuntimeError(f"Runmode '{mode}' does not exist.")


def kill(node: str, pid: str) -> None:
    t = run_remotely(node, Command(f"kill {pid}; wait {pid}"), mode=RunMode.THREAD)
    t.join(timeout=10.0)
    if t.is_alive():
        run_remotely(node, Command(f"kill -9 {pid}"))


def run_experiment(path: str, nodes: list, **kwargs) -> None:
    # TODO check the existence of all necessary files and directories before starting the experiment.
    assert len(path) > 0
    assert os.path.isdir(path)
    assert len(nodes) > 0

    config_path = os.path.join(path, "experiment-config.toml")
    assert os.path.isfile(config_path)
    config = toml.load(config_path)

    # TODO make a function for this, with the required checks.
    opencraft_matches = glob.glob(os.path.join(path, "../../resources/opencraft*.jar"), recursive=False)
    assert len(opencraft_matches) == 1
    opencraft = opencraft_matches[0]
    # TODO support opencraft world folder, through extra resources dir

    yardstick_matches = glob.glob(os.path.join(path, "../../resources/yardstick*.jar"), recursive=False)
    assert len(yardstick_matches) == 1
    # TODO support more than one node.
    yardstick = yardstick_matches[0]

    experiment_iterations = config["iterations"]
    assert experiment_iterations is not None
    assert type(experiment_iterations) is int
    try:
        opencraft_jvm_args = config["opencraft"]["jvm"]
    except KeyError:
        opencraft_jvm_args = []
    try:
        yardstick_jvm_args = config["yardstick"]["jvm"]
    except KeyError:
        yardstick_jvm_args = []
    for i in range(experiment_iterations):
        run_iteration(i, nodes, path, opencraft, opencraft_jvm_args, yardstick, yardstick_jvm_args)


def run_iteration(iteration: int, nodes: list, path: str, opencraft: str, opencraft_jvm_args: str, yardstick: str,
                  yardstick_jvm_args: str) -> None:
    iteration_dir = os.path.join(path, str(iteration))
    if os.path.isdir(iteration_dir):
        return
    else:
        os.mkdir(iteration_dir)

    node = nodes[0]
    opencraft_wd = run_remotely(node, Command(f"mktemp -d"), mode=RunMode.OUTPUT)
    run_remotely(node, Command(f"cp -r {os.path.join(path, '../../resources/config')} {opencraft_wd}"), debug=True)
    opencraft_pid = run_remotely(node, Command(f"java {' '.join(opencraft_jvm_args)} -jar {opencraft}"),
                                 wd=opencraft_wd, debug=True,
                                 mode=RunMode.FORGET)

    run_remotely(node, Command(f"cp {os.path.join(path, '../../resources/yardstick.toml')} {iteration_dir}"),
                 debug=True)
    print(datetime.now())
    yardstick_thread = run_remotely(node, Command(f"java {' '.join(yardstick_jvm_args)} -jar {yardstick}"),
                                    wd=iteration_dir,
                                    debug=True, mode=RunMode.THREAD)
    # TODO add timeout.
    yardstick_thread.join()
    print(datetime.now())
    run_remotely(node, Command(f"rm {os.path.join(iteration_dir, 'yardstick.toml')}"))
    kill(node, opencraft_pid)
    run_remotely(node, Command(f"mv {os.path.join(opencraft_wd, 'dyconits.log')} {iteration_dir}"))
    run_remotely(node, Command(
        f"mv {os.path.join(opencraft_wd, node + '.log')} {os.path.join(iteration_dir, node + '.opencraft.log')}"))
    run_remotely(node, Command(f"rm -rf {opencraft_wd}"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()
    run = sp.add_parser("run")
    run.add_argument("path", help="path to experiment")
    run.add_argument("nodes", nargs="+", help="hostnames of nodes to use for experiment")
    run.set_defaults(func=run_experiment)
    args = parser.parse_args()
    args.func(**vars(args))

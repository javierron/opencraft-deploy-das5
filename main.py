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


# class Node(object):
#     def __init__(self, address: str):
#         self.address = address


class Command(object):
    def __init__(self, command: str):
        self.command = command

    def build(self, address, wd=None, nohup=False, debug=False) -> List[str]:
        full_command = ["ssh", address]
        if wd is not None:
            full_command += ["cd", wd, ";"]
        if nohup:
            full_command += ["nohup"]
        if debug:
            out_file = f"{address}.log"
        else:
            out_file = "/dev/null"
        full_command += [f"{self.command} >> {out_file} 2>&1 < /dev/null"]
        if nohup:
            full_command += ["&", "echo", "$!"]
        if debug:
            print(" ".join(full_command))
        return full_command


def run_remotely(node: str, command: Command, wd=None, debug=False, mode: RunMode = RunMode.OUTPUT):
    if mode == mode.FORGET:
        full_command = command.build(node, wd=wd, debug=debug, nohup=True)
        return subprocess.check_output(full_command).strip().decode()
    else:
        full_command = command.build(node, wd=wd, debug=debug, nohup=False)
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


# class RunningProgram(object):
#     def __init__(self, program, pid, node, wd, output_dir, remove_wd):
#         self.program = program
#         self.pid = pid
#         self.node = node
#         self.wd = wd
#         self.output_dir = output_dir
#         self.remove_wd = remove_wd
#
#     def stop(self):
#         run_remotely(self.node, Command(f"kill -9 {self.pid}"))
#
#     def clean(self):
#         for file in self.program.output_files:
#             run_remotely(self.node, Command(f"cp {file} {self.output_dir}"))
#         if self.remove_wd:
#             run_remotely(self.node, Command(f"rm -rf {self.wd}"))


# class Program(object):
#     def __init__(self, command: Command, input_files=None, output_files=None):
#         # TODO add debug file to output files
#         if output_files is None:
#             output_files = []
#         if input_files is None:
#             input_files = []
#         self.command = command
#         self.input_files = input_files
#         self.output_files = output_files
#
#     def run(self, node: Node, output_dir: str, mode: RunMode):
#         tempdir = True
#         working_directory = self.command.wd
#         if working_directory is None:
#             working_directory = run_remotely(node, Command("mktemp -d"))
#         else:
#             tempdir = False
#         for file in self.input_files:
#             run_remotely(node, Command(f"cp {file} {working_directory}"))
#         pid = run_remotely(node, self.command, mode)
#         running = RunningProgram(self, pid, node, working_directory, output_dir, tempdir)
#         if mode == RunMode.FORGET:
#             return running
#         else:
#             running.clean()


# def run_remotely_async(address: str, command: str, cwd: str = None, debug: bool = False) -> str:
#     """
#     Runs the given command remotely. Disconnects the STDIN from the local machine and forwards STDIN and STDOUT to
#     /dev/null. Return the PID of the remotely started process, such that we can terminate it at some later time.
#     """
#     return subprocess.check_output(Command(command, cwd, debug, nohup=True).build(address)).strip().decode()


# TODO copy dyconits.log to iteration dir.
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
    for i in range(experiment_iterations):
        run_iteration(i, nodes, path, opencraft, yardstick)


def run_iteration(iteration: int, nodes: list, path: str, opencraft: str, yardstick: str) -> None:
    iteration_dir = os.path.join(path, str(iteration))
    if os.path.isdir(iteration_dir):
        return
    else:
        os.mkdir(iteration_dir)

    node = nodes[0]
    run_remotely(node, Command(f"cp -r {os.path.join(path, '../../resources/config')} {iteration_dir}"), debug=True)
    # TODO set the right amount of heap space.
    opencraft_pid = run_remotely(node, Command(f"java -jar {opencraft}"), wd=iteration_dir, debug=True,
                                 mode=RunMode.FORGET)

    run_remotely(node, Command(f"cp {os.path.join(path, '../../resources/yardstick.toml')} {iteration_dir}"),
                 debug=True)
    print(datetime.now())
    # TODO use Yardstick config.
    yardstick_thread = run_remotely(node,
                                    Command(f"java -jar {yardstick} -e 4 -E bots=1 -E duration=60 -E joininterval=5"),
                                    wd=iteration_dir, debug=True, mode=RunMode.THREAD)
    # TODO add timeout.
    yardstick_thread.join()
    print(datetime.now())
    run_remotely(node, Command(f"rm {os.path.join(iteration_dir, 'yardstick.toml')}"))
    # TODO make kill friendly
    run_remotely(node, Command(f"kill -9 {opencraft_pid}"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()
    run = sp.add_parser("run")
    run.add_argument("path", help="path to experiment")
    run.add_argument("nodes", nargs="+", help="hostnames of nodes to use for experiment")
    run.set_defaults(func=run_experiment)
    args = parser.parse_args()
    args.func(**vars(args))

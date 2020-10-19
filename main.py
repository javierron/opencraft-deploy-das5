import argparse
import glob
import os
import subprocess
import threading
from abc import ABC, abstractmethod
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


class Instance(ABC):

    def __init__(self, node: str, experiment_path: str, iteration: int):
        self.node = node
        self.iteration = iteration
        self.experiment_path = experiment_path
        self.iteration_dir = os.path.join(self.experiment_path, str(self.iteration))

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def clean(self):
        pass


class Opencraft(Instance):

    def __init__(self, node: str, experiment_path: str, iteration: int, jvm_args: List[str]):
        super().__init__(node, experiment_path, iteration)
        assert os.path.isdir(experiment_path)
        self.jvm_args = jvm_args
        self.executable = None
        self.pid = None
        self.opencraft_wd = None

    def setup(self):
        # TODO make a function for this, with the required checks.
        opencraft_matches = glob.glob(os.path.join(self.experiment_path, "../../resources/opencraft*.jar"),
                                      recursive=False)
        assert len(opencraft_matches) == 1
        self.executable = opencraft_matches[0]
        self.opencraft_wd = run_remotely(self.node, Command(f"mktemp -d"), mode=RunMode.OUTPUT)
        run_remotely(self.node,
                     Command(
                         f"cp -r {os.path.join(self.experiment_path, '../../resources/config')} {self.opencraft_wd}"),
                     debug=True)

    def start(self):
        self.pid = run_remotely(self.node, Command(f"java {' '.join(self.jvm_args)} -jar {self.executable}"),
                                wd=self.opencraft_wd, debug=True,
                                mode=RunMode.FORGET)

    def stop(self):
        kill(self.node, self.pid)

    def clean(self):
        run_remotely(self.node, Command(f"mv {os.path.join(self.opencraft_wd, 'dyconits.log')} {self.iteration_dir}"))
        run_remotely(self.node, Command(
            f"mv {os.path.join(self.opencraft_wd, self.node + '.log')} {os.path.join(self.iteration_dir, 'opencraft.' + self.node + '.log')}"))
        run_remotely(self.node, Command(f"rm -rf {self.opencraft_wd}"))


def to_ib(address: str) -> str:
    if address.startswith("10.149."):
        return address
    elif address.startswith("node0"):
        # TODO don't hard-code VU site.
        return f"10.149.0.{address[-2:]}"
    else:
        raise RuntimeError(f"Cannot translate '{address}' to infiniband address.")


class Yardstick(Instance):

    def __init__(self, node: str, experiment_path: str, iteration: int, jvm_args: List[str], opencraft_node: str):
        super().__init__(node, experiment_path, iteration)
        self.jvm_args = jvm_args
        self.opencraft_node = to_ib(opencraft_node)
        self.yardstick_wd = None
        self.thread = None
        self.executable = None

    def setup(self):
        yardstick_matches = glob.glob(os.path.join(self.experiment_path, "../../resources/yardstick*.jar"),
                                      recursive=False)
        assert len(yardstick_matches) == 1
        # TODO support more than one node.
        self.executable = yardstick_matches[0]
        self.yardstick_wd = run_remotely(self.node, Command(f"mktemp -d"), mode=RunMode.OUTPUT)
        run_remotely(self.node, Command(
            f"cp {os.path.join(self.experiment_path, '../../resources/yardstick.toml')} {self.yardstick_wd}"),
                     debug=True)

    def start(self):
        self.thread = run_remotely(self.node,
                                   Command(
                                       f"java {' '.join(self.jvm_args)} -jar {self.executable} --host {self.opencraft_node}"),
                                   wd=self.yardstick_wd,
                                   debug=True, mode=RunMode.THREAD)

    def stop(self):
        self.thread.join()

    def clean(self):
        run_remotely(self.node, Command(
            f"mv {os.path.join(self.yardstick_wd, self.node + '.log')} {os.path.join(self.iteration_dir, 'yardstick.' + self.node + '.log')}"))
        workload_dir = os.path.join(self.yardstick_wd, 'workload')
        run_remotely(self.node, Command(
            f"[ ! -f {workload_dir} ] || mv {workload_dir} {os.path.join(self.iteration_dir, 'yardstick.workload.' + self.node)}"))
        run_remotely(self.node, Command(f"rm -rf {self.yardstick_wd}"))


class Pecosa(Instance):

    def __init__(self, node: str, experiment_path: str, iteration: int, opencraft_pid):
        super().__init__(node, experiment_path, iteration)
        self.opencraft_pid = opencraft_pid
        self.executable = None
        self.log_file = None
        self.pid = None

    def setup(self):
        pecosa_matches = glob.glob(os.path.join(self.experiment_path, "../../resources/pecosa.py"), recursive=False)
        assert len(pecosa_matches) == 1
        self.executable = pecosa_matches[0]
        self.log_file = run_remotely(self.node, Command(f"mktemp -p /local"), mode=RunMode.OUTPUT)

    def start(self):
        self.pid = run_remotely(self.node, Command(f"python {self.executable} {self.log_file} {self.opencraft_pid}"),
                                mode=RunMode.FORGET)

    def stop(self):
        kill(self.node, self.pid)

    def clean(self):
        run_remotely(self.node,
                     Command(f"mv {self.log_file} {os.path.join(self.iteration_dir, 'performance-counters.log')}"))


def run_experiment(path: str, nodes: list, **kwargs) -> None:
    # TODO check the existence of all necessary files and directories before starting the experiment.
    assert len(path) > 0
    assert os.path.isdir(path)
    assert len(nodes) > 0

    config_path = os.path.join(path, "experiment-config.toml")
    assert os.path.isfile(config_path)
    config = toml.load(config_path)
    assert max(config["deployment"]["yardstick"]) < len(nodes)

    pecosa_matches = glob.glob(os.path.join(path, "../../resources/pecosa.py"), recursive=False)
    assert len(pecosa_matches) == 1

    # TODO make a function for this, with the required checks.
    opencraft_matches = glob.glob(os.path.join(path, "../../resources/opencraft*.jar"), recursive=False)
    assert len(opencraft_matches) == 1
    # TODO support opencraft world folder, through extra resources dir

    yardstick_matches = glob.glob(os.path.join(path, "../../resources/yardstick*.jar"), recursive=False)
    assert len(yardstick_matches) == 1
    # TODO support more than one node.

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
        run_iteration(i, nodes, path, opencraft_jvm_args, yardstick_jvm_args, config)


def run_iteration(iteration: int, nodes: list, path: str,
                  opencraft_jvm_args: List[str],
                  yardstick_jvm_args: List[str],
                  config) -> None:
    iteration_dir = os.path.join(path, str(iteration))
    if os.path.isdir(iteration_dir):
        return
    else:
        os.mkdir(iteration_dir)

    node = nodes[0]

    opencraft = Opencraft(node, path, iteration, opencraft_jvm_args)
    opencraft.setup()
    opencraft.start()
    pecosa = Pecosa(node, path, iteration, opencraft.pid)
    pecosa.setup()
    pecosa.start()

    yardstick_instances = []
    for index in config["deployment"]["yardstick"]:
        node = nodes[index]
        yardstick = Yardstick(node, path, iteration, yardstick_jvm_args, opencraft.node)
        yardstick.setup()
        yardstick_instances.append(yardstick)

    for yi in yardstick_instances:
        yi.start()
    for yi in yardstick_instances:
        # TODO change call to signal that Yardstick.stop simply waits.
        yi.stop()

    pecosa.stop()
    opencraft.stop()

    pecosa.clean()
    opencraft.clean()
    for yi in yardstick_instances:
        yi.clean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()
    run = sp.add_parser("run")
    run.add_argument("path", help="path to experiment")
    run.add_argument("nodes", nargs="+", help="hostnames of nodes to use for experiment")
    run.set_defaults(func=run_experiment)
    args = parser.parse_args()
    args.func(**vars(args))

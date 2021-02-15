#!/usr/bin/env python

import argparse
import fnmatch
import getpass
import glob
import os
import pathlib
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import unique, Enum
from typing import List

import toml

_RESOURCES_DIR_NAME = "resources"
_RESULTS_DIR_NAME = "results"
_PLOTS_DIR_NAME = "figures"
_SPECIAL_DIRS = [_PLOTS_DIR_NAME, _RESULTS_DIR_NAME, _RESOURCES_DIR_NAME]


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


# TODO use pathlib module
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


def get_reservation(reservation: int) -> List[str]:
    llist = subprocess.check_output(["preserve", "-llist"]).strip().decode()
    for line in llist.split('\n'):
        split = line.split('\t')
        if len(split) > 1:
            split = [part.strip() for part in split]
            if split[0] == str(reservation) and split[1] == getpass.getuser():
                return split
    return []


def get_nodes(reservation: int) -> List[str]:
    info = get_reservation(reservation)
    if len(info) > 0:
        return info[-1].split()
    return []


def wait_for_reservation_ready(reservation: int) -> None:
    ready = False
    while not ready:
        info = get_reservation(reservation)
        if len(info) == 0:
            raise ValueError(f"Reservation {reservation} does not exist for current user!")
        reservation_status = info[6]
        if reservation_status == "R":
            ready = True
        else:
            print(f"Waiting for reservation to become ready. Currently ({datetime.now()}): {reservation_status}")
            time.sleep(5)


def kill(node: str, pid: str) -> None:
    t = run_remotely(node, Command(f"kill {pid}; wait {pid}"), mode=RunMode.THREAD)
    t.join(timeout=10.0)
    if t.is_alive():
        run_remotely(node, Command(f"kill -9 {pid}"))


def find_resource(path: str, pattern: str, file: bool = True, root_path: str = "/"):
    assert os.path.isdir(root_path)
    assert os.path.isdir(path)
    assert os.path.realpath(path).startswith(os.path.realpath(root_path))

    if file:
        func = os.path.isfile
    else:
        func = os.path.isdir
    resource_path = os.path.join(path, "resources")
    if os.path.isdir(resource_path):
        entries = [f for f in os.listdir(resource_path) if func(os.path.join(resource_path, f))]
        for entry in entries:
            if fnmatch.fnmatch(entry, pattern):
                resource = os.path.realpath(os.path.join(resource_path, entry))
                return resource
    if os.path.samefile(path, root_path):
        return None
    return find_resource(os.path.dirname(path), pattern, file, root_path)


class Instance(ABC):

    def __init__(self, node: str, path: str, root_path: str, iteration: int):
        assert os.path.isdir(path)
        assert os.path.isdir(root_path)
        assert os.path.realpath(path).startswith(os.path.realpath(root_path))
        self.node = node
        self.iteration = iteration
        self.path = path
        self.root_path = root_path

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

    def __init__(self, node: str, iteration_path: str, root_path: str, iteration: int, jvm_args: List[str]):
        super().__init__(node, iteration_path, root_path, iteration)
        self.jvm_args = jvm_args
        self.executable = None
        self.pid = None
        self.opencraft_wd = None

    def setup(self):
        # TODO make a function for this, with the required checks.
        opencraft = find_resource(self.path, "opencraft*.jar",
                                  root_path=self.root_path)
        assert opencraft is not None
        self.executable = opencraft
        self.opencraft_wd = run_remotely(self.node, Command(f"mktemp -d"), mode=RunMode.OUTPUT)
        opencraft_config = find_resource(self.path, "config", file=False,
                                         root_path=self.root_path)
        assert os.path.isdir(opencraft_config)
        run_remotely(self.node, Command(f"cp -r {opencraft_config} {self.opencraft_wd}"), debug=True)

    def start(self):
        self.pid = run_remotely(self.node, Command(f"java {' '.join(self.jvm_args)} -jar {self.executable}"),
                                wd=self.opencraft_wd, debug=True,
                                mode=RunMode.FORGET)

    def stop(self):
        kill(self.node, self.pid)

    def clean(self):
        dyconits_log = os.path.join(self.opencraft_wd, 'dyconits.log')
        dyconits_log_dst = os.path.join(self.path, f"dyconits.{self.node}.log")
        run_remotely(self.node, Command(f"[ ! -f {dyconits_log} ] || mv {dyconits_log} {dyconits_log_dst}"))
        player_log = os.path.join(self.opencraft_wd, 'opencraft-events.log')
        player_log_dst = os.path.join(self.path, f"opencraft-events.{self.node}.log")
        run_remotely(self.node, Command(f"[ ! -f {player_log} ] || mv {player_log} {player_log_dst}"))
        run_remotely(self.node, Command(
            f"mv {os.path.join(self.opencraft_wd, self.node + '.log')} {os.path.join(self.path, 'opencraft.' + self.node + '.log')}"))
        run_remotely(self.node, Command(f"rm -rf {self.opencraft_wd}"))


def to_ib(address: str) -> str:
    if address.startswith("10.149."):
        return address
    elif address.startswith("node0"):
        # TODO don't hard-code VU site.
        return f"10.149.0.{int(address[-2:])}"
    else:
        raise RuntimeError(f"Cannot translate '{address}' to infiniband address.")


class Yardstick(Instance):

    def __init__(self, node: str, experiment_path: str, root_path: str, iteration: int, jvm_args: List[str],
                 opencraft_node: str):
        super().__init__(node, experiment_path, root_path, iteration)
        self.jvm_args = jvm_args
        self.opencraft_node = to_ib(opencraft_node)
        self.yardstick_wd = None
        self.thread = None
        self.executable = None

    def setup(self):
        yardstick = find_resource(self.path, "yardstick*.jar", root_path=self.root_path)
        assert yardstick is not None
        self.executable = yardstick
        self.yardstick_wd = run_remotely(self.node, Command(f"mktemp -d"), mode=RunMode.OUTPUT)
        yardstick_config = find_resource(self.path, "yardstick.toml", root_path=self.root_path)
        assert yardstick_config is not None
        run_remotely(self.node, Command(f"cp {yardstick_config} {self.yardstick_wd}"), debug=True)

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
            f"mv {os.path.join(self.yardstick_wd, self.node + '.log')} {os.path.join(self.path, 'yardstick.' + self.node + '.log')}"))
        workload_dir = os.path.join(self.yardstick_wd, 'workload')
        run_remotely(self.node, Command(
            f"[ ! -f {workload_dir} ] || mv {workload_dir} {os.path.join(self.path, 'yardstick.workload.' + self.node)}"))
        run_remotely(self.node, Command(f"rm -rf {self.yardstick_wd}"))


class Pecosa(Instance):

    def __init__(self, node: str, experiment_path: str, root_path: str, iteration: int, opencraft_pid):
        super().__init__(node, experiment_path, root_path, iteration)
        self.opencraft_pid = opencraft_pid
        self.executable = None
        self.log_file = None
        self.pid = None

    def setup(self):
        pecosa = find_resource(self.path, "pecosa.py", root_path=self.root_path)
        assert os.path.isfile(pecosa)
        self.executable = pecosa
        # TODO make all log files go to /local?
        self.log_file = run_remotely(self.node, Command(f"mktemp -p /local"), mode=RunMode.OUTPUT)

    def start(self):
        self.pid = run_remotely(self.node, Command(f"python {self.executable} {self.log_file} {self.opencraft_pid}"),
                                mode=RunMode.FORGET)

    def stop(self):
        kill(self.node, self.pid)

    def clean(self):
        filename = f"pecosa.{self.node}.log"
        run_remotely(self.node,
                     Command(f"mv {self.log_file} {os.path.join(self.path, filename)}"))


def run_experiment(path: str, reservation: int, **kwargs) -> None:
    # TODO check the existence of all necessary files and directories before starting the experiment.
    assert len(path) > 0
    assert os.path.isdir(path)
    exp_group_path = str(pathlib.Path(path).parent)
    assert os.path.isdir(exp_group_path)

    for entry in os.listdir(path):
        if entry not in _SPECIAL_DIRS and os.path.isdir(os.path.join(path, entry)):
            run_configuration(reservation, os.path.join(path, entry), exp_group_path)


def run_configuration(reservation: int, configuration_path: str, root_path: str):
    config_file = find_resource(configuration_path, "experiment-config.toml", root_path=root_path)
    assert config_file is not None
    config = toml.load(config_file)
    num_iterations = config["iterations"]
    assert num_iterations is not None
    assert type(num_iterations) is int
    for i in range(num_iterations):
        iteration_path = os.path.join(configuration_path, str(i))
        run_iteration(reservation, iteration_path, root_path, i)


def run_iteration(reservation: int, path: str, root_path: str, iteration: int) -> None:
    wait_for_reservation_ready(reservation)
    nodes = get_nodes(reservation)
    if len(nodes) <= 0:
        print(f"Reservation {reservation} not found. Skipping...")
        return
    if os.path.isdir(path):
        print(f"Results for iteration {iteration} already exist at {path}. Skipping...")
        return
    else:
        os.mkdir(path)

    config_path = find_resource(path, "experiment-config.toml", root_path=root_path)
    assert os.path.isfile(config_path)
    config = toml.load(config_path)
    assert max(config["deployment"]["yardstick"]) < len(nodes)
    # TODO support opencraft world folder, through extra resources dir
    try:
        opencraft_jvm_args = config["opencraft"]["jvm"]
    except KeyError:
        opencraft_jvm_args = []
    try:
        yardstick_jvm_args = config["yardstick"]["jvm"]
    except KeyError:
        yardstick_jvm_args = []

    opencraft_node = nodes[0]
    yardstick_nodes = [node for node in nodes if nodes.index(node) in config["deployment"]["yardstick"]]

    _run_iteration(iteration, opencraft_node, yardstick_nodes, opencraft_jvm_args, yardstick_jvm_args, path, root_path)


def _run_iteration(iteration, opencraft_node, yardstick_nodes, opencraft_jvm_args, yardstick_jvm_args, path, root_path):
    opencraft = Opencraft(opencraft_node, path, root_path, iteration, opencraft_jvm_args)
    opencraft.setup()
    opencraft.start()
    # Add delay for server to start before yardstick/pecosa connection
    time.sleep(30)
    pecosa = Pecosa(opencraft_node, path, root_path, iteration, opencraft.pid)
    pecosa.setup()
    pecosa.start()
    yardstick_instances = []
    for node in yardstick_nodes:
        yardstick = Yardstick(node, path, root_path, iteration, yardstick_jvm_args, opencraft.node)
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


def collect_results(path: str, **kwargs):
    collect_results_for_prefix(path, file_prefix="pecosa")
    collect_results_for_prefix(path, file_prefix="opencraft-events")
    collect_results_for_prefix(path, file_prefix="dyconits")


def collect_results_for_prefix(path: str, file_prefix: str):
    assert os.path.isdir(path)
    results_dir = os.path.join(path, _RESULTS_DIR_NAME)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    result_files = []
    for configuration_dirname in os.listdir(path):
        configuration_dir = os.path.join(path, configuration_dirname)
        if os.path.basename(configuration_dir) not in _SPECIAL_DIRS and os.path.isdir(configuration_dir):
            pecosa_glob_string = os.path.join(configuration_dir, f"**/{file_prefix}.node*.log")
            result_files += glob.glob(pecosa_glob_string, recursive=True)
    results_file = os.path.join(results_dir, f"{file_prefix}.log")
    print(f"merging {result_files}")
    headers = get_all_headers(result_files)
    headers += ["node", "iteration", "config"]
    with open(results_file, "w+") as fout:
        fout.write("\t".join(headers) + os.linesep)
        for partial_results_file in result_files:
            node = partial_results_file.split(".")[-2]
            iteration_dir = os.path.dirname(partial_results_file)
            iteration = os.path.basename(iteration_dir)
            config = os.path.basename(os.path.dirname(iteration_dir))
            with open(partial_results_file, "r") as fin:
                header = fin.readline().strip().split("\t") + ["node", "iteration", "config"]
                header_mapping = [headers.index(column) for column in header]
                for line in fin.readlines():
                    output_line = [""] * len(headers)
                    values = line.strip().split("\t") + [node, iteration, config]
                    for i in range(len(values)):
                        output_line[header_mapping[i]] = values[i]
                    fout.write("\t".join(output_line) + os.linesep)


def get_all_headers(result_files):
    headers = []
    for partial_results_file in result_files:
        assert os.path.isfile(partial_results_file)
        with open(partial_results_file, "r") as fin:
            header = fin.readline().strip().split("\t")
            for column in header:
                if column not in headers:
                    headers.append(column)
    return headers


def is_number(something) -> bool:
    try:
        float(something)
        return True
    except ValueError:
        return False


def inspect_result(path: str, **kwargs):
    assert os.path.isfile(path)
    file_columns = 0
    file_types = []
    line_number = 0
    faulty_lines = []
    with open(path) as fin:
        for line in fin:
            line_number += 1
            line_parts = line.strip().split("\t")
            num_columns = len(line_parts)
            if line_number == 1:
                file_columns = num_columns
            else:
                types = ["number" if is_number(x) else "string" for x in line_parts]
                if num_columns != file_columns:
                    print(f"Line {line_number} has {num_columns} columns. Expected {file_columns}")
                    faulty_lines.append(line_number)
                    continue
                if line_number == 2:
                    file_types = types
                elif types != file_types:
                    print(f"Line {line_number} has types {types}. Expected {file_types}")
                    faulty_lines.append(line_number)
                    continue
    if len(faulty_lines) > 0:
        print("If you are certain the lines above can/should be deleted, you can run:")
        # sed -i.bak -e '5,10d;12d' file
        formatted_line_numbers = ";".join([f"{x}d" for x in faulty_lines])
        print(f"sed -i.bak -e '{formatted_line_numbers}' {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers()

    run = sp.add_parser("run")
    run.add_argument("path", help="path to experiment")
    run.add_argument("reservation", help="hostnames of nodes to use for experiment")
    run.set_defaults(func=run_experiment)

    collect = sp.add_parser("collect")
    collect.add_argument("path", help="path to experiment")
    collect.set_defaults(func=collect_results)

    inspect = sp.add_parser("inspect")
    inspect.add_argument("path", help="path to results file")
    inspect.set_defaults(func=inspect_result)

    args = parser.parse_args()
    args.func(**vars(args))

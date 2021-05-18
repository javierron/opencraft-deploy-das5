
# Opencraft Deploy DAS-5

This program deploys Opencraft and Yardstick on the DAS-5 to conduct experiments.
This readme will help you set things up.

_Troughout this README, we use the example of conducting an experiment on the effects of using different
Dyconit policies in Opencraft.
You don't need to know what these are, except that they are a part of Opencraft which can be configured._

## Directory Structure

The scripts in this repository assume a certain directory structure.
This allows us to simplify the scripts.

```
experiments
- <experiment-collection-name>  # create a copy of this when updating static assets (e.g., in case you find and fix a bug in the system under test)
  - resources
    - opencraft.jar
    - yardstick.jar
  - <experiment-name>           # create a copy of this if you want to conduct a new experiment
    - resources
      - yardstick.toml
      - experiment-config.toml  # experiment configuration, including number of experiment iterations
    - <configuration-name>      # create a copy of this if you want to add a configuration to your experiment
      - resources
        - config
          - opencraft.yml
      - 0                       # results directory; created automatically based on the number of experiment iterations
        - opencraft.node001.log
        - dyconit.node001.log
        - players.node001.log
        - yardstick.node002.log
        - yardstick.node003.log
```

Advantages of this structure include:

- Experiment results are easy to parse. For each experiment, loop over the configuration
directories (`<configuration-name>`) and iteration directories (`0-99`), append the configuration name and iteration
to the results file, and concatenate all results files.
- Experiments are easy to repeat if something goes wrong. Simply copy the experiment (collection) directory,
fix whatever is broken, and rerun this experiment deployment tool.

Disadvantages of this structure include:

- The high level of nesting can make navigation cumbersome. Automate everything!


## Preparing an Experiment

Create the following directories somewhere under `/var/scratch/<username>/`:

1. `opencraft-experiments`
2. `opencraft-experiments/my-first-experiments`

Populate the `resources` directory with the resources that will remain static across all experiments.
Because we are evaluating different dyconit policies, we can (re)use a single Opencraft JAR, and only change
its configuration. Therefore, we can place the Opencraft JAR in the `resources` directory.

In the `results` directory, we create a directory that describes our experiment.
For example, `100players-policy1`, indicating that the workload consists of 100 players, and that Opencraft is using
dyconit policy 1.
In this directory we place our Yardstick configuration file, configured to connect 100 players,
and our Opencraft configuration file, configured to use policy 1.

Finally, create a `config.toml` file that indicates how many nodes are required to run Yardstick,
and how many iterations of the experiment should be performed. For example:

```
iterations = 50

[nodes]
yardstick = 1
```

## Running Experiments

After preparing an experiment, running it is simple:

```
python main.py run /path/to/<timestamp>-<name>/results/<experiment-name> hostname1 [hostname2...]
```

The hostname(s) are the machines reserved for the experiment.
You can reserve machines on the DAS-5 using the command `preserve`.

## Plotting Experiment Results

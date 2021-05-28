
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

Create the following directories somewhere under `/var/scratch/<username>/`, each one with a corresponding nested `resources` directory:

1. `opencraft-experiments`
2. `opencraft-experiments/my-first-experiment`
3. `opencraft-experiments/my-first-experiment/100players-policy1`

Populate the `resources` directories with the resources needed for the experiment. 

Since there is a `resources` directory in every level of the hierarchy, we can use specific resources for each experiment, 
as well as reuse static resources between experiments, e.g.: Because we are evaluating different dyconit policies, we can 
(re)use a single Opencraft JAR, by placing it at the top level of the hierarchy, and only change its configuration. 
Therefore, we can place the Opencraft JAR in the `opencraft-experiments/resources` directory.

Also in `opencraft-experiments/my-first-experiment/resources`, we create a `experiment-config.toml` file that indicates in which nodes
Yardstick will be run on, and how many iterations of the experiment should be performed. For example:


```
iterations = 50

[deployment]
yardstick = [1, 2] # array of nodes to run yardstick on. Opencraft will run on node 0.
```


The `100players-policy1` directory describes a specific configuration for the experiment, 
in this case the name indicates that the workload consists of 100 players, and that Opencraft is using dyconit policy 1.
Accordingly, in its own `resources` directory we place our Yardstick configuration file, configured to connect 100 players, 
and our Opencraft configuration file, configured to use policy 1. 

We can have several configurations per experiment, and each one will be run the number of iterations described
in the `experiment-config.toml` file.


## Running Experiments

After preparing an experiment, running it is simple:

```
python ocd.py run /path/to/<experiment-collection-name>/<experiment-name> <reservation-id>
```

The `reservation-id` identifies the machines reserved for the experiment.
You can reserve machines on the DAS-5 using the command `preserve -np <number-of-nodes> -t <time-in-seconds>`.

## Plotting Experiment Results

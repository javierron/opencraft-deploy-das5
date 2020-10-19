
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
- <timestamp>-<name>
|- resources
||- opencraft.jar
||- yardstick.jar
|- results
||- <experiment-name>
|||- config.toml
|||- 0
||||- opencraft.result
||||- yardstick.result
||||- dyconit.result
```
## Preparing an Experiment

If this is your first experiment, create a `<timestamp>-<name>` directory in your `experiments` directory.
If you have already completed experiments, create a new `<timestamp>-<name>` directory if you want to change any of your
resources. If not, you can reuse the existing directory.

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

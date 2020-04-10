# Scientific PyTorch Template

1. [Introduction](#introduction)
2. [Implementation Details](#implementation-details)
   1. [Controller](#controller)
   2. [Trainer](#trainer)
      1. [Components](#components)
      2. [Logger](#logger)
      3. [Monitor](#monitor)
3. [Create a new Project](#create-a-new-project)

## Introduction

In this repository we present a PyTorch template to facilitate scientific research using deep learning. The framework is designed to hide most of the boilerplate code and expose only intuitive interfaces that allow the user to tailor the pipeline to their specific needs. We focus on simplifying the scheduling of several experiments and create result directories that ensure full reproducibility. Furthermore, we put an emphasis on informative status updates and support both, epoch-based and iteration-based training. In the following, we describe the individual components of the template and provide information on how to create a new project.

## Implementation Details

At the highest level of abstraction, the template is based on the concepts of _Sessions_ and _Experiments_. A session is associated with a single config file that specifies all relevant parameters for training and evaluation. The session-level config file can contain parameter values for an arbitary number of experiments. If parameters for multiple experiments are provided, all combinations are extracted automatically and parsed into experiment-level configs which are then used to create individual experiments. This setup allows us to formulate hyper-parameter searches as a single session with several experiments. An illustration of the framework is displayed in the figure below. In the following sections we provide a more detailed explanation of the pipeline's main components.

<p align="center">
  <img src="images/Illustration.svg" width="100%">
</p>

### Controller

The controller object operates at the highest level of the framework and controls the execution of individual experiments within a session. Its main task consists of analyzing the session-level config file and extracting all parameter combinations, parsing them into experiment-level config files. In order to be able to intuitively define several experiments from a single config file, we rely on a simple syntax, which we describe below. Given a set of experiment-level configs, the controller merely instantiates an iterator over all experiments, leaving the execution of the training process to the respective trainer object.

#### Session-level Config

In order to be able to create several experiments using a single config file, we rely on a simple mechanism. Every parameter
for which several values should be tested, is to be provided in the form of a list containing all values. Since there might exist other parameters for which a list is required to specify a single experiment, e.g. a list of loss weights, we choose to add a trailing underscore to indicate parameter values for several experiments. For example, the following config file would create six experiments for all possible combinations of batch size and learning rate, using the same loss weights for all experiments.

```
learning_rate_: [0.001, 0.01, 0.1]
batch_size_: [8, 16]
loss_weights: [1.0, 10.0, 0.001]
```

For now, the template only supports grid searches. However, the concept can be extended to other methods such as random search or bayesian optimization by slightly adjusting thes session-level config syntax.

### Trainer

The trainer object operates at the experiment-level and has no knowledge of other experiments being scheduled within the same session. It is instantiated by the controller, passing only the config file for the respective experiment and the path to the result directory. The trainer handles the training and evaluation for one fixed set of parameters and is further responsible for the visualization and saving of the experimental results. In the figure below we provide an illustration of the trainer's modules and explain their individual functions in the following sections.

#### Components

#### Logger

#### Monitor

## Create a new Project

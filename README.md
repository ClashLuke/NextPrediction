# NextPrediction
An API providing metadata of the future.

## Structure
* [**Getting Started**](#getting-started)
    + [Noteook](#Notebook)
    * [Local](#Local)
* [**Description**](#Description)
    * [Badges](#badges)
    * [Components](#components)
    * [Design](#design)
    * [Repository](#repository)
    * [Method](#method)
    * [Performance](#performance)
    * [Application](#application)
    * [Connection to existing projects](#connection-to-existing-projects)

## Getting Started
### Notebook
This repository contains an example [notebook](https://github.com/ClashLuke/NextPrediction/blob/master/example.ipynb) 
demonstrating and explaining the usage of this library. \
You can find the basic usage as well as configuration options and example outputs in this .notebook.\
A generated sample, proving the validity of the generation can be found [here](#Application).

### Local
To get started on a local machine, you should first clone the repository recursively, to ensure the "LocAtE" library is
cloned as well.
```
git clone --recursive https://github.com/ClashLuke/NextPrediction 
```
Afterwords you can go straight ahead into your python file or console and import the AutoEncoder class from the
NextPrediction package using `from NextPrediction import AutoEncoder`.\
Lastly, all that needs to be done to set it up is to create an AutoEncoder instance, add a dataset to it and start the
training for, say, 10 epochs.
```
model = AutoEncoder(feature_list=[96, 96], inputs=1)
model.add_dataset("NextPrediction/nextbike.csv")
model.train(10)
```

## Description
### Badges
 | Type | Status |
 | --- | --- |
 | Readability | [![BCH compliance](https://bettercodehub.com/edge/badge/ClashLuke/NextPrediction?branch=master)](https://bettercodehub.com/) |
 | Conformity | [![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/ClashLuke/NextPrediction/?ref=repository-badge) | 
 
 
### Components
1. **Predictions**: of user-behaviour, accessible under the same license as the source data.
2. **Extendability**: with any other data, such as user_id, bike_id or the operators favorite_colour.
3. **Open Access**: for the API as well as the code, allowing for private deployment and public datasets.


### Design
The three core components defined above shall not only be implemented in such a way that the code is both readable and
maintainable by foreign developers, but instead should also consider the technical feasibility and scalability of an
implemented design. With this premise, the only way to make proper predictions for sequential data is a
[transformer](https://arxiv.org/abs/1706.03762), as recurrent neural networks quickly become unfeasible for big data.
That's why a convolutional neural network is used in sequence prediction, even though they are more commonly seen in
image recognition. \
With all this set up, the most natural thing to do is to take the state-of-the-art LocAtE library, plug in the datasets
and wait for results.

TL;DR: The code is stolen from LocAtE

### Repository
This repository contains a basic python script (dubbed "main.py") showing demonstrating the usage of a python api
implemented to train, test and deploy machine learning models. It also contains a configuration file, located in the
libs folder. This file can be used to increase the networks width, depth and even training loop.\
The code is also accompanied by two jupyter notebooks, which can be executed with a single click of a button using the free GPU
quota in google's [colaboratory](https://colab.research.google.com/). \
In summary, this repository works as an example for a LocAtE-based application and does not contain any API code but
instead only the bare backend, allowing for more freedom in integration and design.

### Method
Before jumping into the [#performance](#performance) section, let's discuss the methodology used for training, testing
and evaluation first.

#### Data preparation
Neural networks tend to like zero-centered input data, especially if its standard deviation is one as well. Using this
knowledge, each collumn of the dataset was first subtracted by its mean just to then be divivided by its standard deviation.
This results in a nice, zero-centered dataset, the machine can learn with easily.

#### Input and target
This model is built to denoise the inputs. More specific, some inputs are zeroed out and have to then be recovered by
the model. Therefore the target is the actual data point, while the input is the same entry, but without either start
or end time as well as only one location. Therefore four input permutations
(start_time+start_location, start_time+end_location,...) exist. Therefore, knowing that the input and output data both 
are zero-centric with a std of one, we use the absolute distance between target and output as a measurement of
performance. 

#### Train test eval
Instead of training on the entire dataset, we first split of 20% for testing during the training and another 10% for
evaluation after the training has finished.\
We then train one epoch on the training dataset, which is directly followed by a testing phase, on the entire testing
dataset. While those are being computed, the most recent error is displayed. Afterwords an average across all batches is 
taken and printed.\
During the testing phase, one can also optionally (opt-out) print a list of elements created by the network, to manually
evaluate the performance. 

### Performance
For a basic machine-learning based backend the most important metric isn't how responsive its UI is or how innovative the
idea is. For a program everyone already needs, the most important measurement to provide is the raw performance data.\
With less than 0.4 average error after training the model for half an hour on a low-end CPU, one can comfortably say that the
convergence is fast and the results are powerful. However, while still having visible convergence, they aren't close to 
good enough for production-level accuracy. That's why a testing environment using a jupyter notebook on colaboratory
with 17 million parameters was deployed, yielding similar results in minutes.\
The best raw performance achieved, with two hours of training on CPU, is an incredible 0.3  . With this accuracy, 
predictions can not just be made but also relied on.

### Application
To# ensure that those outputs (at a loss of 0.25) are not pure bogus, the first list in the last list of lists seen in
the example [notebook](https://github.com/ClashLuke/NextPrediction/blob/master/example.ipynb) was fed into google 
[maps](https://www.google.com/maps/dir/50.756832750298166,+13.326533113131694/50.71492613025646,+13.255191700282955/@50.7514885,13.2113978,12z/data=!3m1!4b1!4m10!4m9!1m3!2m2!1d13.3265331!2d50.7568328!1m3!2m2!1d13.2551917!2d50.7149261!3e1).
![https://github.com/ClashLuke/NextPrediction/blob/master/readme_resources/example.png](https://github.com/ClashLuke/NextPrediction/blob/master/readme_resources/example.png)
With the console showing the time difference (in minutes) between start and end, and the map showing start and end points
as well as the expected time to get from one point to another.
While 49 minutes instead of 34 minutes does seem a little far-fetched, you can't know what the used did between those stops.\
Visualizing the generated data shows that decent geolocation data can be generated as well as more-or-less accurate
timestamps, implying that real-world application is possible.

### Connection to existing projects
#### NextBike
As this model can be used to accurately predict where a person will go, just by knowing when and where they started
their trip, nextbike could give an incentive to people to move towards the approximated destination of the current driver,
making this system more peer-to-peer and less station-based. This would further improve the downtime statistics and
therefore improve the overall efficiency of the entire company. 

#### Jelbi
Jelbi could build around such a system by enriching it with their station. As you already know where the ride will end
before it actually does, it would be possible to reinforce the incentive of stopping by a station rather than dropping 
the bike off at their home. Additionally some form of "bike-juicer" could become popular if there is financial incentive
to be at the estimated end at the predicted time to take the bike and move it back to a station. This would significantly
simplify the lives of many people without being intrusive, almost ensuring a success.

#### Swobbee
Since this system can be expanded to e-scooters and any piece of technology, it's possible to accurately estimate the
battery usage as well as the route of the trip. Using this information, one can drop the user a hint, pointing
towards the nearest swobbee station, so that they can quickly change their battery, without having to wait for the scooter
to charge up again or even leaving the ecosystem entirely in favour of a taxi.


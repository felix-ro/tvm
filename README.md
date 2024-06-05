<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

<img src=https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png width=128/> 

----

Apache TVM is a compiler stack for deep learning systems. It is designed to close the gap between productivity-focused deep learning frameworks and performance and efficiency-focused hardware backends. TVM works with deep learning frameworks to provide end-to-end compilation to different backends.

This version of TVM enables the use of **Bayesian Optimization** as a search strategy in MetaSchedule.

## Installation
We recommend following [this guide](https://llm.mlc.ai/docs/install/tvm.html) or the steps below:

1. Clone and build TVM-Bayesian-Optimization
    ```sh
    $ git clone --recursive https://github.com/felix-ro/TVM-Bayesian-Optimization
    $ cd TVM-Bayesian-Optimization
    $ mkdir build
    $ cp cmake/config.cmake build/
    $ cd build
    # Now configure the `config.cmake` file in the build directory
    $ cmake ..
    $ make -j4
    ```
2. Standard TVM Python dependencies
    ```sh 
    $ pip install numpy decorator attrs tornado psutil 'xgboost>=1.1.0' cloudpickle bayesian-optimization
    ```
3. Add TVM to python path
    ```sh
    $ export PYTHONPATH=/path-to-tvm-unity/python:$PYTHONPATH
    ```

## How To Use
General usage examples with ready-to-use scripts can be found [here](https://github.com/felix-ro/TVM-Benchmarking).

(A performance evaluation and further details will be added here soon)

<!-- ## Table of Contents
1. [Introduction](#introduction)
2. [Why Bayesian Optimization](#why-bayesian-optimization)
3. [Configurations & How To Use](#configurations--how-to-use)
4. [Performance Evaluation](#performance-evaluation)
5. [Installation](#installation)

## Introduction
Machine Learning (ML) models require high-performance tensor programs for optimal inference latencies. Traditional compilation systems rely on hardware-specific libraries, which include handwritten high-performance operator primitives. However, due to the libraries’ substantial engineering requirements, they often struggle to keep up with the hardware development cycles and the evolution of the AI landscape. _Deep Learning Compilers_ aim to address this challenge by creating a search space of semantically equivalent programs, i.e., programs that produce the same output but differ in their internal structure. The created space is then searched for a performant implementation. However, as the search space often includes billions of programs, an efficient search strategy is required to minimize the number of programs that must be benchmarked until an efficient implementation is discovered.

Bayesian Optimization (BO) is a sample-efficient sequential design strategy for the optimization of expensive to evaluate objective functions. BO has previously been used, e.g., for hyperparameter optimization problems and material discovery, and has shown to be more sample-efficient than the Genetics Algorithm Evolutionary Search (ES). In this fork of Apache TVM, we implemented BO as a novel search strategy for the scheduling system MetaSchedule and evaluated it against the current ES strategy.

In the evaluation below, we find that our BO search strategy can discover efficient tensor programs with significantly fewer trials than TVM’s ES strategy for CPU targets. When generating code for CPU targets, our search strategy delivers an up to 10% decrease in end-to-end latency when using the same number of trials, corresponding to a reduction of up to 68% in trials over the state-of-the-art. For GPU targets, the performance of the BO strategy is limited by the unique characteristics of the black-box objective function.

## Why Bayesian Optimization
Current state-of-the-art search strategy in MetaSchedule uses the Genetics Algorithm Evolutionary Search (ES) with search durations ranging from hours to days. ES is a form of random walk and takes inspiration from evolution by applying random mutations to a population of performant candidates. It builds on the idea that performant candidates are in proximity to other performant candidates. ES is widely used for hyperparameter optimization problems. Another highly effective parameter optimization strategy has shown to be Bayesian Optimization (BO). BO is often considered the most sample-efficient search strategy regarding the number of evaluations required to find a performant solution. Unlike ES, which relies on random mutations, BO employs a more informed approach by considering previously evaluated points when selecting the next point to evaluate in the search space. BO has been shown to outperform ES regarding the number of evaluations required to identify high-performing solutions and is especially suitable for expensive to-evaluate functions.

## Configurations & How To Use
- in progress

## Performance Evaluation
- in progress

![Apple M3 Max Results](/assets/m3_max_results.png)

![M3 Max MobileNet Long Run](/assets/mobilenet_long_run.png)

![CPU Trial Reduction](/assets/cpu_trial_reduction.png)

![NVIDIA A100 Results](/assets/a100_results.png) -->
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

## Performance Analysis
For a brief overview of the results, see the slides [here](assets/tvm_bayesian_optimization.pdf) (fyi Safari sometimes fails to render them). For a more in-depth discussion of the project, see the report [here](https://www.cl.cam.ac.uk/~ey204/pubs/MPHIL_P3/2024_Felix.pdf).
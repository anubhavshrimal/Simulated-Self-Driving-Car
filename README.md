# Simulated Self Driving Car

## Overview

This is the code for training a machine learning model to drive a simulated car using Convolutional Neural Networks. I used Udacity's [self driving car simulator](https://github.com/udacity/self-driving-car-sim) as a testbed for training an autonomous car.

## Dependencies

1. You can install all dependencies by running one of the following commands

    You need a [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html) to use the environment setting.

    ```python
    # Use TensorFlow without GPU
    conda env create -f environments.yml

    # Use TensorFlow with GPU
    conda env create -f environment-gpu.yml
    ```

    Or you can manually install the required libraries (see the contents of the environemnt*.yml files) using pip.

 2. Download Udacity's self driving car simulator from [here](https://github.com/udacity/self-driving-car-sim).

## Usage

### Clone this repository

Type the following commands in your terminal:
```bash
cd path/to/directory/you/like/
git clone https://github.com/anubhavshrimal/Simulated_Self_Driving_Car.git
cd Simulated_Self_Driving_Car/
```

### Run the pretrained model

Start up the [Udacity self-driving simulator](https://github.com/udacity/self-driving-car-sim), choose a scene and press the Autonomous Mode button.  Then, run the model as follows:

```python
python drive.py model.h5
```

### To train the model

1. Start up the Udacity self-driving simulator, choose a scene and press the Training Mode button.

2. Then press `R key` and select the **data** folder, where our training images and CSV will be stored.

3. Press R again to start recording and R to stop recording. Let the processing of video complete.

4. You should do somewhere between 1 and 5 laps of the simulated road track.

5. The run the following command:

    ```python
    python model.py
    ```

This will generate a file `model-<epoch>.h5` whenever the performance in the epoch is better than the previous best.  For example, the first epoch will generate a file called `model-000.h5`.

## Vote of Thanks

NVIDIA's paper: [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) for the inspiration and model structure.

[Siraj Raval](https://github.com/llsourcell) & [naokishibuya](https://github.com/naokishibuya) for the knowledge and code help.




# Exploiting Hankel–Toeplitz Structures for Fast Computation of Kernel Precision Matrices
Supplementary material consisting of GPJax code for the paper in the title.

The code does **not** include MATLAB code for the underwater magnetic field experiment - we do however include code for the hyperparameter optimization of that data set. The prediction plots are are also easily reproduced by the code contained here, but the timings are unfortunately not.
The necessary data files for the precipitation and underwater data sets are unfortunately not included.
Please contact the authors of this repo for access to those data sets.

# Running the code
It is highly recommended to build the Docker image defined by the Dockerfile - see [Docker image](#docker-image).
If you do **not** want to do so, please use the requirements.txt to set up your Python environment which should be enough for you to run the code.

The primary files for reproducing the results in the paper are: `precipitation-fitting.py`, `precipitation-timing.py` and `hankel_timing.py`. These are run in the following way:

```
python precipitation-fitting.py bf.m=45 experiment=precipitation

python precipitation-timing.py -m experiment=precipitation

python hankel_timing.py
```
You can use whatever value for `bf.m` that you wish, but 45 should reproduce the results in the paper.

The plots can then be reproduced by opening each respective Jupyter notebook and following the instructions in those.

## Docker image
To build the Docker image run
```
docker build -t fasthgp .
```
To run, please use the included `docker-compose.yml` file by running
```
docker-compose up
```
This will start a Jupyter lab session for you that you can run in any browser.
To run the raw Python scripts, it's most convenient to do so within the Docker container itself. Run
```
docker exec -it <enter-docker-container-name-here> /bin/bash
```
navigate to `work` and you should be able to run any of the Python scripts in [Running the code](#running-the-code).

## Pipenv (not tested thoroughly)
You should also be able to run the code using e.g. `pipenv`. The following are instructions to setup that environment.
Create a virtual environment with Python 3.11
```
python3 -m pipenv --python 3.11
```
Install dependencies (skip-lock to speed up the process)
```
python3 -m pipenv install --skip-lock
```
Run the virtual environment
```
python3 -m pipenv shell
```
Now you should be able to run everything in [Running the code](#running-the-code). 
Note that you will need to start a Jupyter lab instance by yourself (the environment contains the package at least).
# enseisro
ENsemble asteroSEISmology using numpyRO

## Building the conda environment
Execute the following set of commands in your terminal to install and activate the ```conda``` environment ```jax-gpu``` (don't worry, it'll work
even if you do not have access to a gpu device).
```
conda create --name jax-gpu python=3.8
conda activate jax-gpu
pip install numpyro
pip install --upgrade pip
pip install --upgrade jaxlib==0.1.62+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
At this point, you should be ready to run the sample code ```sample_NUTS.py``` if the installation was successful.

## Test environment installation
Prior to running the basic NumPyro code, we shall test a couple of lines to see if all is well and good. To do this, open the  ```python``` prompt
by typing
```
python
```
Once you are in, execute the following code snippets
```
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
```
This should show a warning saying "WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)". This is OK!
Then run one more check by executing
```
import numpyro
numpyro.set_platform('cpu')
```
This should return nothing if it is installed correctly. If so, you're all set!

## Run the sample code on linear regression
At this point, you should be able to run the code ```sample_NUTS.py``` which carries out linear regression using NumPyro.
```
python sample_NUTS.py cpu 1
```
This should run the code with a progressbar and end with printing the optimal values of the parameters.

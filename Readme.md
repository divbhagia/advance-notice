
This repository contains the code for the paper "Duration Dependence and Heterogeneity: Learning from Early Notice of Layoff" by Div Bhagia. 

## Software Requirements

The project was built using Python 3.12.

You can install Python 3.12 using `pyenv`.
```bash
pyenv install 3.12
```
To create a virtual environment for this project using `pyenv`, you need `pyenv-virtualenv`. To create a new virtual environment, use the following command:
```bash
pyenv virtualenv 3.12.x env-notice
```
Here, `x` is the version of python you are using and `env-notice` is the name of the virtual environment. You can give it any name you like. Note that `pyenv` manages environments in a central location but allows you to "activate" them in specific directories. 

## Running the code

If you used `pyenv` to create a virtual environment as described above, you can clone the repository, install the required packages, and run all the code using the following commands:
```bash
git clone git@github.com:divbhagia/notice.git
cd notice
pyenv activate env-notice
pip install -r requirements.txt
make 
```
If you did not create a virtual environment, just omit the `pyenv activate env-notice` command.


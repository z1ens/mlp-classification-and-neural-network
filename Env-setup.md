# TL;DR: Setting up Python = 3.8, conda environment

If you new to python, first set up the environment. We mostly use conda environment here as an example, of course you have other alternatives. So right now we start working on it.

## MacOS as an example

I am using Miniconda for a quick and light-weighted set-up.
> Miniconda supplies only the package management system. This allows to set up minimalistic environments when the disk space is limited (e.g., pool PCs offered by ISP). If you have Anaconda already installed and would like to use it instead, the instructions below should also work.

### Set up Miniconda

 Install Miniconda following the [instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
 
 The required packages for the assignment are written in the requirements.txt file.

 Next we create an conda environment called "cv" with following commands:
 ```
 conda env create -f requirements.txt -n cv
 conda activate cv1  // source activate cv for windows users
 ```

 Till here, you should have successfully activated our "cv" conda environment.

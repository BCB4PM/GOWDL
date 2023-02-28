# GOWDL: Gene ontology-driven wide and deep learning architecture

This repository provides the source code for a gene ontology-driven wide and deep learning architecture for cell typing of single-cell RNA-seq data.

To write and run our Python scripts we used Python version 3.9.7. We provide a read-to-use conda environment configuration file to install all required python
packages. To import the conda environment into your system go to the "conda_env" directory and type the following (after installing Anaconda):

```
$ conda create --name <env> --file env.txt
```

where \<env\> is name of your environment to be created from file. You can now activate your \<env\> imported environment simply typing:

```
$ conda activate <env>
```

Alternatively you can install single python packages when needed to run source code. Once the environment is configured with all the needed packages, you have to 
unzip ".zip" files in the same directory where they are.

## Example dataset

We provide a dataset to perform a cell-type classification with our model. However, it is possible to perform classifications with other datasets, following instructions within scripts inside the "scripts" directory. 
To run the script with our prepared dataset, please download and unzip the "data.zip" file from the external link contained in the "scripts/data" directory and go to the "scripts" directory. Then run the following line from the command line.

```
$ python goWideDeepLearning.py --method multiclass --kernel 5
```

## Usage

To perform a cell-type classification with an external dataset (in cells x genes matrix form), you have to align dataset cell type names in "cellsFiltering.py" file
with cell types within "cellmatch.csv" file and put your file inside "scripts/data" directory and change tissue filtering in "relevantGenesExtraction.py" ("tissueType"
or "cancerType"). You have also to unzip "util.zip". You can then run all the scripts in the following order:

```
$ python cellsFiltering.py
$ python genesDictFiltering.py
$ python relevantGenesExtraction.py
$ python createKernelDataset.py --kernel [kernel size]
$ python goWideDeepLearning.py --method multiclass --kernel [kernel size]
```

where [kernel size] in steps 4 and 5 must be the same.

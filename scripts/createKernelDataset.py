# This script defines needed functions to create a dataset having cells as rows and gene tuples as columns, each of which represents a
# kernel that contains the closest genes, from original dataset. This so-called kernel dataset is created for the CNN model, as it is 
# made up of kernels. Specifically, this file uses "OrderedGeneNames_filtered.pkl" file (generated previously), which contains the 
# genes distance dictionary. To run this script, you must give the size of the kernel you want to use to generate the kernel dataset as
# input from command line. So this script runs correctly if formatted as follows:
# python createKernelDataset.py --kernel [kernel size] 
# Example: python createKernelDataset.py --kernel 3


# Import of utility libraries.
import pandas as pd
import numpy as np
import argparse
import sys
from util import readDictionary, writeDataset, extractBioRelevantGenes, printSummary


# This function returns a list with column names of the kernel dataset that will be created, based on the input kernel size.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing cells-genes expression values.
# kernelSize : int
#       Kernel size.
# genesDictionary : dict
#       Genes distance dictionary.
#
# Returns
# ------------------------------------------------------------------------------------------------
# columnIndexList : list of strings
#       List with column names of the kernel dataset that will be created.
#
def createColumnIndexList(dataset, kernelSize, genesDictionary):
    columnIndexList = []
    count = 1   # Help variable to define column names.

    for column in dataset.columns:
        genesList = genesDictionary[column][0: kernelSize - 1]
        tmpList = [column]
        for i in range(0, len(genesList)):
            tmpList.insert(0, genesList[i]) if i % 2 == 0 else tmpList.append(genesList[i])
        for element in tmpList:
            columnIndexList.append(element + "__" + str(count))
        count = count + 1
    
    return columnIndexList


# This function returns kernel dataset having:
# - row indices of original dataset (cell names) as row indices
# - new column indices generated based on kernel size as column indices
# - new rows generated based on kernel size as rows
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing cells-genes expression values.
# kernelSize : int
#       Kernel size.
# genesDictionary : dict
#       Genes distance dictionary.
# 
# Returns
# ------------------------------------------------------------------------------------------------
# kernelDataset : pandas.DataFrame
#       Kernel dataset.
#
def createKernelDataset(dataset, kernelSize, genesDictionary):
    print("Kernel dataset creation with kernel size of " + str(kernelSize) + "...")
    columnIndex = createColumnIndexList(dataset, kernelSize, genesDictionary)
    newColumnList = [[]]    # # Help variable with kernel dataset column names.

    for column in columnIndex:
        oldColumn = column.split("__")
        newColumnList.append(dataset[oldColumn[0]])
    newColumnList.pop(0)

    newColumnList = np.array(newColumnList)
    newColumnList = newColumnList.T

    # Kernel dataset creation.
    kernelDataset = pd.DataFrame(data = newColumnList, index = dataset.index, columns = columnIndex)

    return kernelDataset



# Main function.
if __name__ == '__main__':

    # This script requires an input parameter --kernel.
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--kernel", type = str, default = 3, help = "kernel size")
        args = vars(ap.parse_args())
        kernel = args['kernel']
        kernel = int(kernel)
    except(ValueError):
        print("Error: give an integer number as parameter.\n" + 
            "Example: python createKernelDataset.py --kernel 3")
        sys.exit(1)

    # File path.
    filePath = "data/"

    try:
        # Reading of the dataset containing the gene-cell matrix with gene expression values.
        print("Reading gene expression values dataset...")
        dataset = pd.read_pickle(filePath + "Dataset_filtered3.pkl")

        # Reading of the dataset containing cell types.
        print("Reading cell types dataset...")
        labels = pd.read_pickle(filePath + "Labels_filtered.pkl")

        # Reading of genes distance dictionary.
        genesDictionary = readDictionary(filePath + "OrderedGeneNames_filtered.pkl")

        # Kernel dataset creation.
        kernelDataset = createKernelDataset(dataset, kernel, genesDictionary)

        # Dataset after kernel creation step.
        print("KERNEL DATASET INFORMATION\n")
        printSummary(kernelDataset, labels)
        
        # Appending biologically relevant genes to the dataset.
        datasetBioRelevant = pd.read_pickle(filePath + "Dataset_filtered2.pkl")
        bioRelevantGenes = extractBioRelevantGenes(filePath + "biologicalRelevantGenes.pkl")
        datasetBioRelevant = datasetBioRelevant[bioRelevantGenes]
        kernelDataset = pd.concat([kernelDataset, datasetBioRelevant], axis=1)
        
        # Writing of dataset without relevant genes to a file.
        writeDataset(kernelDataset, filePath + "Dataset_filtered4(kernel" + str(kernel) +").pkl")

    # File not found exception.
    except(FileNotFoundError):
        print("Error: file not found.")
        sys.exit(1)
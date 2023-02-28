# This script defines needed functions for genes based filtering. Specifically, this file uses the "OrderedGeneNames.pkl" 
# file which contains the genes distance dictionary (previously calculated with GOGO algorithm). Furthermore, this file
# generates two files: the "removedGenes.txt" file, which contains the list of removed genes from the dataset as they 
# are not within genes distance dictionary; the file "OrderedGeneNames_filtered.pkl" which contains filtered genes distance
# dictionary, taking into account only the genes within the dataset.


# Import of utility libraries.
import pandas as pd
import sys
from util import readDictionary, writeDictionary, writeDataset, printSummary


# This function writes the list of removed genes from dataset to a file.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# list : list of strings
#       List of genes names removed from dataset after filtering based on genes distance dictionary.
# filePath : string
#       File path where write the list.
#
def writeRemovedGenesDict(list, filePath):
    print("Writing list of removed genes from dataset to a file...")
    with open(filePath, "w") as file:
        file.write("List of removed genes from dataset:\n\n[")
        if list == []:
            file.write("]\n\n")
        else:
            for i in range(0, len(list) - 1):
                file.write(list[i] + ", ")
            file.write(list[-1] + "]\n\n")

# This function filters input dataset, removing genes that are not within the genes distance dictionary. It writes the filtered
# dataset to a file and the list of removed genes from the original dataset to a second file.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing cells-genes expression values.
# genesDictionary : dict
#       Genes distance dictionary.
# filePath : string
#       File path where write the files.
# Returns
# ------------------------------------------------------------------------------------------------
# filtered_dataset : pandas.DataFrame
#       Filtered dataset containing only genes within the genes distance dictionary.
#
def filterDataset(dataset, genesDictionary, filePath):
    print("Dataset filtering...")
    dataset_gene_list = dataset.columns.tolist()
    genes_distance_list = genesDictionary.keys()
    filtered_dataset = dataset
    removedGenes = []

    for i in dataset_gene_list:
        if i not in genes_distance_list:
            removedGenes.append(i)

    filtered_dataset.drop(removedGenes, axis = 1, inplace = True)
    
    # Memorizzazione della lista di geni rimossi dal dataset in un file.
    writeRemovedGenesDict(removedGenes, filePath + "removedGenes.txt")

    return filtered_dataset


# This function filters input genes distance dictionary, removing genes that are not within input gene expression values dataset.
# Finally it returns filtered dictionary.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing gene expression values (already filtered).
# genesDictionary : dict
#       Genes distance dictionary.
#
# Returns
# ------------------------------------------------------------------------------------------------
# filtered_dictionary : dict
#       Filtered genes distance dictionary.
#
def filterGenesDictionary(dataset, genesDictionary):
    print("Genes distance dictionary filtering...")
    dataset_gene_list = dataset.columns.tolist()
    genes_distance_list = genesDictionary.keys()
    filtered_dictionary = genesDictionary
    removedGenes = []

    for i in genes_distance_list:
        if i not in dataset_gene_list:
            removedGenes.append(i)

    # Rimuove le chiavi (i geni) che non sono presenti nel dataset dal dizionario
    for element in removedGenes:
        filtered_dictionary.pop(element, None)

    for key in filtered_dictionary.keys():
        filtered_dictionary[key] = list(set(filtered_dictionary[key]) - set(removedGenes))

    return filtered_dictionary



# Main function.
if __name__ == '__main__':

    # File paths.
    filePath = "data/"
    filePath2 = "../"

    try:
        # Reading of the dataset containing the gene-cell matrix with gene expression values.
        print("Reading gene expression values dataset...")
        dataset = pd.read_pickle(filePath + "Dataset_filtered.pkl")

        # Reading of the dataset containing cell types.
        print("Reading cell types dataset...")
        labels = pd.read_pickle(filePath + "Labels_filtered.pkl")

        # Reading of genes distance dictionary.
        genesDictionary = readDictionary(filePath2 + "OrderedGeneNames.pkl")

        # Filtering of genes distance dictionary.
        genesDictionary = filterGenesDictionary(dataset, genesDictionary)

        # Writing of filtered genes distance dictionary to a file.
        writeDictionary(genesDictionary, filePath + "OrderedGeneNames_filtered.pkl")
        
        # Dataset before filtering step based on genes distance dictionary.
        print("\nDATASET INFORMATION BEFORE FILTERING BASED ON GENES DISTANCE DICTIONARY.\n")
        printSummary(dataset, labels)

        # Dataset filtering.
        dataset = filterDataset(dataset, genesDictionary, filePath)

        # Dataset after filtering step based on genes distance dictionary.
        print("\nDATASET INFORMATION AFTER FILTERING BASED ON GENES DISTANCE DICTIONARY\n")
        printSummary(dataset, labels)
        
        # Writing of filtered dataset to a file.
        writeDataset(dataset, filePath + "Dataset_filtered2.pkl")

    # File not found exception.
    except(FileNotFoundError):
        print("Error: file not found.")
        sys.exit(1)
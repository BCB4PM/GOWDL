# This script defines needed functions to select biologically relevant genes from the dataset. Specifically, it uses "cellmatch.csv" and 
# "celltype2subtype.csv" files to perform genes extraction. It also writes three lists of relevant genes in the 
# "biologicalRelevantGenes.txt" text file: first one containing genes of normal cells, second one containing genes of cancer cells and 
# last one containing genes of both cell types. Finally, this file also generates a "biologicalRelevantGenes.pkl" file, which contains
# a dictionary with cell type as key, and list of biologically relevant genes within the dataset for that cell type as value.


# Import of utility libraries.
import pandas as pd
import sys
from util import writeDataset, printSummary,printDictionary, writeDictionary


# This function replaces each cell sub-type of input dataset with the corresponding main cell type. This function uses 
# the "celltype2subtype.csv" file to remove cell sub-types.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# cellmatch : pandas.DataFrame
#       Database containing cell types and sub-types.
#
# Returns
# ------------------------------------------------------------------------------------------------
# cellmatch : pandas.DataFrame
#       Database containing only cell types (no sub-types).
#
def removeSubClasses(cellmatch):
    filter = pd.read_csv("../celltype2subtype.csv", header = 0, index_col = 0)
    for i in range (0, len(cellmatch)):
        for j in range (0, len(filter)):
            if(cellmatch["cellName"][i] == filter["Cell type"][j]):
                cellmatch["cellName"][i] = filter["Cell-type"][j]
    return cellmatch


# This function writes the list of biologically relevant genes within both the dataset and the "cellmatch.csv" file 
# to a file.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# list : list of strings
#       List of biological relevant gene names for tissue dataset.
# cellType : string
#       Selected cell type.
# filePath : string
#       File path where write the list.
#
def writeRelevantGenesInfo(list, cellType, filePath):
    print("Scrittura della lista di geni biologicamente rilevanti per '" + cellType + "' in un file di testo...")
    with open(filePath, "a") as file:
        file.write("Lista dei geni biologicamente rilevanti per '" + cellType + "':\n[")
        if list == []:
            file.write("]\n\n")
        else:
            for i in range(0, len(list) - 1):
                file.write(list[i] + ", ")
            file.write(list[-1] + "]\n\n")


# This function writes list of biologically relevant genes within the "cellmatch.csv" file but not within the tissue
# dataset to a file.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# list : list of strings
#       List of removed biological relevant gene names, as they are not within tissue dataset.
# cellType : string
#       Selected cell type.
# filePath : string
#       File path where write the list.
#
def writeRemovedRelevantGenes(list, cellType, filePath):
    with open(filePath, "a") as file:
        file.write("List of removed genes for '" +
            cellType + "':\n[")
        if list == []:
            file.write("]\n\n")
        else:
            for i in range(0, len(list) - 1):
                file.write(list[i] + ", ")
            file.write(list[-1] + "]\n\n")


# This function returns a dictionary with cell types of input dataset as keys and the relevant genes lists
# for those cells as values.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Tissue dataset containing cells information.
# cellsList: list of strings
#       List of cells within tissue dataset.
# genes_set: set of strings
#       Genes set within tissue dataset.
#
# Returns
# ------------------------------------------------------------------------------------------------
# cellsWithRelevantGenesDict : dict
#       Dictionary with cell types of input dataset as keys and the relevant genes lists for those cells as values.
#
def createCellsWithRelevantGenesDictionary(dataset, cellsList, genesList):
    # Dictionary creation.
    cellsWithRelevantGenesDict = dict()

    for cell in cellsList:
        tmp = dataset[(dataset["cellName"] == cell)]
        if(tmp.empty == False):
            genes_set = set(tmp["geneSymbol"])
            genes = list(genes_set.intersection(genesList))
            key_value_pair = {cell : genes}
            cellsWithRelevantGenesDict.update(key_value_pair)
    
    return cellsWithRelevantGenesDict


# This function returns the list of biologically relevant genes for the dataset and writes it to a file (depending from 
# the input cell type).
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing cells-genes expression values.
# labels : pandas.DataFrame
#       Dataset containing cell types.
# cellmatch : pandas.DataFrame
#       File for the extraction of relevant genes list.
# cellType : string
#       Cell type of selected cells from cellmatch file.
# filePath : string
#       File path where write the list.
#
# Returns
# ------------------------------------------------------------------------------------------------
# bioRelevantGenes : list of strings
#       Biological relevant genes list.
# bioRelevantDict : dict
#       Dictionary containg cell types and relevant genes per cell type.
#
def generateRelevantGenesList(dataset, labels, cellmatch, cellType, filePath):
    # List of cells within the dataset.
    cellsList = labels.drop_duplicates()
    cellsList = list(cellsList["Cell_type"])

    # List of genes within the dataset.
    genes_set = set(dataset.columns)
    
    # Biological relevant genes set.
    bioRelevantGenes = set()

    tmp = cellmatch[(cellmatch["cellName"].isin(cellsList))]

    if(cellType != "All cells"):
        tmp = tmp[(tmp["cellType"] == cellType)]

    bioRelevantDict = createCellsWithRelevantGenesDictionary(tmp, cellsList, genes_set)
    bioRelevantGenes = set(tmp["geneSymbol"])
    removedGenes = list(bioRelevantGenes.difference(genes_set))
    bioRelevantGenes = bioRelevantGenes.intersection(genes_set)
    bioRelevantGenes = list(bioRelevantGenes)

    # Writing of biological relevant genes list to a file.
    writeRelevantGenesInfo(bioRelevantGenes, cellType, filePath)

    # Writing of removed biological relevant genes to a file.
    writeRemovedRelevantGenes(removedGenes, cellType, filePath)

    return bioRelevantGenes, bioRelevantDict



# Main function.
if __name__ == '__main__':

    # File paths.
    filePath = "data/"
    filePath2 = "../"

    try:
        # Reading of the dataset containing the gene-cell matrix with gene expression values.
        print("Reading gene expression values dataset...")
        dataset = pd.read_pickle(filePath + "Dataset_filtered2.pkl")

        # Reading of the dataset containing cell types.
        print("Reading cell types dataset...")
        labels = pd.read_pickle(filePath + "Labels_filtered.pkl")

        # Reading "cellmatch.csv" file containing genes and cells information.
        print("Reading 'cellmatch.csv' file...")
        cellmatch = pd.read_csv(filePath2 + "cellmatch.csv", header = 0)
        
        # "cellmatch.csv" file filtering by human species and tissue.
        cellmatch = cellmatch[(cellmatch["tissueType"] == "Pancreas") & (cellmatch["speciesType"] == "Human")]
        cellmatch = cellmatch.reset_index()

        # Cell-subtypes removal from "cellmatch.csv" file.
        cellmatch = removeSubClasses(cellmatch)

        # Relevant genes list creation in 3 different cases.
        cellType = "Normal cell"
        bioRelevantGenes, bioRelevantDict = generateRelevantGenesList(dataset, labels, cellmatch, cellType, filePath + "biologicalRelevantGenes.txt")
        cellType = "Cancer cell"
        bioRelevantGenes, bioRelevantDict = generateRelevantGenesList(dataset, labels, cellmatch, cellType, filePath + "biologicalRelevantGenes.txt")
        cellType = "All cells"
        bioRelevantGenes, bioRelevantDict = generateRelevantGenesList(dataset, labels, cellmatch, cellType, filePath + "biologicalRelevantGenes.txt")

        # Dataset before biological genes removal step.
        print("\nDATASET WITH BIOLOGICAL RELEVANT GENES\n")
        printSummary(dataset[bioRelevantGenes], labels)
        print("LIST OF BIOLOGICALLY RELEVANT GENES PER CELL TYPE\n")
        printDictionary(bioRelevantDict)

        # Writing of dataset with only a column with relevant genes to a file.
        writeDataset(dataset[bioRelevantGenes], filePath + "BioRelevantGenes_dataset.pkl")

        # Writing of dictionary containing cell types and relevant genes per cell type to a file.
        writeDictionary(bioRelevantDict, filePath + "biologicalRelevantGenes.pkl")

        # Removing biological relevant genes from dataset.
        dataset.drop(bioRelevantGenes, axis = 1, inplace = True)
            
        # Dataset after biological genes removal step.
        print("\nDATASET WITHOUT BIOLOGICAL RELEVANT GENES\n")
        printSummary(dataset, labels)

        # Writing of filtered dataset to a file.
        writeDataset(dataset, filePath + "Dataset_filtered3.pkl")

    # File not found exception.
    except(FileNotFoundError):
        print("Error: file not found.")
        sys.exit(1)

# This script defines needed functions to filter dataset by cells. Specifically, it uses the "celltype2subtype.csv" file
# to remove sub-classes of a cell, treating them as main classes of the cell. It also removes cells having a number of
# examples below a defined threshold (0.05% of the total examples in the dataset).


# Import of utility libraries.
import pandas as pd
import sys
from util import writeDataset, printSummary


# This function encodes cell types within the dataset with "cellmatch.csv" file (filtered by "tissueType"), permitting
# subsequent genes extraction.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing cells-genes expression values (cells x genes matrix).
#
# Returns
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing cells-genes expression values (without columns of cell types).
# labels : pandas.DataFrame
#       Dataset containing mapping of cell types to strings.
#
def encodeCells(dataset):
    print("Cell types encoding based on 'cellmatch.csv' file...")
    tmp = {"Cell_type" : dataset["Cell_type"]}
    tmp = pd.DataFrame(data = tmp)
    labels = tmp.copy()

    dataset = dataset.drop(["Cell_type"], axis=1)
    
    # Orginal dataset classes.
    labelsCellsList = list(labels["Cell_type"].value_counts().index)

    # Classes in "cellmatch.csv" file with "tissueType" = current tissue.
    cellmatchList = ["Alpha cell", "Beta cell", "Delta cell", "Acinar cell", "Ductal cell", "Mesenchymal cell", "Unsure"]

    cellsToBeRemoved = []

    # This loop keeps only the cells belonging to certain cell types indicated.
    for i in range (0, len(labels)):

        # Keeping T cell
        if(tmp["Cell_type"][i] == labelsCellsList[0]):
            labels["Cell_type"][i] = cellmatchList[0]

        # Keeping B cell
        elif(tmp["Cell_type"][i] == labelsCellsList[1]):
            labels["Cell_type"][i] = cellmatchList[3]
        
        # Keeping Macrophage
        elif(tmp["Cell_type"][i] == labelsCellsList[2]):
            labels["Cell_type"][i] = cellmatchList[4]
    
        # Keeping Endothelial cell
        elif(tmp["Cell_type"][i] == labelsCellsList[3]):
            labels["Cell_type"][i] = cellmatchList[1]
        
        # Keeping Cancer-associated fibroblast
        elif(tmp["Cell_type"][i] == labelsCellsList[5]):
            labels["Cell_type"][i] = cellmatchList[2]
        
        # Keeping NK cell
        elif(tmp["Cell_type"][i] == labelsCellsList[6]):
            labels["Cell_type"][i] = cellmatchList[5]
        
        else:
            cellsToBeRemoved.append(labels.index[i])
    
    labels = labels[~labels.index.isin(cellsToBeRemoved)]
    dataset = dataset[~dataset.index.isin(cellsToBeRemoved)]

    return dataset, labels


# This function removes cell sub-types and considers them as examples of the corresponding main cell type. 
# This function uses the "celltype2subtype.csv" file to remove cell sub-types.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# labels : pandas.DataFrame
#       Dataset containing cell types and sub-types.
#
# Returns
# ------------------------------------------------------------------------------------------------
# labels : pandas.DataFrame
#       Dataset without cell sub-types (only main cell types).
#
def removeSubClasses(labels):
    filter = pd.read_csv("../celltype2subtype.csv", header = 0, index_col = 0)
    for i in range (0, len(labels)):
        for j in range (0, len(filter)):
            if(labels["Cell_type"][i] == filter["Cell type"][j]):
                labels["Cell_type"][i] = filter["Cell-type"][j]
    return labels


# This function removes classes that have fewer examples than the parameter passed as input to the function.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing cells-genes expression values (cells x genes matrix).
# labels : pandas.DataFrame
#       Dataset containing cell types.
# n_examples : float
#       Minimum number of examples that a class (a cell type) must have to be maintained.
#
# Returns
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Filtered dataset containing cells-genes expression values (cells x genes matrix).
# labels : pandas.DataFrame
#       Filtered dataset containing cell types.
#
def removeFewExamplesClasses(dataset, labels, n_examples):
    counts = labels["Cell_type"].value_counts()
    cellsToBeRemoved = labels[labels["Cell_type"].isin(counts[counts < n_examples].index)].index
    tmp = []
    for i in cellsToBeRemoved:
        tmp.append(i)
    labels = labels[~labels.index.isin(cellsToBeRemoved)]
    dataset = dataset[~dataset.index.isin(cellsToBeRemoved)]
    return dataset, labels



# Main function.
if __name__ == '__main__':

    # File path.
    filePath = "data/"
    
    try:
       # Reading of the dataset containing the gene-cell matrix with gene expression values and cell types.
        print("Reading gene expression values dataset...")
        dataset = pd.read_csv(filePath + "Pancreas_data.csv", sep = ";", index_col = 0, header = 0)

        # Minimum number of examples of each class (a class must be kept if it has at least 0.05% of examples of the total).
        n_examples = (0.05 * dataset.shape[0]) / 100

        # Original dataset.
        print("ORIGINAL DATASET INFORMATION\n")
        printSummary(dataset, dataset)

        # Cell types encoding based on file "cellmatch.csv".
        dataset, labels = encodeCells(dataset)
        print("\nDATASET INFORMATION AFTER CELL TYPES ENCODING STEP\n")
        printSummary(dataset, labels)

        # Sub-classes removal and substitution with main cell types.
        labels = removeSubClasses(labels)
        print("DATASET INFORMATION AFTER SUB-CLASSES REMOVAL STEP\n")
        printSummary(dataset, labels)

        # Removal of poorly represented classes.
        dataset, labels = removeFewExamplesClasses(dataset, labels, n_examples)
        print("DATASET INFORMATION AFTER FEW EXAMPLES CLASSES REMOVAL STEP\n")
        printSummary(dataset, labels)

        # Writing of filtered datasets to two different files.
        writeDataset(dataset, filePath + "Dataset_filtered.pkl")
        writeDataset(labels, filePath + "Labels_filtered.pkl")

    # File not found exception.
    except(FileNotFoundError):
        print("Error: file not found.")
        sys.exit(1)

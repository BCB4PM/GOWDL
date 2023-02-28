# This script defines utility functions needed by the other files.


# Import of utility libraries.
import pickle as pkl


# This function writes input dataset to a file.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset to write.
# filePath : string
#       File path where write the dataset.
#
def writeDataset(dataset, filePath):
    print("Writing dataset to a file...")
    dataset.to_pickle(filePath)
    

# This function prints dataset dimensions (cells x genes) and the number of examples for each class in the dataset.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing cells-genes expression values.
# labels : pandas.DataFrame
#        Dataset containing cell types.
#
def printSummary(dataset, labels):
    counts = labels["Cell_type"].value_counts()
    print("Dataset dimensions (cells x genes):\n" + str(dataset.shape) + "\n")
    print("Number of examples per cell type: \n" + str(counts) + "\n\n")


# For each input dictionary key this function prints the number of elements for that key.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dictionary : dict
#       Dictionary to print.
#
def printDictionary(dictionary):
    for k in dictionary.keys():
        print("'" + k + "': " + str(len(dictionary[k])) + " gene/s")
    print()


# This function reads a dictionary from input file path.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# filePath : string
#       File path of the dictionary.
#
# Returns
# ------------------------------------------------------------------------------------------------
# dictionary : dict
#       Dictionary in the key-value form.
#
def readDictionary(filePath):
    print("Reading dictionary from file...")
    with open(filePath, "rb") as file:
        dictionary = pkl.load(file)
    return dictionary


# This function writes input dictionary to a file.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dictionary : dict
#       Dictionary to be read.
# filePath : string
#       File path where write the dictionary.
#
def writeDictionary(dictionary, filePath):
    print("Writing dictionary to a file...")
    with open(filePath, "wb") as f:
        pkl.dump(dictionary, f)


# This function returns the biologically relevant genes list from input dictionary.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# filePath : string
#       File path of dictionary to read.
#
# Returns
# ------------------------------------------------------------------------------------------------
# bioRelevantList: list
#        List of biologically relevant genes extracted from input dictionary.
#
def extractBioRelevantGenes(filePath):
    # Dictionary with cell type as a key and relevant genes list to that cell type as value.
    dictionary = readDictionary(filePath)
    bioRelevantList = []

    for k in dictionary.keys():
        bioRelevantList.extend(dictionary[k])

    return bioRelevantList


# This function the biologically relevant genes list from input dictionary for each cell type.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# filePath : string
#       File path of dictionary to read.
#
# Returns
# ------------------------------------------------------------------------------------------------
# bioRelevantList: tuple
#       Tuple of biologically relevant genes lists extracted from input dictionary (each list 
#       within the tuple represents relevant genes for a cell type).
#
def extractBioRelevantGenesPerCell(filePath):
    dictionary = readDictionary(filePath)
    bioRelevantList = []
    for k in dictionary.keys():
        bioRelevantList.append(dictionary[k])
    
    return tuple(bioRelevantList)
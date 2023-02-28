# This script defines needed functions to train a model for cell type classification from a human tissue dataset. This script
# allows you to train CNN, wide model or WDL model. To run this script, you must give two input parameters from command line,
# as follows:
# python goWideDeepLearning.py --method method --model_type model_type
# where "method" indicates the objective of the model (multiclass classification, logistic regression) and "model_type" indicates
# the type of model you want to prepare for training. Optionally you can specify the CNN kernel size (deep model) with the
# "kernel" parameter.
# Examples:
# 1. WDL model for multiclass classification:
#       python goWideDeepLearning.py --method multiclass --kernel 3
# 2. WDL model for logistic regression:
#       python goWideDeepLearning.py
# 3. Deep model for multiclass classification:
#       python goWideDeepLearning.py --method multiclass --model_type deep
# 4. Deep model for multiclass classification with kernel size 3 (default value if not specified):
#       python goWideDeepLearning.py --method multiclass --model_type deep --kernel 3
#
#
# Before running this script, you need to make sure that the other files have been run in the following order:
# 1) "cellsFiltering.py"
# 2) "genesDictFiltering.py"
# 3) "relevantGenesExtraction.py"
# 4) "createKernelDataset.py"


# Import of utility libraries.
import numpy as np
import pandas as pd
import argparse
import sys
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense
from keras.layers import Input, concatenate
from keras.layers import Flatten, concatenate, Dropout, Conv1D, MaxPooling1D
from keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold
import operator
from util import extractBioRelevantGenes, printSummary, extractBioRelevantGenesPerCell, readDictionary, writeDataset
from keras import backend as K
import gc
import seaborn as sns
import matplotlib.pyplot as plt


# Help function to build crossed columns.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# x_cols : tuple
#       Crossed columns to convert.
#
# Returns
# ------------------------------------------------------------------------------------------------
# crossed_columns : dict
#       Crossed columns with dictionary structure.
#
def cross_columns(x_cols):
    crossed_columns = dict()
    colnames = ['_'.join(x_c) for x_c in x_cols]
    for cname, x_c in zip(colnames, x_cols):
        crossed_columns[cname] = x_c
    return crossed_columns


# Function for encoding an input vector in one hot vector format.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# x : numpy.ndarray
#       Training set or test set.
#
# Returns
# ------------------------------------------------------------------------------------------------
# np.array() : numpy.ndarray
#       Training set or test set encoded as one hot vector.
#
def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())


# This function removes cells without marker genes and therefore whitout relevant genes in input for the wide part.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Dataset containing cells-genes expression values.
# labels : pandas.DataFrame
#       Dataset containing cell types.
# filePath : string
#       File path of the dictionary.
#
# Returns
# ------------------------------------------------------------------------------------------------
# dataset : pandas.DataFrame
#       Filtered dataset containing cells-genes expression values.
# labels : pandas.DataFrame
#       Filtered dataset containing cell types.
#
def removeCellsWithoutMarkers(dataset, labels, filePath):
    # Dictionary with cell type as key and list of relevant genes concerning that cell type by value.
    cellsToBeRemoved = readDictionary(filePath)
    cellsToBeRemoved = set(cellsToBeRemoved.keys())
    cellsToBeRemoved = labels[~labels["Cell_type"].isin(cellsToBeRemoved)].index
    dataset.drop(cellsToBeRemoved, axis = 0, inplace = True)
    labels.drop(cellsToBeRemoved, axis = 0, inplace = True)

    return dataset, labels


# This function defines model for wide part, it is called when you want to design and train an entirely wide model.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# df_train : numpy.darray
#       Training set.
# df_test : numpy.darray
#       Test set.
# wide_cols : list of strings
#       List of biological relevant genes (columns/features), to train wide model.
# x_cols : tuple
#       List of columns to merge to generate crossed columns.
# target : string
#       Feature/Label to be predicted by model.
# model_type: string
#       Type of model.
# method : string
#       Objective of the model. Accepted values: "regression", "logistic", "multiclass".
#
# Returns
# ------------------------------------------------------------------------------------------------
# if model_type == "wide" :
#       returns:
#       1. cnf : pandas.DataFrame
#           Confusion matrix resulting from model validation.
#       2. scores : tuple
#           Evaluation metrics to evaluate the model.
# if model_type == "wide_deep" :
#       Returns X_train, y_train, X_test, y_test, i.e. the inputs required to build wide and deep model.
#
def wide(df_train, df_test, wide_cols, x_cols, target, model_type, method):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_wide = pd.concat([df_train, df_test])
    crossed_columns_d = cross_columns(x_cols)
    categorical_columns = list(df_wide.select_dtypes(include=['object']).columns)

    #wide_cols += list(crossed_columns_d.keys())

    #for k, v in crossed_columns_d.items():
       #df_wide[k] = df_wide[v].apply(lambda x: '-'.join(x.astype(str)), axis=1)  #ho aggiunto .astype(str) perche' mi dava problemi

    df_wide = df_wide[wide_cols + [target] + ['IS_TRAIN']]

    dummy_cols = [
        c for c in wide_cols if c in categorical_columns + list(crossed_columns_d.keys())]
    df_wide = pd.get_dummies(df_wide, columns=[x for x in dummy_cols])

    train = df_wide[df_wide.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_wide[df_wide.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)
    assert all(train.columns == test.columns)

    cols = [c for c in train.columns if c != target]
    X_train = train[cols].values
    y_train = train[target].values.reshape(-1, 1)
    X_test = test[cols].values
    y_test = test[target].values.reshape(-1, 1)

    if method == 'multiclass':
        y_train = onehot(y_train)
        y_test = onehot(y_test)

    if model_type == 'wide':

        activation, loss, metrics = fit_param[method]

        if metrics:
            metrics = [metrics]

        # Definition of wide model.
        wide_inp = Input(shape=(X_train.shape[1],), dtype = 'float32', name = 'wide_inp')
        w = Dense(y_train.shape[1], activation = activation)(wide_inp)
        wide = Model(wide_inp, w)

        # Model training and evaluation.
        wide.compile(loss=loss, metrics = metrics, optimizer = 'Adam')
        wide.fit(X_train, y_train, epochs = 30, batch_size = 64, verbose = 1)
        y_pred = wide.predict(X_test)

        # Results.
        results = wide.evaluate(X_test, y_test)
        cnf = confusion_matrix(y_test.argmax(axis = 1), y_pred.argmax(axis = 1))
        scores = precision_recall_fscore_support(y_test.argmax(axis = 1), y_pred.argmax(axis = 1), average='weighted')

        # Selection of the first 3 elements of scores (as the last is None) and appending accuracy.
        return cnf, scores[:-1] + (results[1], )

    else:

        return X_train, y_train, X_test, y_test


# This function defines model for deep part, it is called when you want to design and train a deep learning model.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# df_train : numpy.darray
#       Training set.
# df_test : numpy.darray
#       Test set.
# deep_cols : list of strings
#       List of columns to train deep model.
# target : string
#       Feature/Label to be predicted by model.
# model_type: string
#       Type of model.
# method : string
#       Objective of the model. Accepted values: "regression", "logistic", "multiclass".
# kernel : int
#       Kernel size for CNN model.
#
# Returns
# ------------------------------------------------------------------------------------------------
# if model_type == "deep" :
#       Ritorna:
#       1. cnf : pandas.DataFrame
#           Confusion matrix resulting from model validation.
#       2. scores : tuple
#           Evaluation metrics to evaluate the model.
# if model_type == "wide_deep" :
#       Riturns X_train, y_train, X_test, y_test, i.e. the inputs required to build wide and deep model.
#
def deep(df_train, df_test, deep_cols, target, model_type, method, kernel):
    df_train['IS_TRAIN'] = 1
    df_test['IS_TRAIN'] = 0
    df_deep = pd.concat([df_train, df_test])
    df_deep = df_deep[deep_cols + [target,'IS_TRAIN']]
    train = df_deep[df_deep.IS_TRAIN == 1].drop('IS_TRAIN', axis=1)
    test = df_deep[df_deep.IS_TRAIN == 0].drop('IS_TRAIN', axis=1)

    X_train = train.drop(target, axis=1)
    y_train = np.array(train[target].values).reshape(-1, 1)
    X_test = test.drop(target, axis=1)
    y_test = np.array(test[target].values).reshape(-1, 1)
    
    if method == 'multiclass':
        y_train = onehot(y_train)
        y_test = onehot(y_test)

    input = Input(shape = ( len(X_train.columns), 1))
    
    if model_type == 'deep':
        activation, loss, metrics = fit_param[method]

        if metrics:
            metrics = [metrics]
        
        # Definition of deep model.
        #inp = Input(shape = ( len(X_train.columns), ))
        #d = concatenate(inp_embed)
        #d = Flatten()(inp)
        dropout0 = Dropout(0.5) (input)
        conv1D = Conv1D(filters = 64, kernel_size = kernel, activation = 'relu', strides = kernel) (dropout0)
        #conv1D_2 = Conv1D(filters = 64, kernel_size = 3, activation = 'relu', strides = 3) (conv1D)
        #dropout1 = Dropout(0.25) (conv1D_2)
        dropout1 = Dropout(0.25) (conv1D)
        max_pooling1D = MaxPooling1D(pool_size = 2) (dropout1)
        flatten = Flatten() (max_pooling1D)
        dense = Dense(128, activation = 'relu') (flatten)
        dropout2 = Dropout(0.25) (dense)
        output = Dense(y_train.shape[1], activation = 'softmax') (dropout2)
        cnn = Model(inputs = input, outputs = output)

        # Model training and evaluation.
        cnn.compile(loss = loss, metrics = metrics, optimizer = 'Adam')
        cnn.fit(X_train, y_train, batch_size = 128, epochs = 30, verbose = 1)
        y_pred = cnn.predict(X_test)

        # Results.
        results = cnn.evaluate(X_test, y_test)
        cnf = confusion_matrix(y_test.argmax(axis = 1), y_pred.argmax(axis = 1))
        scores = precision_recall_fscore_support(y_test.argmax(axis = 1), y_pred.argmax(axis = 1), average='weighted')

        # Selection of the first 3 elements of scores (as the last is None) and appending accuracy.
        return cnf, scores[:-1] + (results[1], )

    else:

        return X_train, y_train, X_test, y_test, input


# This function defines WDL model, which combines deep part with wide part.
#
# Parameters
# ------------------------------------------------------------------------------------------------
# df_train : numpy.darray
#       Training set.
# df_test : numpy.darray
#       Test set.
# wide_cols : list of strings
#       List of biological relevant genes (columns/features), to train wide model.
# x_cols : tuple
#       List of columns to merge to generate crossed columns.
# deep_cols : list of strings
#       List of columns to train deep model.
# target : string
#       Feature/Label to be predicted by model.
# model_type: string
#       Type of model.
# method : string
#       Objective of the model. Accepted values: "regression", "logistic", "multiclass".
# kernel : int
#       Kernel size for CNN model (deep part).
#
# Returns
# ------------------------------------------------------------------------------------------------
# cnf : pandas.DataFrame
#       Confusion matrix resulting from model validation.
# scores : tuple
#       Evaluation metrics to evaluate the model.
#
def wide_deep(df_train, df_test, wide_cols, x_cols, deep_cols, target, model_type, method, kernel):

    X_train_wide, y_train_wide, X_test_wide, y_test_wide = \
            wide(df_train, df_test, wide_cols, x_cols, target, model_type, method)

    X_train_deep, y_train_deep, X_test_deep, y_test_deep, deep_inp = \
            deep(df_train, df_test, deep_cols, target, model_type, method, kernel)
    
    Y_tr_wd = y_train_deep
    Y_te_wd = y_test_deep

    activation, loss, metrics = fit_param[method]
    if metrics: metrics = [metrics]

    # Definition of model
    # WIDE
    wide_inp = Input(shape = (X_train_wide.shape[1],), dtype = 'float32', name = 'wide')


    # DEEP
    dropout0 = Dropout(0.25) (deep_inp)
    conv1D = Conv1D(filters = 64, kernel_size = kernel, activation = 'relu', strides = kernel) (dropout0)
    dropout1 = Dropout(0.75) (conv1D)
    max_pooling1D = MaxPooling1D(pool_size = 2) (dropout1)
    flatten = Flatten() (max_pooling1D)
    dense = Dense(128, activation = 'relu') (flatten)
    dropout2 = Dropout(0.5) (dense)
    wd_inp = concatenate([wide_inp, dropout2])
    dropout3 = Dropout(0.25) (wd_inp)
    wd_out = Dense(Y_tr_wd.shape[1], activation = activation, name = 'wide_deep') (dropout3)
    wide_deep = Model(inputs=[wide_inp,deep_inp], outputs = wd_out)

    # Model training and evaluation.
    wide_deep.compile(optimizer='Adam', loss = loss, metrics = metrics)
    wide_deep.fit([X_train_wide, X_train_deep], Y_tr_wd, epochs = 30, batch_size = 128, verbose = 1)
    y_pred = wide_deep.predict([X_test_wide, X_test_deep])

    # Results
    results = wide_deep.evaluate([X_test_wide, X_test_deep], Y_te_wd)
    cnf = confusion_matrix(y_test_deep.argmax(axis = 1), y_pred.argmax(axis = 1))
    scores = precision_recall_fscore_support(y_test_deep.argmax(axis = 1), y_pred.argmax(axis = 1), average='weighted')

    # Selection of the first 3 elements of scores (as the last is None) and appending accuracy.
    return cnf, scores[:-1] + (results[1], )



# Main function.
if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type = str, default = "logistic", help = "fitting method")
    ap.add_argument("--model_type", type = str, default = "wide_deep", help = "wide, deep or both")
    ap.add_argument("--train_data", type = str, default = "train.csv")
    ap.add_argument("--test_data", type = str, default = "test.csv")
    ap.add_argument("--kernel", type = str, default = "3", help = "kernel size")
    args = vars(ap.parse_args())
    method = args["method"]
    model_type = args['model_type']
    train_data = args['train_data']
    test_data = args['test_data']
    kernel = args['kernel']

    fit_param = dict()
    fit_param['logistic']   = ('sigmoid', 'binary_crossentropy', 'accuracy')
    fit_param['regression'] = (None, 'mse', None)
    fit_param['multiclass'] = ('softmax', 'categorical_crossentropy', 'accuracy')

    filePath = "data/"

    try:
        print("Reading gene expression values dataset...\n")
        dataset = pd.read_pickle(filePath + "Dataset_filtered4(kernel" + kernel + ").pkl")
        print("Reading cell types dataset...\n")
        labels = pd.read_pickle(filePath + "Labels_filtered.pkl")
        print("DATASET INFORMATION\n")
        printSummary(dataset, labels)
        kernel = int(kernel)

        # Two datasets into a single dataset concatenation.
        dataset_labels = pd.concat([dataset, labels], axis=1)

        # Extraction of biologically relevant genes that will be columns for wide part.
        wide_cols = extractBioRelevantGenes(filePath + "biologicalRelevantGenes.pkl")
        
        # Crossed columns.
        x_cols = extractBioRelevantGenesPerCell(filePath + "biologicalRelevantGenes.pkl")

        # Deep part columns.
        cont_cols = list(dataset_labels)
        cont_cols.remove('Cell_type')
        
        # Removing of columns with relevant genes in case of deep or wide and deep algorithm.
        if model_type != "svm" or model_type != "wide":
            cont_cols = [x for x in cont_cols if x not in set(wide_cols)]

        # Target label to predict.
        target = 'Cell_type'

        # Evaluation metrics.
        classesNames = sorted(labels["Cell_type"].unique())
        total_cnf = pd.DataFrame(data = 0, index = classesNames,  columns = classesNames)
        total_score = (0, 0, 0, 0)

        # 10 fold splitting.
        n_folds = 10
        skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 5)

        for train_index, test_index in skf.split(dataset, labels):
            df_train = dataset_labels.iloc[train_index]
            df_test = dataset_labels.iloc[test_index]
            df_train = df_train.copy()
            df_test = df_test.copy()

            if model_type == 'wide':
                cnf, scores = wide(df_train, df_test, wide_cols, x_cols, target, model_type, method)
            elif model_type == 'deep':
                cnf, scores = deep(df_train, df_test, cont_cols, target, model_type, method, kernel)
            else:
                cnf, scores = wide_deep(df_train, df_test, wide_cols, x_cols, cont_cols, target, model_type, method, kernel)
                
            cnf = pd.DataFrame(cnf, classesNames,  classesNames)
            print("\nConfusion matrix:")
            print(cnf)
            print("\nPrecision, Recall, FScore, Accuracy:", scores)
            total_score = tuple(map(operator.add, total_score, scores))
            total_cnf = total_cnf.add(cnf)
            print("-----------------------------------------------------------------------------------------------------------------------------")
            
            # Free GPU memory.
            K.clear_session()
            gc.collect()

        total_score = tuple(i/n_folds for i in total_score)
        print("\nTOTAL PRECISION, RECALL, FSCORE, ACCURACY:")
        print(total_score)
        print("\nTOTAL CONFUSION MATRIX:")
        print(total_cnf)
        writeDataset(total_cnf, "confusion_matrix_kernel" + str(kernel) +  ".pkl")

        # Writing confusion matrix to a file.
        #plt.figure(figsize = (20, 15))
        #sns.heatmap(cnf, annot = True, cmap = plt.cm.Blues, fmt='g')
        #plt.xlabel('Predicted cell types', fontsize = 30)
        #plt.ylabel('Actual cell types', fontsize = 30)
        #plt.savefig("cnf_kernel3.png", bbox_inches="tight")
        #plt.figure(figsize = (20, 15))
    

    # File not found exception.
    except(FileNotFoundError):
        print("Error: file not found.")
        sys.exit(1)
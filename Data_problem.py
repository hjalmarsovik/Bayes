#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Method to handle the data properly
def data_handle():
    #Read the given csv, no header
    df = pd.read_csv("data_problem2.csv", header=None)

    #Csv consists of first row: values, second row: labels
    #We want to re-format the csv to understand the data properly

    #Extract the two rows by iloc[] for the indexed rows
    values = df.iloc[0]
    labels = df.iloc[1]

    #Because our labels are in scientific notation
    #We convcert to float by astype(float)

    labels = labels.astype(float)

    #Create a new dataframe based on the re-formated data
    refreshed_df = pd.DataFrame({'value':values, 'label':labels})
    
    return refreshed_df

#Split the data
def datasplit(df, label_col, test_size=0.2, random_state=42):
    
    #Help from deepseek with syntax
    #np.random.seed
    np.random.seed(random_state)
    train_dfs = []
    test_dfs = []

    #For loop to split each label separately, ensuring same class dist.
    for label in df[label_col].unique():
        class_data = df[df[label_col] == label]
        n_test = int(len(class_data) * test_size)

        #Shuffle
        shuffled_ind = np.random.permutation(len(class_data))

        #80/20 split
        test_ind = shuffled_ind[:n_test]
        train_ind = shuffled_ind[n_test:]

        #Append to the dataframes
        train_dfs.append(class_data.iloc[train_ind])
        test_dfs.append(class_data.iloc[test_ind])

    #Finalize the dataframes by concat
    final_train_df = pd.concat(train_dfs, ignore_index=True)
    final_test_df = pd.concat(test_dfs, ignore_index=True)

    #Shuffle again
    final_train_df = final_train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    final_test_df = final_test_df.sample(frac=1, random_state=random_state).reset_index(drop=True )

    #Return the finalized dataframes
    return final_train_df, final_test_df

#Method for the estimated parameters, 2b
#For training data, and task states the gamma parameter alpha to be 2
def MLE_calculation(train_df, alpha=2.0):
    
    #Important to separate the two and then handle by corresponding dist.
    data_class0 = train_df[train_df['label'] == 0]['value'].values
    data_class1 = train_df[train_df['label'] == 1]['value'].values

    #Store the MLE parameters in a dict, which this function will ultimately return
    MLEs = {}

   

    #First find beta, class 0 follows gamma dist.
    #Length of data will equal how many n-times
    n_0 = len(data_class0)
    #If statement for the calculation itself
    if n_0 > 0:
        estimated_beta = np.sum(data_class0) / (n_0 * alpha)
        #Store in dict the estimated beta as beta
        MLEs['beta'] = estimated_beta 

        #Same for n_0
        MLEs['n_0'] = n_0

        #Handy to store mean value for data, using mean()
        MLEs['meanclass0'] = np.mean(data_class0)

    #Else statement for n_0 <= 0
    else:
        MLEs['beta']=None
        MLEs['n_0'] = 0


    #Same logic with if/else, but for class 1 following gaussian dist.
    n_1 = len(data_class1)
    if n_1 > 0:
        #Formulas in 2b, for finding the estimates
        estimated_mu = np.mean(data_class1)
        estimated_sigm_sq = np.sum((data_class1-estimated_mu) ** 2) / n_1

        #Same goes here, add to dict
        #Also add other handy parameters
        MLEs['mu'] = estimated_mu
        MLEs['sigma_squared'] = estimated_sigm_sq
        MLEs['sigma'] = np.sqrt(estimated_sigm_sq)
        MLEs['n_1'] = n_1
        MLEs['meanclass1'] = estimated_mu #Already calculated

    #Again make sure it does not return anything if we have no n-amounts
    else:
        MLEs['mu'] = None
        MLEs['sigma_squared'] = None
        MLEs['n_1'] = 0

    #Finally returns the dict, easy to remove parameters
    return MLEs
    
#Final method calling on both datasplit and MLE_calculation, proper parameters for both
def final_analysis(df, alpha=2.0, test_size=0.2, random_state=42):
    
    #Split the data with proper dataframes
    final_train_df, final_test_df = datasplit(df, 'label', test_size=test_size, random_state=random_state)

    #Task states to calculate MLE parameters for training data
    MLE_parameters = MLE_calculation(final_train_df, alpha)
    
    return {'train_df' : final_train_df, 'test_df' : final_test_df, 'MLE_parameters' : MLE_parameters}



if __name__ == "__main__":

    #Get values0 and values1 from data_handle()
    refreshed_df = data_handle()
    values0 = refreshed_df[refreshed_df['label'] == 0]['value']
    values1 = refreshed_df[refreshed_df['label'] == 1]['value']
    
    #Histogram based on the labels
    #Help from deepseek with syntax for hist plot
    plt.hist(values0, bins=30, alpha=0.7, label='Label 0', color='blue')
    plt.hist(values1, bins=30, alpha=0.7, label='Label 1', color='red')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Value dist. by label')
    plt.show()


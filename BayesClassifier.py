#Necessary libraries as well as Data_problem
import numpy as np
import math
import Data_problem 
import matplotlib.pyplot as plt

#Class for Bayes classification
class BayesClassifier:

    #Alpha known and equal to 2
    def __init__(self, alpha=2.0):
        self.alpha = alpha
        self.parameters = {}

    #Calculate probability density funct. for gamma
    #Partial help from Deepseek with function
    def PDF_gamma(self, x, alpha, beta):
        return (x**(alpha-1) * math.exp(-x/beta)) / (beta**alpha * math.gamma(alpha))
    
    #PDF, but this time for gaussian
    def PDF_gauss(self, x, mu, sigma):
        return (1/(sigma * math.sqrt(2*math.pi))) * math.exp(-0.5 * ((x-mu)/sigma)**2)


    #Fit method by calculating MLE from training data
    def fit(self, train_df):

        #Same code for datasplitting, separate the data belonging to each class
        data_class0 = train_df[train_df['label'] == 0]['value'].values
        data_class1 = train_df[train_df['label'] == 1]['value'].values

        #Calculate the previous probabilities, store in dict, used for Bayes
        ns = len(train_df)
        self.parameters['prior_0'] = len(data_class0) / ns
        self.parameters['prior_1'] = len(data_class1) / ns

        #Calculate MLE for gamma, same logic as seen in Data_problem.py
        if len(data_class0) > 0:
            self.parameters['beta'] = np.sum(data_class0) / (len(data_class0) * self.alpha)

        #Default value
        else:
            self.parameters['beta'] = 1.0 

        #Calculate MLE for gauss
        if len(data_class1) > 0:
            self.parameters['mu'] = np.mean(data_class1)

            #Calculate sigma squared, partial help from deepseek
            self.parameters['sigma_squared'] = np.sum((data_class1 - self.parameters['mu']) ** 2 )/ len(data_class1)
            self.parameters['sigma'] = np.sqrt(self.parameters['sigma_squared'])

        #Default values
        else: 
            self.parameters['mu'] = 0.0
            self.parameters['sigma'] = 1.0

        #Now fitted the model, returning the correct parameters
        return self.parameters
    
    #Predict class probabilities, by Bayes theorem
    #X will be input, similar to array 
    def pred_prob(self, X):

        #Help from deepseek
        #Arrays for probability of each sample data
        prob_0 = np.zeros(len(X)) #P(class=0|x)
        prob_1 = np.zeros(len(X)) #P(class=1|x)

        #For loop to calculate log likelihood for gauss and gamma, as well as post probabilites
        #Iterate through each single input sample of the data, hence enumerate(X)
        for i, x in enumerate(X):
            
            #Call on the corresponding calculation method, with updated parameters, again alpha fixed
            loglikelihood_0 = self.PDF_gamma(x, self.alpha, self.parameters['beta'])
            loglikelihood_1 = self.PDF_gauss(x, self.parameters['mu'], self.parameters['sigma'])

            #Calculate posterior probabilities by Bayes theorem
            #P(x) = P(x|class=0) * P(class=0) + P(x|class=1) * P(class=1), thats essentially whats going on here
            evidence = (loglikelihood_0 * self.parameters['prior_0'] + loglikelihood_1 * self.parameters['prior_1'])

            #If else statement for prob at [i]
            #Avoid zerodivision
            if evidence > 0:

                #Apply Bayes
                #P(class=0|x) = P(x|class=0) * P(class=0)/P(X)
                prob_0[i] = (loglikelihood_0 * self.parameters['prior_0']) / evidence
                
                #P(class=1|x) = P(x|class=1) * P(class=1)/P(X)
                prob_1[i] = (loglikelihood_1 * self.parameters['prior_1']) / evidence

            #Unlikely, but if evidence is zero p=0.5
            else:
                prob_0[i] = 0.5 
                prob_1[i] = 0.5

        #Help from deepseek
        #Returns the probabilities stacked, so each row contains [P(class=0|x), P(class=1|x)]
        return np.column_stack([prob_0, prob_1])
    
    #Method for prediction of class (0,1) based on input
    def predict(self, X):

        #Call on class prob method
        probabilities = self.pred_prob(X)

        #Return the class with highest prob, if tie choose class 1
        #Help from deepseek with syntax
        #np.where applies decision rule, to predict class 1 if P(class=1|x) >= P(class=0|x), and in the other cases predict class 0
        return np.where(probabilities[:, 1] >= probabilities[:, 0], 1, 0)
    

    #Method for evaluation of performance
    #Again, X input but also the true label this time
    def evaluate(self, X, y_true):

        #Call on prediction method for all data samples on given input
        y_pred = self.predict(X)

        #Calculate accuracy, how many of total predictions were true
        accuracy = np.mean(y_pred == y_true)

        #Returns overall accuracy, an array of the predicted labels as well as an array of the true labels
        return {'accuracy' : accuracy, 'predictions' : y_pred, 'true_labels' : y_true}
    
#Method for 2d, plotting the misclassified data
#Data, predictions as well as true labels as arguments
def misclassedplot(test_df, predictions, true_labels):

    #Handle the misclassified data, create mask for plot
    mask = predictions != true_labels
    data = test_df[mask]

    #Set figsize
    plt.figure(figsize=(12,6))

    #Help from deepseek with syntax
    #Plot the correctly classified 0s
    correct_class0 = test_df[(test_df['label'] == 0) & (~mask)]
    
    plt.scatter(correct_class0['value'], np.zeros(len(correct_class0)), color='blue', alpha=0.7, label='Correctly classified 0', marker='o')
    
    #Plot the correctly classified 1s
    correct_class1 = test_df[(test_df['label'] == 1) & (~mask)]
    
    plt.scatter(correct_class1['value'], np.zeros(len(correct_class1)), color='red', alpha=0.7, label='Correctly classified 1', marker='o')
    
    #Plot the misclassified data
    #Help from deepseek with syntax to make this data plotted more visibly
    plt.scatter(data['value'], np.zeros(len(data)), color = 'black', s=100, label='Misclassed', marker = 'x', linewidth=2)

    #Plot-specs
    plt.xlabel('Value')
    plt.title('Plot of classification results')
    plt.legend()
    plt.yticks([]) #Removes ticks on y-axis
    plt.show()






#Name main block, we want to report on the accuracy
#Use the Data_problem.py file to handle the data
if __name__ == "__main__":
    refreshed_df = Data_problem.data_handle()
    results = Data_problem.final_analysis(refreshed_df)
    classifier = BayesClassifier()
    classifier.fit(results['train_df'])

    #Store the results in a variable, for plot
    eval_results = classifier.evaluate(results['test_df']['value'].values, results['test_df']['label'].values)
    
    accuracy = classifier.evaluate(results['test_df']['value'].values, results['test_df']['label'].values)['accuracy']
    
    
    print(f"Accuracy on the training set: {accuracy:.4f}")
    misclassedplot(results['test_df'], eval_results['predictions'], results['test_df']['label'].values)
From Scratch Built Supervised Classification Machine Learning Library

Note: This is a side project and is not complete, It is still under work. I do add or imporve things whenever I get the time. If you find any bugs or issues or things that you think can be improved, do let me know.<br>
<br>
Constraints:<br>
For working with any of the machine learning algorithm inside the library, there are certain constraints for now, which are as follow:<br>
==> The Dataset being used must have output column at end<br>
==> Categorical Columns must not have been encoded to numerical<br><br>

Stuff Inside The Library:<br>
==> DataPreprocessing<br>
    Data Preprocessing will accept raw dataset and preform the following things:<br>
	i) Normalize Numerical Columns if the differnce between max and min value in that column is greater then 50.<br>
	ii) Perform One Hot Encoding on categorical columns.<br>
	iii) Take care of NaN/Missing values by replacing them with either Mean/Mode depending on whether the column is numerical or categorical.<br>
	iv) At last split dataset into Input(X) and Output(Y) and return them.<br><br>
    Note: Data Preprocessing part can be skipped if none of the above techniques needs to be applied on your data. Meaning if you are sure that your data is already in the format it would've been if passed to data preprocessing.<br><br>

After using the DataPrepocessing, you will have X(input) and Y(output).<br>
Now you can start using any of the machine learning algorithm inside the library, which include for now:<br>
1)==> Binary Logistic Regressor<br>
2)==> Naive Bayes Classifier(For Mix Data)<br>
3)==> Classification Tree<br>
4)==> Artificial Nueral Network(Multilayer Preceptron)<br>
5)==> K-Nearest Neighbour<br><br>

Almost all of these are only for Binary Classification, atleast for now. I'm working on expanding them to include multi-class too.<br><br>

Example code of how to Use these is given below:<br><br>

```python
from sklearn.model_selection import train_test_split
import pandas as pd

from triolearn.data_preprocessing import DataPrepocessing
from triolearn.machine_learning.MLP import NeuralNetwork
from triolearn.machine_learning.Probabillistic import NaiveBayes
from triolearn.machine_learning.Regression import Binary_Logistic_Regressor
from triolearn.machine_learning.NearestNeighbor import KNN

data = pd.read_csv('dataset.csv')
train_dataframe =  data.iloc[:int(len(data)*0.8),:]
test_dataframe =  data.iloc[int(len(data)*0.8):,:]

X, y = DataPrepocessing().dataCleaning(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=41)

NN_clf  = NeuralNetwork([64,32,16,8], epochs=10)
NB_clf  = NaiveBayes()
LR_clf  = Binary_Logistic_Regressor(
            main_gradient_descent = 'mini-batch',regularizer='l2', 
            hyper_parameters_assign='auto-assign',
            hyper_parameters_tuning_gradient_descent = 'mini-batch', lamda_value = None, lr_value = None, 
            max_iter = 1000, early_stopping = True, monitor='val_accuracy', paitiance = 3, 
            error_roundoff=8, acc_roundoff=4, acc_change=1e-3, error_change=1e-7, verbose=True)
KNN_clf = KNN()

print("Training...")
print('\nModel 1/4 ====> Nueral Network\n')
NN_clf.fit(X_train, y_train)
print('\nModel 2/4 ====> Naive Bayes\n')
#NB_clf.fit(train_dataframe)
print('\nModel 3/4 ====> Logistic Regression\n')
LR_clf.fit(X_train, y_train, X_test, y_test)
print('\nModel 4/4 ====> K-Nearest Neigbour\n')
KNN_clf.fit(X_train, y_train)

print("Evaluating..")
print('\nModel 1/4 ====> Nueral Network\n')
print("Neural Networks: ", NN_clf.score(X_test, y_test))
print('\nModel 2/4 ====> Naive Bayes\n')
print("Naive Bayes: ", NB_clf.score(test_dataframe.iloc[:,-1], test_dataframe))
print('\nModel 3/4 ====> Logistic Regression\n')
print("Logistic Regression: ", LR_clf.score)
print('\nModel 4/4 ====> K-Nearest Neigbour\n')
print("KNN: ", KNN_clf.score(X_test, y_test))


```
<br>
I know its not the best, but yeah I'm working on making it efficient, optimized and more generic. Please bear with me.<br>
Also any help on how to improve or if you want to contribute to making it better, do contact me using the contact option on my profile.<br>


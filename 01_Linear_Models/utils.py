from typing import Optional, Literal
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, precision_recall_fscore_support

class Dataset:
    '''
    A class used to import data for classification and regression practice.
    ...

    Attributes:
    -----------
    `classification`: bool, optional [default: True]
        a variable indicating requirement of classification data.
    `split_ratio`: float, optional [default: 0.2]
        a variable to indicate the split ratio for train and test datasets.
    `random_state`: int, optional [default 42]    
        a variable to define the random state from spliting dataset.
    
    Methods: {for use}
    --------
    `data()`:
        returns splitted diabetes data for regression and iris data for classification
    '''
    def __init__(self, classification:Optional[bool] = True, split_ratio:Optional[float] = 0.2, random_state: Optional[int] = 42):
        self.type = "Classification" if classification else "Regression"
        self.split_ratio = split_ratio
        self.random_state = random_state

    # Splits the dataset into training and test datesets.
    def _splitter(self, X, y):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.split_ratio, random_state=self.random_state)
        return (x_train, x_test, y_train, y_test)

    # Loads the Diabetes dataset and returns features and targets for train and test dataset.
    def _regression(self):
        dataset = load_diabetes()
        X = dataset.data
        y = dataset.target
        (x_train, x_test, y_train, y_test) = self._splitter(X, y)
        return (x_train, x_test, y_train, y_test)
    
    # Loads the Iris dataset and returns features and targets for train and test dataset.
    def _classification(self):
        dataset = load_iris()
        X = dataset.data
        y = dataset.target
        (x_train, x_test, y_train, y_test) = self._splitter(X, y)
        return (x_train, x_test, y_train, y_test)

    # Returns requested datasets.
    def data(self):
        if self.type=="Classification":
            return self._classification()
        else:
            return self._regression()

class Metrics:
    '''
    A class used to calculate the metric scores for model predictions.
    ...

    Attributes:
    -----------
    `y_pred`:
        model prections
    `y_true`:
        true values corresponding to the predicted values
    `classification`: bool, optional [default: True]
        a variable indicating classification mode
    
    Methods: {for use}
    --------
    `print_result()`:
        prints the metric scores for model predictions.
    `results()`:
        returns metric scores for model predictions.
    '''
    def __init__(self, y_pred, y_true, classification:Optional[bool] = True, average:Literal['binary', 'micro', 'macro', 'samples', 'weighted', None] = 'weighted'):
        self.type = "Classification" if classification else "Regression"
        self.y_pred = y_pred
        self.y_true = y_true
        self.average = average
    
    # Calculate prediction score for regression data.
    def _regression(self):
        r2 = r2_score(y_true=self.y_true, y_pred=self.y_pred)        
        return r2

    # Calculate prediction score for classification data.
    def _classification(self):
        acc = accuracy_score(y_true=self.y_true, y_pred=self.y_pred)
        prec, rec, f1, sup = precision_recall_fscore_support(y_true=self.y_true, y_pred=self.y_pred, average=self.average)
        result = (acc, prec, rec, f1)
        return result

    # Prints required mertic score for prediction.
    def print_result(self) -> None:
        if self.type=="Classification":
            (acc, prec, rec, f1) = self._classification()
            print(f"Accuracy: {acc} \nPrecision: {prec} \nRecall: {rec} \nF1-Score: {f1}")
        else:
            r2 = self._regression()
            print(f"R2 Score: {r2}")
    
    # Return requires metric scores for predictions.
    def results(self):
        '''
        Method to return metric scores for model predictions.
        
        ...
        Parameters:
        -----------
        None

        Returns:
        --------
        Accuracy [Classification]
        Precision [Classification]
        Recall [Classification]
        F1-Score [Classification]
        R2-Score [Regression]
        '''
        if self.type=="Classification":
            (acc, prec, rec, f1) = self._classification()
            return (acc, prec, rec, f1)
        else:
            r2 = self._regression()
            return r2
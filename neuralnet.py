from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#split data into training and testing sets
data = load_breast_cancer()
attributes = data.data
labels = data.target

attributes_train, attributes_test, labels_train, labels_test = train_test_split(attributes, labels, test_size=0.33)

neuralnet = MLPClassifier()
neuralnet.fit(attributes_train, labels_train)
accuracy = neuralnet.score(attributes_test, labels_test)


from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import re

passwords = np.genfromtxt('training.data', delimiter='|', comments='********', usecols = (0), dtype=str)
password_scores = np.genfromtxt('training.data', delimiter='|', comments='********', usecols = (1))

def extract_features(input):
	def f(password):
		password_length = len(password)
		has_special_characters = len(set('[~!@#$%^&*()_+{}":;\']+$').intersection(password)) > 0
		has_capitals = re.match(".*[A-Z].*", password) != None
		has_numeric_character = re.match(".*[0-9].*", password) != None

		return [
			password_length,
			has_special_characters,
			has_capitals,
			has_numeric_character
		]
	return map(f, input)

features = extract_features(passwords)

#The classifier
clf = tree.DecisionTreeClassifier()
#Training the classifier
clf.fit(features, password_scores)

test_passwords = np.genfromtxt('testing.data', delimiter='|', comments='********', usecols = (0), dtype=str)
test_password_scores = np.genfromtxt('testing.data', delimiter='|', comments='********', usecols = (1))

test_features = extract_features(test_passwords)

print clf.score(test_features, test_password_scores)

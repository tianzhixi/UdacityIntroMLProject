#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


### All the features that I can use, with the created new feature "from_to_poi".
#features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
#'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
#'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
#'long_term_incentive', 'restricted_stock', 'director_fees', 
#'to_messages', 'from_poi_to_this_person', 'from_messages', 
#'from_this_person_to_poi', 'shared_receipt_with_poi', "from_to_poi"]

### Selected features by feature importances.
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'bonus', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 
'long_term_incentive', 'restricted_stock', 'to_messages', 'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi']



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Data Exploration
#print len(data_dict)
#
#poi = 0
#for key, value in data_dict.iteritems() :
#    temp = data_dict[key]["poi"]
#    poi = poi + temp
#print poi    
#
#for key, value in data_dict.iteritems() :
#    print len(data_dict[key])


### Examples of features with many missing values
#count = 0
#for person in data_dict:
#    salary = data_dict[person]["salary"]
#    if salary == 'NaN':
#        count = count+1
#print count
#
#count = 0
#for person in data_dict:
#    total_payments = data_dict[person]["total_payments"]
#    if total_payments == 'NaN':
#        count = count+1
#print count
#
#count = 0
#for person in data_dict:
#    bonus = data_dict[person]["bonus"]
#    if bonus == 'NaN':
#        count = count+1
#print count



### Task 2: Remove outliers
    
### I plotted salary versus bonus and found an outlier, which has the key 'TOTAL'.
#import matplotlib
#for person in data_dict:
#    salary = data_dict[person]["salary"]
#    bonus = data_dict[person]["bonus"]
#    matplotlib.pyplot.scatter( salary, bonus )
#
#matplotlib.pyplot.xlabel("salary")
#matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()

#for key, value in data_dict.iteritems() :
#    print key
data_dict.pop('TOTAL', 0 ) # Removing 'TOTAL'



### Task 3: Create new feature(s)
### Creating a new feature which is the sum of from poi to this person and from this person to poi.
for person in data_dict:
    from_to_poi = data_dict[person]["from_poi_to_this_person"] + data_dict[person]["from_this_person_to_poi"]
    if from_to_poi == 'NaNNaN':
        from_to_poi = 'NaN'
    data_dict[person]['from_to_poi'] = from_to_poi

#for person in data_dict:
#    print data_dict[person]["from_to_poi"]


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

### I tried RandomForest and AdaBoost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

### I set random_state=42 to have reproducible result.
#rfc = RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, 
#                             min_samples_split=2, min_samples_leaf=1,
#                             max_features='auto', max_leaf_nodes=None, bootstrap=True, 
#                             oob_score=False, n_jobs=1, random_state=42, verbose=0)

### This line is for GridSearchCV
#rfc = RandomForestClassifier(random_state=42)

### My final model.
adb = AdaBoostClassifier(algorithm='SAMME', base_estimator=None, 
                         learning_rate=2, n_estimators=15, random_state=42)

### This line is for GridSearchCV
#adb = AdaBoostClassifier(random_state=42)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### GridSearchCV for AdaBoost
#from sklearn import grid_search
#parameters = {'algorithm': ('SAMME', 'SAMME.R'), "learning_rate": [1, 1.5, 2], "n_estimators": [10, 15, 20]}
#clf = grid_search.GridSearchCV(adb, parameters, scoring='f1')
#
#clf = clf.fit(features_train, labels_train)
#print clf.best_estimator_

### GridSearchCV did not give me a recall greater than 0.3. By manually tuning I found my final parameters.

### GridSearchCV for RandomForest
#parameters = {'criterion': ('gini', 'entropy'), "n_estimators": [10, 15, 20], "max_features": [5, 8, 11]}
#clf = grid_search.GridSearchCV(rfc, parameters, scoring='f1')
#
#clf = clf.fit(features_train, labels_train)
#print clf.best_estimator_

### Even by tuning the parameter, RandomForest is not as good as AdaBoost, especially for recall.


### The following codes are for accessing RandomForest at the early stage.
#rfc = rfc.fit(features_train, labels_train)
#pred = rfc.predict(features_test)
#
#from sklearn.metrics import accuracy_score
#print accuracy_score(labels_test, pred)
#from sklearn.metrics import confusion_matrix
#print confusion_matrix(labels_test, pred)


### The following codes are for feature importances.
#rfc = rfc.fit(features_train, labels_train)
#import numpy
#importances = rfc.feature_importances_
#print importances
#indices = numpy.argsort(importances)[::-1]
#for f in range(10):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


### Finally, I set my classifier as AdaBoost.
clf = adb



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
import json
import pandas as pd
import numpy as np
from sklearn.utils import column_or_1d
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def getData(trainfile, testfile, testfiley):
	#
	print "adasd"
	infile = open(trainfile, 'r')
	n = infile.readline()
	index = range(int(n))
	cols=["Physics", "Chemistry","Biology","English", "PhysicalEducation", "Accountancy", "BusinessStudies","Economics","ComputerScience"]
	ycols=["Mathematics"]
	x= pd.DataFrame(columns=cols)
	y= pd.DataFrame(columns=ycols)

	a = 0
	#with open('training-and-test/training.json', 'r') as f:
	for line in infile:
		js=json.loads(line)
		if "Physics" in js:
			x.loc[a,"Physics"] = js["Physics"]
		if "Chemistry" in js:
			x.loc[a,"Chemistry"] = js["Chemistry"]
		if "Biology" in js:
			x.loc[a,"Biology"] = js["Biology"]
		x.loc[a,"English"] = js["English"]
		
		if "PhysicalEducation" in js:
			x.loc[a,"PhysicalEducation"] = js["PhysicalEducation"]
		if "Accountancy" in js:
			x.loc[a,"Accountancy"] = js["Accountancy"]
		if "BusinessStudies" in js:
			x.loc[a,"BusinessStudies"] = js["BusinessStudies"]
		if "Economics" in js:
			x.loc[a,"Economics"] = js["Economics"]
		if "ComputerScience" in js:
			x.loc[a,"ComputerScience"] = js["ComputerScience"]

		y.loc[a,"Mathematics"] = js["Mathematics"]
		a=a+1

		#if a>10:
		#	break
	x=x.fillna(0)
	y=y.fillna(0)

	print "All data Loaded !!!" if n==a else  "Partial data loaded !!!"

	infile = open(testfile, 'r')
	n = infile.readline()
	index = range(int(n))
	cols=["Physics", "Chemistry","Biology","English", "PhysicalEducation", "Accountancy", "BusinessStudies","Economics","ComputerScience"]
	tx= pd.DataFrame(columns=cols)
	ty= pd.DataFrame(columns=ycols)
	
	a = 0
	for line in infile:
		js=json.loads(line)
		if "Physics" in js:
			tx.loc[a,"Physics"] = js["Physics"]
		if "Chemistry" in js:
			tx.loc[a,"Chemistry"] = js["Chemistry"]
		if "Biology" in js:
			tx.loc[a,"Biology"] = js["Biology"]
		tx.loc[a,"English"] = js["English"]
		
		if "PhysicalEducation" in js:
			tx.loc[a,"PhysicalEducation"] = js["PhysicalEducation"]
		if "Accountancy" in js:
			tx.loc[a,"Accountancy"] = js["Accountancy"]
		if "BusinessStudies" in js:
			tx.loc[a,"BusinessStudies"] = js["BusinessStudies"]
		if "Economics" in js:
			tx.loc[a,"Economics"] = js["Economics"]
		if "ComputerScience" in js:
			tx.loc[a,"ComputerScience"] = js["ComputerScience"]		
		a=a+1

		#if a>10:
		#	break
	tx=tx.fillna(0)

	ty = pd.read_csv(testfiley, names=ycols)
	ty

	return x,y,tx,ty,n

trainx, trainy, testx, testy, n= getData('training-and-test/training.json', 'training-and-test/sample-test.in.json', 'training-and-test/sample-test.out.json')

print len(trainx)
print len(trainy)
print len(testx)
print len(testy)


#79465
#js=json.loads('{"Physics":2,"Chemistry":2,"Biology":1,"English":1,"serial":221375}')
#js["Physics"]

#js["Physics"]

cols=["Physics", "Chemistry","Biology","English", "PhysicalEducation", "Accountancy", "BusinessStudies","Economics","ComputerScience"]
ycols=["Mathematics"]

'''
import numpy as np
X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
print(clf.predict(X[2:3]))

print "------------"
print X
print y
print X[2:3]
print "------------"
'''

import numpy as np
X = trainx #np.random.randint(5, size=(6, 100))
y = trainy #np.array([1, 2, 3, 4, 5, 6])
#y=np.reshape(y, len(y))
print "*********"
#y=np.ravel(y, order='C')
#y=y.flatten()
print "*********"
#y = column_or_1d(y, warn=True)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
pipeline = Pipeline([('classifier',  clf) ])

pipeline.fit(X[cols].values, y[ycols].values)
MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
print "********** Predict*************"
#print(clf.predict(testx))

p=pipeline.predict(testx[cols].values)
#p.to_csv('training-and-test/pred.csv')
print 'Accuracy Score : \t '+ str(accuracy_score(testy[ycols], p, normalize=False))
print '\a'


testx["pred"]=p
textx[i]["pred"] 
p()


testx.to_csv('training-and-test/pred.csv')
sum=0
for i in range(int(n)):
	if abs(textx[i]["pred"] - testy[0]) <=1:
		sum = sum+1


print sum
print 100.0*sum/n
print '\a'
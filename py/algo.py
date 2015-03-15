import sys, math
from sets import Set
from collections import Counter
import random
import numpy
from multiprocessing import Pool
import gc
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import accuracy_score

basePath = '/homes/pgurumur/dan/labelPrediction/data/'
school = "facebook"

# graph
edges = {}
label = {}
nodeAttributes = {}

#f_in = open(basePath + '../data/polblogs-nodes.txt')
f_in = open(basePath + '../data/' + school + '-nodes.txt')

# no need for first line...Skipping the header
junk_ = f_in.readline()

for line in f_in:
	fields = line.strip().split()
	label[int(float(fields[0]))] = int(float(fields[1]))
	edges[int(float(fields[0]))] = set([])

f_in.close()

f_in = open(basePath + '../data/' + school + '.attr')

for line in f_in:
	fields = line.strip().split("::")
	index = int(float(fields[0]))
	nodeAttributes[ index ] = []
	nodeAttributes[ index ].append( int(fields[1]) )
	nodeAttributes[ index ].append( int(fields[2]) )

f_in.close()



# Preprocess Wall posts

userPosting = {}
wallPosted = {}

for i in label:
	userPosting[ i ] = []
	wallPosted[ i ] = []

f = open(basePath + 'unique-wall-entries.txt')

for line in f:
	fields = line.strip().split('->')
	fields = fields[1].split('::')
	poster = int(float(fields[0]))
	postee = int(float(fields[1]))
	#print line
	text = fields[3]

	text = text.lower()
	#print text
	words = text.split(' ')
	
	if poster in label and postee in label:	
		for w in words:
			userPosting[ poster ].append( w )  
			wallPosted[ postee ].append( w )
	elif poster in label:
		for w in words:
			userPosting[ poster ].append( w )  
	elif postee in label:
		for w in words:
			wallPosted[ postee ].append( w )

f.close()


cu = 0
cw = 0
labelsToRemove = set()

for i in label:

	if len(userPosting[ i ]) == 0:
		#print "poster",i
		labelsToRemove.add( i )
		cu += 1
	if len(wallPosted[ i ]) == 0:
		#print "postee",i
		cw += 1
		labelsToRemove.add( i )
	 
print "cu",cu
print "cw",cw


# Remove those labels with no wall posting
print len(label)
for i in labelsToRemove:
	del label[i]
print len(label)



# Function to compute the Accuracy, Precision, Recall
# Input : test labels and the predicted labels
# Output : Accuracy, Precision, Recall
def computeAccuracy(testLabels,resultingLabels):
	counts = numpy.zeros([2,2])
	for i in range(len(testLabels)):
		counts[ testLabels[i], resultingLabels[i] ] += 1

	#print counts
	accuracy = (counts[0,0]+counts[1,1] + 0.0)/sum(sum(counts))

	precision = 0.0
	recall = 0.0
	if (counts[0,1]+counts[1,1]) != 0:
		precision = counts[1,1] /(counts[0,1]+counts[1,1])
	if (counts[1,0]+counts[1,1]) != 0:
		recall = counts[1,1]  / (counts[1,0]+counts[1,1])
	return accuracy,precision,recall



def constructFeatures(label,trainingLabels,corpus1,corpus2=None):

	wordFrequency = Counter()
	for user in corpus1:
		for w in corpus1[user]:
			wordFrequency[w] += 1

	#removeWords
	del wordFrequency['']


	topWords = {}
	ctr = 0
	for w in wordFrequency.most_common(15000):
		topWords[ w[0] ] = ctr
		ctr += 1

	N = len(topWords)
	features = {}
	for user in corpus1:
		f = [0]*N
		for w in corpus1[user]:
			if w in topWords:
				f[ topWords[w] ] = 1
		features[ user ] = f 

	print "features complete....."
	return features


# This function is used to combine two sets of features into one by appending one at the end of the other.
# Input : (Two feature to be combined) features1,features2
# Output : Combined Features (feature)
def combineFeatures(features1,features2):
	features = {}

	for i in features1:
		features[i] = features1[i] + features2[i]
		
	return features



def basicModel(originalLabels,trainingLabels,nodeAttributes):
	trainLabels = []
	trainFeatures = []
	testFeatures = []
	testLabels = []

	for i in originalLabels:
		if i in trainingLabels:
			trainLabels.append( originalLabels[i] )
			#print nodeAttributes[i]
			l = [1] + nodeAttributes[i]
			#print l
			trainFeatures.append( l )
		else:
			testLabels.append( originalLabels[i] )
			l = [1] + nodeAttributes[i]
			#print l
			testFeatures.append( l )

	"""
	logit = sm.Logit(trainLabels, trainFeatures)	 
	# fit the model
	result = logit.fit()
	#print result.summary()
	print result.params
	
	predicted = result.predict(testFeatures)
	resultingLabels = (predicted > threshold).astype(int)
	accuracy,precision,recall = computeAccuracy(testLabels,resultingLabels)
	print accuracy

	#return result.params,accuracy
	"""

	clf = linear_model.LogisticRegression()
	clf.fit(trainFeatures, trainLabels)
	pred = clf.predict(testFeatures)
	accuracy = accuracy_score(testLabels,pred)
	print accuracy
	print computeAccuracy(testLabels,pred)
	#print clf.get_params()
	#print clf.coef_[0]
	#print clf.intercept_

	return 0,accuracy
	




noofProcesses = 7
noOfTimeToRunGibbsSampling = 25


threshold = 0.5

arg1 = sys.argv[1]

trainingSizeList = [ float(arg1) ]

"""
#Read the testLabels from the files to make it constant across runs
testLabelsList = []
testSize = 1-trainingSizeList[0]
f = open(basePath + "RandomTestLabelsForIterations/" + str(testSize) + "_testLabels.txt")
tLL = f.readlines()
f.close

for line in tLL:
	t = line.strip().split(',')
	testLabelsList.append([int(i) for i in t])
"""

for trainingSize in trainingSizeList:

	print "\n\n\n\n\ntrainingSize:",trainingSize
				
	testSize = 1-trainingSize
	noOfLabelsToMask = int(testSize*len(label))
	print "testLabels Size:",noOfLabelsToMask

	for i in range(1):
		print "\nRepetition No.:",i+1

		# Uncomment the first line to generate random testLables for each iteration
		# Uncomment the second line to read the generated random testLables for each iteration. Based on Jen's suggestion to keep the testLabels constant across iterations.
		testLabels = random.sample(label,noOfLabelsToMask)
		#testLabels = testLabelsList[i]
		
		#print "Start test:",len(testLabels)
		trainingLabels = [i for i in label if i not in testLabels]
		#print "Start trainLabels:",len(trainingLabels)
		mleParameters,indepModel_accuracy = basicModel(label,trainingLabels,nodeAttributes)

		userPosting_features = constructFeatures(label,trainingLabels,userPosting)
		mleParameters,indepModel_accuracy = basicModel(label,trainingLabels,userPosting_features)

		wallPosted_features = constructFeatures(label,trainingLabels,wallPosted)
		mleParameters,indepModel_accuracy = basicModel(label,trainingLabels,wallPosted_features)		
		
		combinedFeatures = combineFeatures(userPosting_features,wallPosted_features)
		mleParameters,indepModel_accuracy = basicModel(label,trainingLabels,combinedFeatures)
		
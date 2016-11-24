import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt

# Set the global default size of matplotlib figures
plt.rc('figure', figsize=(10, 5))

# Size of matplotlib figures that contain subplots
fizsize_with_subplots = (10, 10)

# Size of matplotlib histogram bins
bin_size = 10

df_train = pd.read_csv('train.csv')

def showData():
	print "\n--------------------------TRAINING DATA HEAD--------------------------\n%s" % training_data.head();
	print "\n--------------------------TRAINING DATA TAIL--------------------------\n%s" % training_data.tail();
	print "\n--------------------------TRAINING DATA TYPE--------------------------\n%s" % training_data.dtypes;
	print "\n--------------------------TRAINING DATA INFO--------------------------\n%s" % training_data.info();
	print "\n--------------------------TRAINING DATA DESC--------------------------\n%s" % training_data.describe();


def showCharts():
	# Set up a grid of plots
	fig = plt.figure(figsize=fizsize_with_subplots) 
	fig_dims = (3, 2)

	# Plot death and survival counts
	plt.subplot2grid(fig_dims, (0, 0))
	df_train['Survived'].value_counts().plot(kind='bar', title='Death and Survival Counts')

	# Plot death and survival counts
	plt.subplot2grid(fig_dims, (0, 0))
	df_train['Survived'].value_counts().plot(kind='bar', 
	                                         title='Death and Survival Counts')

	# Plot Pclass counts
	plt.subplot2grid(fig_dims, (0, 1))
	df_train['Pclass'].value_counts().plot(kind='bar', 
	                                       title='Passenger Class Counts')

	# Plot Sex counts
	plt.subplot2grid(fig_dims, (1, 0))
	df_train['Sex'].value_counts().plot(kind='bar', 
	                                    title='Gender Counts')
	plt.xticks(rotation=0)

	# Plot Embarked counts
	plt.subplot2grid(fig_dims, (1, 1))
	df_train['Embarked'].value_counts().plot(kind='bar', 
	                                         title='Ports of Embarkation Counts')

	# Plot the Age histogram
	plt.subplot2grid(fig_dims, (2, 0))
	df_train['Age'].hist()
	plt.title('Age Histogram')
	plt.show()

def showStatsOnClassSurvival(data):
	pclass_xt = pd.crosstab(df_train[data], df_train['Survived'])
	print(pclass_xt)

	pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)

	pclass_xt_pct.plot(kind='bar', 
	                   stacked=True, 
	                   title='Survival Rate by Passenger Classes')
	plt.xlabel(data)
	plt.ylabel('Survival Rate')
	plt.show()

# showCharts()
showStatsOnClassSurvival("Age")






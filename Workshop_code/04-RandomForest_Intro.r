# Databricks notebook source
# MAGIC %md
# MAGIC # Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC - Popular ML algorithm
# MAGIC - Type of ensemble model (Boostrap Aggregation or bagging)
# MAGIC - Random Forest algorithm that makes a small tweak to Bagging and results in a very powerful classifie

# COMMAND ----------

# MAGIC %md
# MAGIC - Random forests are a modification of bagged decision trees that build a large collection of de-correlated trees to further improve predictive performance. 
# MAGIC - They have become a very popular “out-of-the-box” or “off-the-shelf” learning algorithm that enjoys good predictive performance with relatively little hyperparameter tuning. 
# MAGIC - Many modern implementations of random forests exist; however, Leo Breiman’s algorithm (Breiman 2001) has largely become the authoritative procedure. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ![RF](https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC - RF improve  performance of a single decision tree by taking the average of many trees. ==> an ensemble method, or model averaging approach.
# MAGIC - Why Random?   
# MAGIC  - each tree is a bagged sample and a subset of all predictors are used as candidates at each split (not all predictors like single decision trees)
# MAGIC   - Reduced number of predictor candidates at each split allows for something other than the best split to be the top split, thus growing many different looking trees - this "decorrelates" the trees.
# MAGIC - Why Forests?   
# MAGIC  - Forests because many trees are grown!

# COMMAND ----------

# MAGIC %md
# MAGIC ### What is bootstrapping?
# MAGIC - bootstrap is a powerful statistical method for estimating a quantity from a data sample
# MAGIC - know that our sample is small and that our mean has error in it
# MAGIC - How to bootstrap?
# MAGIC  - Create many (e.g. 1000) random sub-samples of our dataset with replacement (meaning we can select the same value multiple times).
# MAGIC  - Calculate the mean of each sub-sample.
# MAGIC  - Calculate the average of all of our collected means and use that as our estimated mean for the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bootstrap Aggregation (Bagging)?
# MAGIC - simple and very powerful ensemble method.
# MAGIC - ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model
# MAGIC - Bagging is the application of the Bootstrap procedure to a high-variance machine learning algorithm
# MAGIC - How bagging works?
# MAGIC  - Create many (e.g. 100) random sub-samples of our dataset with replacement.
# MAGIC  - Train a CART model on each sample.
# MAGIC  - Given a new dataset, calculate the average prediction from each model.

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest
# MAGIC - Random Forests are an improvement over bagged decision trees.
# MAGIC 
# MAGIC - Random forest changes the algorithm for the way that the sub-trees are learned so that the resulting predictions from all of the subtrees have less correlation.
# MAGIC 
# MAGIC - It is a simple tweak. In CART, when selecting a split point, the learning algorithm is allowed to look through all variables and all variable values in order to select the most optimal split-point. The random forest algorithm changes this procedure so that the learning algorithm is limited to a random sample of features of which to search.
# MAGIC 
# MAGIC - The number of features that can be searched at each split point (m) must be specified as a parameter to the algorithm. You can try different values and tune it using cross validation.
# MAGIC 
# MAGIC - For classification a good default is: m = sqrt(p)
# MAGIC - For regression a good default is: m = p/3
# MAGIC 
# MAGIC - Where m is the number of randomly selected features that can be searched at a split point and p is the number of input variables. For example, if a dataset had 25 input variables for a classification problem, then:
# MAGIC 
# MAGIC  - m = sqrt(25)
# MAGIC   - m = 5
# MAGIC 
# MAGIC - Estimated Performance
# MAGIC  - For each bootstrap sample taken from the training data, there will be samples left behind that were not included. These samples are called Out-Of-Bag samples or OOB.
# MAGIC 
# MAGIC  - The performance of each model on its left out samples when averaged can provide an estimated accuracy of the bagged models. This estimated performance is often called the OOB estimate of performance.
# MAGIC 
# MAGIC  - These performance measures are reliable test error estimate and correlate well with cross validation estimates.
# MAGIC 
# MAGIC - Variable Importance
# MAGIC  - As the Bagged decision trees are constructed, we can calculate how much the error function drops for a variable at each split point.
# MAGIC 
# MAGIC  - In regression problems this may be the drop in sum squared error and in classification this might be the Gini score.
# MAGIC 
# MAGIC  -These drops in error can be averaged across all decision trees and output to provide an estimate of the importance of each input variable. The greater the drop when the variable was chosen, the greater the importance.
# MAGIC 
# MAGIC  - These outputs can help identify subsets of input variables that may be most or least relevant to the problem and suggest at possible feature selection experiments you could perform where some features are removed from the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC #### The basic algorithm for a regression or classification random forest can be generalized as follows:
# MAGIC 
# MAGIC 1.  Given a training data set
# MAGIC 2.  Select number of trees to build (n_trees)
# MAGIC 3.  for i = 1 to n_trees do
# MAGIC 4.  |  Generate a bootstrap sample of the original data
# MAGIC 5.  |  Grow a regression/classification tree to the bootstrapped data
# MAGIC 6.  |  for each split do
# MAGIC 7.  |  | Select m_try variables at random from all p variables
# MAGIC 8.  |  | Pick the best variable/split-point among the m_try
# MAGIC 9.  |  | Split the node into two child nodes
# MAGIC 10. |  end
# MAGIC 11. | Use typical tree model stopping criteria to determine when a 
# MAGIC     | tree is complete (but do not prune)
# MAGIC 12. end
# MAGIC 13. Output ensemble of trees 

# COMMAND ----------

library(randomForest)
library(caret)
library(e1071)
library(tidyverse)

# COMMAND ----------

# Set random seed to make results reproducible:
set.seed(17)
# Calculate the size of each of the data sets:
data_set_size <- floor(nrow(iris)/2)
# Generate a random sample of "data_set_size" indexes
indexes <- sample(1:nrow(iris), size = data_set_size)
# Assign the data to the correct sets
training <- iris[indexes,]
validation1 <- iris[-indexes,]

# COMMAND ----------

# Perform training:
rf_classifier = randomForest(Species ~ ., data=training, ntree=100, mtry=2, importance=TRUE)

# COMMAND ----------

 rf_classifier

# COMMAND ----------

varImpPlot(rf_classifier)

# COMMAND ----------

prediction_for_table <- predict(rf_classifier,validation1[,-5])
table(observed=validation1[,5],predicted=prediction_for_table)

# COMMAND ----------

?confusionMatrix

# COMMAND ----------

confusionMatrix(prediction_for_table, validation1[,5])

# COMMAND ----------

# MAGIC %md
# MAGIC #let's try another example here
# MAGIC ## PimaIndiansDiabetes2

# COMMAND ----------

# Load the data and remove NAs
data("PimaIndiansDiabetes2", package = "mlbench")
PimaIndiansDiabetes2 <- na.omit(PimaIndiansDiabetes2)
# Inspect the data
head(PimaIndiansDiabetes2, 3)

# COMMAND ----------

# Split the data into training and test set
set.seed(123)
training.samples <- PimaIndiansDiabetes2$diabetes %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- PimaIndiansDiabetes2[training.samples, ]
test.data <- PimaIndiansDiabetes2[-training.samples, ]

# COMMAND ----------

?createDataPartition

# COMMAND ----------

# MAGIC %md
# MAGIC #Try on your own

# COMMAND ----------

# Build the model
set.seed(123)
rf_classifier2 = randomForest(diabetes ~ ., data=train.data, ntree=100, mtry=2, importance=TRUE)

# COMMAND ----------

# Make predictions on the test data
predicted.classes <- rf_classifier2 %>% 
  predict(test.data, type = "class")
head(predicted.classes)

# COMMAND ----------

confusionMatrix(predicted.classes, test.data$diabetes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take aways
# MAGIC - RF provide very powerful out-of-the-box algorithm w/ great predictive accuracy. 
# MAGIC - All the benefits of decision trees (with the exception of surrogate splits) and bagging but greatly reduce instability and between-tree correlation. 
# MAGIC - And due to the added split variable selection attribute, RF are also faster than bagging as they have a smaller feature search space at each tree split. 
# MAGIC 
# MAGIC ## Drawbacks
# MAGIC - RF suffer from slow computational speed as your data sets get larger but, similar to bagging, the algorithm is built upon independent steps.
# MAGIC  - Modern implementations (e.g., ranger, h2o) allow for parallelization.

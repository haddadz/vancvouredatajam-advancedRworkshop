# Databricks notebook source
# MAGIC %md
# MAGIC # Decision Trees
# MAGIC ### Not focused on best way to perform modelling using Decision Trees or Random Forest
# MAGIC #### Instead focus on intro and getting you started and using Decision Trees & Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC ## Who has used modelling in the past?
# MAGIC ### Slack Poll

# COMMAND ----------

library("caret")
library("tidyverse")
data(diamonds)
model <- lm(price ~ ., diamonds)
p <- predict(model, diamonds)

# COMMAND ----------

head(diamonds)

# COMMAND ----------

summary(model)

# COMMAND ----------

summary(p)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Trees
# MAGIC * Tree-based models 
# MAGIC  - Non-parametric algorithms 
# MAGIC  - Partitioning the feature space into a number of smaller (non-overlapping) regions with similar response values using a set of splitting rules. 
# MAGIC * Predictions are obtained by fitting a simpler model (e.g., a constant like the average response value) in each region. 
# MAGIC * Such divide-and-conquer methods can produce simple rules that are easy to interpret and visualize with tree diagrams. 

# COMMAND ----------

# MAGIC %md
# MAGIC - One methodology for constructing Decision Trees is CART the classification and regression tree (CART) algorithm
# MAGIC   - Partition training data into homogeneous subgroups (nodes)
# MAGIC and then fits a simple constant in each node
# MAGIC - Nodes are formed recursively using binary partitions formed by asking simple yes-or-no questions about each feature (e.g., is age < 18?). 
# MAGIC  - This is done a number of times until a suitable stopping criteria is satisfied (e.g., a maximum depth of the tree is reached). 
# MAGIC - Then the model predicts the output based on 
# MAGIC  - (1) the average response values for all observations that fall in that subgroup (regression problem), or 
# MAGIC  - (2) the class that has majority representation (classification problem). 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Example of Decision Tree
# MAGIC 
# MAGIC ![DCT1.jpg](https://cdn-images-1.medium.com/max/824/0*J2l5dvJ2jqRwGDfG.png)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ![DCT2.jpg](https://cdn-images-1.medium.com/max/688/0*pb-1ufHK-OmR8k7r.png)

# COMMAND ----------

library(rpart)
model <- rpart(Species ~., data = iris)
par(xpd = NA) # otherwise on some devices the text is clipped
plot(model)
text(model, digits = 3)

# COMMAND ----------

dim(iris)

# COMMAND ----------

print(model, digits = 2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Trees: Partitioning
# MAGIC 
# MAGIC - CART uses binary recursive partitioning (it’s recursive because each split or rule depends on the splits above it). 
# MAGIC - The objective at each node is to find the “best” feature to partition the remaining data into one of two regions (R1 and R2) such that the overall error between the actual response (yi) and the predicted constant (ci) is minimized.
# MAGIC - Having found the best feature/split combination, the data are partitioned into two regions and the splitting process is repeated on each of the two regions (hence the name binary recursive partitioning). 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Decision Trees: Deep
# MAGIC - Early stopping
# MAGIC   - Restrict the tree depth to a certain level or 
# MAGIC  - Restrict the mini number of observations allowed in any terminal node
# MAGIC - Pruning
# MAGIC   - grow a very large, complex tree and then prune it back to find an optimal subtree

# COMMAND ----------

# Load the data and remove NAs
data("PimaIndiansDiabetes2", package = "mlbench")
PimaIndiansDiabetes2 <- na.omit(PimaIndiansDiabetes2)
# Inspect the data
sample_n(PimaIndiansDiabetes2, 3)
#https://rdrr.io/cran/mlbench/man/PimaIndiansDiabetes.html

# COMMAND ----------

# Split the data into training and test set
set.seed(123)
training.samples <- PimaIndiansDiabetes2$diabetes %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- PimaIndiansDiabetes2[training.samples, ]
test.data <- PimaIndiansDiabetes2[-training.samples, ]

# COMMAND ----------

# Build the model
set.seed(123)
model1 <- rpart(diabetes ~., data = train.data, method = "class")
# Plot the trees
par(xpd = NA) # Avoid clipping the text in some device
plot(model1)
text(model1, digits = 3)

# COMMAND ----------

# Make predictions on the test data
predicted.classes <- model1 %>% 
  predict(test.data, type = "class")
head(predicted.classes)

# COMMAND ----------

# Compute model accuracy rate on test data
mean(predicted.classes == test.data$diabetes)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pruning the tree

# COMMAND ----------

?train

# COMMAND ----------

# Fit the model on the training set
set.seed(123)
model2 <- train(
  diabetes ~., data = train.data, method = "rpart",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
  )
# Plot model accuracy vs different values of
# cp (complexity parameter)
plot(model2)

# COMMAND ----------

# Print the best tuning parameter cp that
# maximizes the model accuracy
model2$bestTune

# COMMAND ----------

# Plot the final tree model
par(xpd = NA) # Avoid clipping the text in some device
plot(model2$finalModel)
text(model2$finalModel,  digits = 3)

# COMMAND ----------

# Decision rules in the model
model2$finalModel

# COMMAND ----------

# Make predictions on the test data
predicted.classes <- model2 %>% predict(test.data)
# Compute model accuracy rate on test data
mean(predicted.classes == test.data$diabetes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take aways: 
# MAGIC - Simple and easy to implement (no preprocessing required and categorical variables handled, handle missing values)
# MAGIC - Not great on predictive accuracy
# MAGIC - Deep trees have high variance / Shallow Trees bias.
# MAGIC 
# MAGIC ## More technical: 
# MAGIC - A problem with decision trees like CART is that they are greedy. 
# MAGIC - Even with Bagging, the decision trees can have a lot of structural similarities and in turn have high correlation in their predictions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Another way to think about Decision Trees
# MAGIC - Great advantage of decision trees: make a complex decision simpler by breaking it down into smaller, simpler decisions using a divide-and-conquer strategy. 
# MAGIC - Identify a set of if-else conditions that split the data according to the value of the features.

# COMMAND ----------

# MAGIC %md
# MAGIC # Before jumping to the next section, let's try out few examples

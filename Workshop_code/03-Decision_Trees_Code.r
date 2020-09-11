# Databricks notebook source
# install.packages(c("mlbench","rpart.plot", "e1071"))

# COMMAND ----------

library("caret")
library("e1071")
library("rpart")
library("tidyverse")

# COMMAND ----------

# Load the data
data("Boston", package = "MASS")
head(Boston)
#https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's look into the Boston Dataset
# MAGIC ## Build a Decision Tree to predict Median Value

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is the first step?

# COMMAND ----------

# Inspect the data
sample_n(Boston, 3)
# Split the data into training and test set
set.seed(123)
training.samples <- Boston$medv %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- Boston[training.samples, ]
test.data <- Boston[-training.samples, ]

# COMMAND ----------

# Fit the model on the training set
set.seed(123)
model <- train(
  medv ~., data = train.data, method = "rpart",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
  )
# Plot model error vs different values of
# cp (complexity parameter)
plot(model)


# COMMAND ----------

# Print the best tuning parameter cp that
# minimize the model RMSE
model$bestTune

# COMMAND ----------

# Plot the final tree model
par(xpd = NA) # Avoid clipping the text in some device
plot(model$finalModel)
text(model$finalModel, digits = 3)

# COMMAND ----------

# Decision rules in the model
model$finalModel
# Make predictions on the test data
predictions <- model %>% predict(test.data)
head(predictions)


# COMMAND ----------

summary(test.data$medv)

# COMMAND ----------

summary(predictions)

# COMMAND ----------

# Compute the prediction error RMSE
RMSE(predictions, test.data$medv)

# COMMAND ----------

# plot(predictions, test.data$medv)
 plot(predictions,test.data$medv,
      xlab="predicted",ylab="actual")
 abline(a=0,b=1)

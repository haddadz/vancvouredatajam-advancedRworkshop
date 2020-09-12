# Databricks notebook source
# MAGIC %md 
# MAGIC #Try it out on your own
# MAGIC ## Build a Decision Tree and Random Forest - Compare both models

# COMMAND ----------

# MAGIC %r
# MAGIC # install.packages("e1071")

# COMMAND ----------

#packages

library(randomForest)
library(rpart)
library(caret)
# library(e1071)

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC #load my data
# MAGIC # Data Source: https://archive.ics.uci.edu/ml/machine-learning-databases/car/
# MAGIC 
# MAGIC # Load the dataset and explore
# MAGIC data1 <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header = TRUE)
# MAGIC  

# COMMAND ----------

display(head(data1))

#display(class(data1))

# COMMAND ----------

class(data1)
colnames(data1) <- c("BuyingPrice", "Maintenance","NumDoors","NumPersons", "BootSpace","Safety","Condition")


# COMMAND ----------

head(data1)
 


# COMMAND ----------

str(data1)

# COMMAND ----------

summary(data1)

# COMMAND ----------

# head(data1)
# str(data1)

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
set.seed(100)
train <- sample(nrow(data1), 0.7*nrow(data1), replace = FALSE)
TrainSet <- data1[train,]
ValidSet <- data1[-train,]
summary(TrainSet)


# COMMAND ----------

summary(ValidSet)

# COMMAND ----------

# Create a Random Forest model with default parameters
model1 <- randomForest(Condition ~ ., data = TrainSet, importance = TRUE)
model1

# COMMAND ----------

# Fine tuning parameters of Random Forest model
model2 <- randomForest(Condition ~ ., data = TrainSet, ntree = 500, mtry = 6, importance = TRUE)
model2


# COMMAND ----------

# Predicting on train set
predTrain <- predict(model2, TrainSet, type = "class")
# Checking classification accuracy
table(predTrain, TrainSet$Condition)  

# COMMAND ----------

# Predicting on Validation set
predValid <- predict(model2, ValidSet, type = "class")
# Checking classification accuracy
mean(predValid == ValidSet$Condition)                    


# COMMAND ----------

table(predValid,ValidSet$Condition)

# COMMAND ----------

importance(model2) 

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC # To check important variables
# MAGIC # importance(model2)        
# MAGIC varImpPlot(model2)  

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC # Using For loop to identify the right mtry for model
# MAGIC a=c()
# MAGIC i=5
# MAGIC for (i in 3:8) {
# MAGIC   model3 <- randomForest(Condition ~ ., data = TrainSet, ntree = 500, mtry = i, importance = TRUE)
# MAGIC   predValid <- predict(model3, ValidSet, type = "class")
# MAGIC   a[i-2] = mean(predValid == ValidSet$Condition)
# MAGIC }
# MAGIC  
# MAGIC a
# MAGIC  
# MAGIC plot(3:8,a)

# COMMAND ----------

summary(TrainSet)

# COMMAND ----------

library(rpart)
library(caret)
library(e1071)

# COMMAND ----------

# We will compare model 1 of Random Forest with Decision Tree model
 
model_dt = train(Condition ~ ., data = TrainSet, method = "rpart")
# model_dt =rpart(Condition ~ ., data = TrainSet,method = "class")
model_dt_1 = predict(model_dt, data = TrainSet)

class(model_dt)


# COMMAND ----------

table(model_dt_1, TrainSet$Condition)

# COMMAND ----------

mean(model_dt_1 == TrainSet$Condition)

# COMMAND ----------

# MAGIC %r
# MAGIC # Running on Validation Set
# MAGIC model_dt_vs = predict(model_dt, newdata = ValidSet)
# MAGIC table(model_dt_vs, ValidSet$Condition)
# MAGIC  

# COMMAND ----------

mean(model_dt_vs == ValidSet$Condition)

# COMMAND ----------

confusionMatrix(model_dt_vs, ValidSet$Condition)

# COMMAND ----------

library("rpart.plot")
rpart.plot(model_dt)

# COMMAND ----------

plot(model_dt$finalModel)
text(model_dt$finalModel)

# COMMAND ----------



# Databricks notebook source
# MAGIC %md 
# MAGIC #Try it out on your own
# MAGIC ## Build a Decision Tree and Random Forest - Compare both models

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Dataset
# MAGIC - Car Evaluation data set from Data Source: https://archive.ics.uci.edu/ml/machine-learning-databases/car/
# MAGIC 
# MAGIC ## Problem
# MAGIC - Predict Condition of Cars based on features provided
# MAGIC  - Let's try both Decision Trees and Random Forest and compare their performance
# MAGIC 
# MAGIC ## Questions
# MAGIC 1- What is the first step?  
# MAGIC 2- What's after that?   

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Tips
# MAGIC 1- Load the packages
# MAGIC 2- Load the data
# MAGIC 3- Split the data

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

# MAGIC %md
# MAGIC 
# MAGIC ## Try on your own

# COMMAND ----------





# Set working directory
setwd(dirname(file.choose()))
getwd()

# Install packages
library(keras)
#install_keras() #no need here

# Used once here at the beginning
#conda_create("r-reticulate")
# Use here (in my Asus with proper python installed -20200720). Important here
use_condaenv("r-reticulate")

# Read data
data <- read.csv("Cardiotocographic.csv")
str(data)

# Change to matrix
data <- as.matrix(data)
dimnames(data) <- NULL

# Normalize
data[, 1:21] <- normalize(data[, 1:21])
data[,22] <- as.numeric(data[,22]) -1
summary(data)

# Data partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
training <- data[ind==1, 1:21]
test <- data[ind==2, 1:21]
trainingtarget <- data[ind==1, 22]
testtarget <- data[ind==2, 22]

# One Hot Encoding
trainLabels <- to_categorical(trainingtarget)
testLabels <- to_categorical(testtarget)
print(testLabels)

# Create sequential model
## 1st: one hidden layer, units = 8 (21 input columns, 3 categories)
model <- keras_model_sequential()
model %>% #one hidden layer, units = 8 (21 input columns, 3 categories)
         layer_dense(units=8, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units = 3, activation = 'softmax')
summary(model)

## 2nd: one hidden layer, units = 21
model <- keras_model_sequential()
model %>%
         layer_dense(units=50, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units = 3, activation = 'softmax')
		 
## 3rd: two hidden layers, 1st units = 21, 2nd units = 8
model <- keras_model_sequential()
model %>%
         layer_dense(units=21, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units=8, activation = 'relu') %>%
		 layer_dense(units = 3, activation = 'softmax')	

## try different numbers of layer and unit
model <- keras_model_sequential()
model %>%
         layer_dense(units=84, activation = 'relu', input_shape = c(21)) %>%
         layer_dense(units=42, activation = 'relu') %>%
		 layer_dense(units=21, activation = 'relu') %>%
		 layer_dense(units=7, activation = 'relu') %>%
		 layer_dense(units = 3, activation = 'softmax') 

# Compile
model %>%
         compile(loss = 'categorical_crossentropy',
                 optimizer = 'adam',
                 metrics = 'accuracy')

# Fit model
## 1st:
history <- model %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 32,
             validation_split = 0.2)
plot(history)

## try different numbers of epoch, batch_size and validation_split:
history <- model %>%
         fit(training,
             trainLabels,
             epoch = 200,
             batch_size = 64,
             validation_split = 0.25)
plot(history)

# Evaluate model with test data
model1 <- model %>%
         evaluate(test, testLabels)

# Prediction & confusion matrix - test data
prob <- model %>%
         predict_proba(test)

pred <- model %>%
         predict_classes(test)

table <- table(Predicted = pred, Actual = testtarget)
table

library(caret)
confusionMatrix(table <- table(Predicted = pred, Actual = testtarget), mode = "everything")

cbind(prob, pred, testtarget)

# Fine-tune model

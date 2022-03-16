library(data.table)
library(ggplot2)
library(lattice)
library(caret)
library(mlbench)
set.seed(1234)

#Import Data set
data ("PimaIndiansDiabetes")
str(PimaIndiansDiabetes)

summary(PimaIndiansDiabetes)

summary(is.na(PimaIndiansDiabetes))

#Detect and remove redundant features(variables with considerable multicollinearity)
correlation = cor(PimaIndiansDiabetes[, -9])
correlation
highcorrelation = findCorrelation(correlation, cutoff = 0.50)
highcorrelation

highcorrelation = findCorrelation(correlation, cutoff = 0.40)
highcorrelation

highcorrelation = findCorrelation(correlation, cutoff = 0.30)
highcorrelation

#Check for error
RFE = rfe(PimaIndiansDiabetes[, 1:8], 
          PimaIndiansDiabetes[, 9], 
          sizes = c(1:8),
          rfeControl = rfeControl(functions = rfFuncs, method = "cv", number = 10))

RFE
ggplot(RFE)

#prepare with 4 consider the accuracy in ggplot
data = PimaIndiansDiabetes[, c("glucose", "mass", "age", "pregnant", "diabetes")]


#Split into training and test data set
training = createDataPartition(y = data$diabetes, p = 0.75, list = FALSE)
train_set = data[training, ]
test_set = data[-training, ]


#Training the model using SVM Linear
model_fitting_svm = train(data = train_set, diabetes~., method = "svmLinear",
                          trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5),
                          preProcess = c("center", "scale"))
model_fitting_svm
confusionMatrix(predict(model_fitting_svm, newdata = train_set), train_set$diabetes)
confusionMatrix(predict(model_fitting_svm, newdata = test_set), test_set$diabetes)

#Training the model using SVM Radial
model_fitting_svm_RBF = train(data = train_set, diabetes~., method = "svmRadial",
                          trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5),
                          preProcess = c("center", "scale"))
model_fitting_svm_RBF
confusionMatrix(predict(model_fitting_svm_RBF, newdata = train_set), train_set$diabetes)
confusionMatrix(predict(model_fitting_svm_RBF, newdata = test_set), test_set$diabetes)

#SVM Radial provides better accuracy than SVM Linear.

#Training the model using SVM Polynomial
model_fitting_svm_Poly = train(data = train_set, diabetes~., method = "svmPoly",
                              trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5),
                              preProcess = c("center", "scale"))
model_fitting_svm_Poly
confusionMatrix(predict(model_fitting_svm_Poly, newdata = train_set), train_set$diabetes)
confusionMatrix(predict(model_fitting_svm_Poly, newdata = test_set), test_set$diabetes)

#Tuning cost with SVM Linear
# Cost is penalties for misclassification
model_fitting_svm_tuned = train(data = train_set, diabetes~., method = "svmLinear",
                               trControl = trainControl(method = "repeatedcv", number = 10, repeats = 5),
                               preProcess = c("center", "scale"),
                               tuneGrid = expand.grid(C = seq(0, 2, length = 20)))
model_fitting_svm_tuned
#Accuracy remains the same as before
confusionMatrix(predict(model_fitting_svm_tuned, newdata = train_set), train_set$diabetes)
confusionMatrix(predict(model_fitting_svm_tuned, newdata = test_set), test_set$diabetes)


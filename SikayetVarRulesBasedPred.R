library(readxl)
library(C50)
library(gmodels)
library(caret)
library(RWeka)
library(rpart)
library(readr)
library(caTools)
library(dplyr)
library(party)
library(partykit)
library(rpart.plot)
library(arules)
library(class)


data_set <- read.xlsx("SikayetVarData.xlsx", sheetIndex=2)
View(data_set)

data_set <- na.omit(data_set)
View(data_set)

table(data_set$SatisfactionLevel)

data_set_analysis<-data_set[-c(1:3,5,6,9,12)]
#data_set_analysis<-data_set[-c(1:3,6,9,12)]
View(data_set_analysis)

str(data_set_analysis)

#discretization of cont. variables
# ResponseRate <- discretize(data_set_analysis$ResponsePerc, "fixed", 
#                              breaks =c(-Inf, 20, 40, 60, 80, Inf),
#                              labels = c("Very Poor", "Poor", "Average",
#                                         "Good", "Excellent"))
# 
# data_set_analysis$ResponseRate<- ResponseRate
# View(data_set_analysis)
# 
# 
# SolvedRate <- discretize(data_set_analysis$SolvedPerc, "fixed", 
#                                breaks =c(-Inf, 20, 40, 60, 80, Inf),
#                                labels = c("Very Poor", "Poor", "Average",
#                                           "Good", "Excellent"))
# 
# data_set_analysis$SolvedRate<- SolvedRate
# View(data_set_analysis)
# 
# ResponseSpeedLevel <- discretize(data_set_analysis$ResponseSpeed, "interval", 
#                                 breaks =5, labels = c("Excellent", "Good", "Average",
#                                                       "Poor", "Very Poor"))
# data_set_analysis$ResponseSpeedLevel<- ResponseSpeedLevel
# View(data_set_analysis)


SatisfactionLevel <- discretize(data_set_analysis$SatisfactionScore, "interval", 
                                breaks =5, labels = c("Very Poor", "Poor", "Average",
                                                      "Good", "Excellent"))
data_set_analysis$SatisfactionLevel<- SatisfactionLevel
View(data_set_analysis)

#data_set_analysis$Sector<-as.factor(data_set_analysis$Sector)
data_set_analysis<-data_set_analysis[-c(1,5)]
data_set_analysis<-data_set_analysis[-c(1,6)]
View(data_set_analysis)

str(data_set_analysis)



# set.seed(123)
# train_sample <- sample(1246,875)
# train_data <- data_set_analysis[train_sample ,]
# test_data <- data_set_analysis[-train_sample , ]

# Preparation of train-test datasets
set.seed(235)
sample_data = sample.split(data_set_analysis$SatisfactionLevel, SplitRatio = 0.70)
View(sample_data)
train_data <- subset(data_set_analysis, sample_data == TRUE)
test_data <- subset(data_set_analysis, sample_data == FALSE)

View(train_data)
View(test_data)

prop.table(table(train_data$SatisfactionLevel))
prop.table(table(test_data$SatisfactionLevel))

#  tuning parameters for a particular model
modelLookup("rf")


# Implementation of CART Model
rtree <- rpart(SatisfactionLevel ~ ., train_data)
rtree <- rpart(SatisfactionLevel ~ ., train_data, cp=0.01)

rpart.plot(rtree)
printcp(rtree)

rtree_pred <- predict(rtree, test_data, type = 'class')
cm<-confusionMatrix(rtree_pred, test_data$SatisfactionLevel)
test_accuracy <- round(cm$overall[1]*100, digits=2)
zeroR_accuracy <- round(cm$overall[5]*100, digits=2)

cm$byClass
View(cm$byClass)

write.xlsx(cm$byClass, "SVPredResults.xlsx")

# Implementation of C5 Model
C5_model <- C5.0(train_data[-4], train_data$SatisfactionLevel)
summary(C5_model)

# To apply our decision tree to the test dataset,
C5_model_pred <- predict(C5_model, test_data)

CrossTable(test_data$SatisfactionLevel, C5_model_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual ', 'predicted'))

cm<-confusionMatrix(C5_model_pred, test_data$SatisfactionLevel)
test_accuracy <- round(cm$overall[1]*100, digits=2)
zeroR_accuracy <- round(cm$overall[5]*100, digits=2)

summary(C5_model_pred)


## Implementation of RF
#randomForest packet
library(randomForest)
set.seed(500)
rf <- randomForest(SatisfactionLevel ~ ., data = train_data, ntree=350)
rf_model_pred <- predict(rf, test_data)
cm<-confusionMatrix(rf_model_pred, test_data$SatisfactionLevel)
test_accuracy <- round(cm$overall[1]*100, digits=2)
zeroR_accuracy <- round(cm$overall[5]*100, digits=2)

cm$byClass
View(cm$byClass)

write.xlsx(cm$byClass, "SVPredResults.xlsx")


## Implementation of SVM
train_data.scaled <- as.data.frame(apply(train_data[-4], 2, function(x) (x - min(x))/(max(x)-min(x))))

train_data.scaled = data.frame(train_data.scaled,train_data[4])

test_data.scaled <-  as.data.frame(apply(test_data[-4], 2, function(x) (x - min(x))/(max(x)-min(x))))

test_data.scaled = data.frame(test_data.scaled,test_data[4])

str(train_data.scaled)
str(test_data.scaled)

#e1071 packet
library(e1071)
set.seed(1234)
svm <- svm(SatisfactionLevel~., data=train_data.scaled, type="C-classification", 
           kernel="radial", cost=20, gamma=1, scaled=c(), probability = TRUE)

svm_model_pred <- predict(svm, test_data.scaled)
cm<-confusionMatrix(svm_model_pred, test_data$SatisfactionLevel)


test_accuracy <- round(cm$overall[1]*100, digits=2)
zeroR_accuracy <- round(cm$overall[5]*100, digits=2)

cm$byClass
View(cm$byClass)
write.xlsx(cm$byClass, "SVPredResults.xlsx")


## Implementation of NB (each feature must be categorical)
#e1071 packet
data(HouseVotes84, package = "mlbench")
set.seed(1234)
nb <- naiveBayes(SatisfactionLevel ~., data = train_data)
nb_model_pred <- predict(nb, test_data)
cm<-confusionMatrix(nb_model_pred, test_data$SatisfactionLevel)
test_accuracy <- round(cm$overall[1]*100, digits=2)
zeroR_accuracy <- round(cm$overall[5]*100, digits=2)



## Implementation of KNN
View(train_dataLabels)
View(sample_data)
str(test_data)

## store target variable class labels
train_dataLabels <- train_data[, 4]
test_dataLabels <- test_data[, 4]

View(train_data[-4])
knn_model <- knn(train = train_data[-4], test = test_data[-4], 
                 cl = train_dataLabels, k=35)

cm<-confusionMatrix(knn_model, test_dataLabels)
test_accuracy <- round(cm$overall[1]*100, digits=2)
zeroR_accuracy <- round(cm$overall[5]*100, digits=2)

summary(knn_model)


## Implementation of ANN
library(neuralnet)
#View(data_set_analysis.scaled)

##Creating Dummy Variables for Categorical Variables using one-hot coding
#define one-hot encoding function
dmy_train <- dummyVars(" ~ .", data = train_data, fullRank=T)
dmy_test <- dummyVars(" ~ .", data = test_data, fullRank=T)
#perform one-hot encoding on data frame
train_data_analysis <- data.frame(predict(dmy_train, newdata = train_data))
test_data_analysis <- data.frame(predict(dmy_test, newdata = test_data))
View(train_data_analysis)

train_data_analysis.scaled <- as.data.frame(apply(train_data_analysis, 2, function(x) (x - min(x))/(max(x)-min(x))))
test_data_analysis.scaled <-  as.data.frame(apply(test_data_analysis, 2, function(x) (x - min(x))/(max(x)-min(x))))

View(train_data_analysis.scaled)

ann <-  neuralnet(SatisfactionLevel.Poor + SatisfactionLevel.Average +
                  SatisfactionLevel.Good + SatisfactionLevel.Excellent ~ 
                  ResponsePerc + ResponseSpeed + SolvedPerc,
                  data = train_data_analysis.scaled, hidden=1, learningrate=0.1, 
                  stepmax = 1e+06, linear.output=FALSE, algorithm="backprop", err.fct="sse")

summary(ann)
print(ann)
plot(ann)

pr.nn <- compute(ann, test_data_analysis.scaled)

ann_model_pred <- predict(ann, test_data)
length(ann_model_pred)
View(ann_model_pred)
cm<-confusionMatrix(ann_model_pred, test_data$SatisfactionLevel)
test_accuracy <- round(cm$overall[1]*100, digits=2)
zeroR_accuracy <- round(cm$overall[5]*100, digits=2)




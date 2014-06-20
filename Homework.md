Coursera practical machine learning homework
========================================================
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??? a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har).
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.
In the beginning lets load necessary packages and data from the internet.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.0.2 r169 Copyright (c) 2006-2013 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
if (!file.exists("pml-training.csv"))
    download.file(
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
        "pml-training.csv")
if (!file.exists("pml-testing.csv"))
    download.file(
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
        "pml-testing.csv")
d<-read.csv("pml-training.csv", head=T)
testData<-read.csv("pml-testing.csv", head=T)
dim(d);
```

```
## [1] 19622   160
```

```r
head(d$classe)
```

```
## [1] A A A A A A
## Levels: A B C D E
```
Classe is a discrete variable with 5 possible outputs. 
Dataset contains a lot of columns, however many of them contain mostly empty 
strings or NA's. Therefore, before applying algorithm, data needs to be prepared 
and cleansed. It was decided to remove time based data and remove all columns that 
have less data than the training dataset.

```r
d$X<-NULL;                    testData$X<-NULL;
d$user_name<-NULL;            testData$user_name<-NULL
d$raw_timestamp_part_1<-NULL; testData$raw_timestamp_part_1<-NULL
d$raw_timestamp_part_2<-NULL; testData$raw_timestamp_part_2<-NULL
d$cvtd_timestamp<-NULL;       testData$cvtd_timestamp<-NULL
d$new_window<-NULL;           testData$new_window<-NULL
d$num_window<-NULL;           testData$num_window<-NULL
#Assign NA's to empty columns
d[(d=="")]<-NA;               testData[(testData=="")]<-NA
#Select only columns that have all the data in training dataset
selectedFeatureNames<-names(d)[colSums(!is.na(d[,]))==19622]
#Select only columns that have all the data in training dataset
selectedTestFeatureNames<-names(testData)[colSums(!is.na(testData[,]))==20]
#Take only features that make sense in both datasets
selectedFeatureNames<-intersect(selectedFeatureNames, selectedTestFeatureNames)
y<-d$classe
d<-d[,selectedFeatureNames]
testData<-testData[,selectedFeatureNames]
```
For the model validation purposes let's split the dataset into training and 
cross-validation datasets. 

```r
set.seed(34521)
inTrain<-createDataPartition(y, p=0.75, list=F)
y_train<-y[inTrain]
y_cv<-y[-inTrain]
train<-d[inTrain,]
cv<-d[-inTrain,]
```
There is still a large number of variables (52) to consider:

```r
length(names(train))
```

```
## [1] 52
```
Therefore, PCA (as all the variables left are numeric) will be applied to reduce 
number of data dimensions. It will help to save the memory as well.

```r
preproc<-preProcess(train,method="pca", thresh=0.95)
preproc$numComp
```

```
## [1] 25
```

```r
trainPC<-predict(preproc, train)
cvPC<-predict(preproc, cv)
#Save PCA function for later reuse
save(preproc, file="preproc-rmd.rda")
```
This leaves us with 25 dimensions to model.
Considering the non-linearity of the model, we will use bagging to create model.

```r
fit<-train(y_train~.,data=trainPC,method="treebag",prox=T, model=F)
```

```
## Loading required package: ipred
## Loading required package: plyr
```

```r
save(fit, file="model-rmd.rda")
```
Let's validate model accuracy using using cross-validation dataset.

```r
confusionMatrix(y_cv, predict(fit, cvPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1362    8    8   15    2
##          B   20  898   26    0    5
##          C    9   26  807   10    3
##          D    4    3   39  753    5
##          E    3    5   11    5  877
## 
## Overall Statistics
##                                         
##                Accuracy : 0.958         
##                  95% CI : (0.952, 0.963)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.947         
##  Mcnemar's Test P-Value : 6.89e-05      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.974    0.955    0.906    0.962    0.983
## Specificity             0.991    0.987    0.988    0.988    0.994
## Pos Pred Value          0.976    0.946    0.944    0.937    0.973
## Neg Pred Value          0.990    0.989    0.979    0.993    0.996
## Prevalence              0.285    0.192    0.182    0.160    0.182
## Detection Rate          0.278    0.183    0.165    0.154    0.179
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.982    0.971    0.947    0.975    0.989
```
With this model, we can predict type of the activity with 95% accuracy.
We can now predict outcome on the test dataset:

```r
testPC<-predict(preproc,testData)
result<-predict(fit,testPC)
result
```

```
##  [1] B A C A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
We need to save the prediction output into separate files for grading.

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(result)
```
Answers files are written to the working folder.

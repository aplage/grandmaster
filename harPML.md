# Human Activity Recognition: predicting models for Weight Lifting Exercise
Andrey Pereira Lage  
March 27, 2016  

#Summary

This project on Human Activity Recognition was performed as the final assignment of the **Practical Machine Learning** course of the **Data Science** Specialization of **Johns Hopkins Bloomberg School of Public Health / Coursera**. The goal of the project was to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


#Data reading and processing

The data was read from .csv file as **training** ("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv") and **test** ("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv") datasets. 



```
## Note: the specification for S3 class "family" in package 'MatrixModels' seems equivalent to one from package 'lme4': not turning on duplicate class definitions for this class.
```

The dimensions of datasets were:

```r
dim(training)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```

Then, the **training** dataset was evaluated for near zero variables that were removed from the dataset, creating the **training2** dataset with the following dimensions:

```r
nzvindex <- nearZeroVar(training, saveMetrics=TRUE)
training2 <- training[,nzvindex$nzv==FALSE]
dim(training2)
```

```
## [1] 19622   100
```

In the next step, variables with more than 60% of NA were also removed and a new dataset were created, **training3**.

```r
removeNA <- vector()
for(i in 1:length(names(training2))){
        if(sum(is.na(training2[,i]))/length(training2[,i]) >= 0.6){
                NAremove <- i
                removeNA <- c(removeNA, NAremove)
        }
        
}
training3 <- training2[,-removeNA]
```

The dimensions of **training3** dataset were:

```r
dim(training3)
```

```
## [1] 19622    59
```


The first variable, that seems to be just an ordered number or id, was also removed and the dataset was stored as **training4**.



In the next step, the correlation among variables in **training4** was evaluated and the number and percentage of variables with correlation greater than 80% were recorded.


```r
cortraining4 <- abs(cor(training4[,-c(1,4,58)]))
diag(cortraining4) <- 0
length(which(cortraining4 > 0.8, arr.ind = T))/2
```

```
## [1] 38
```

```r
paste("Percentage of correlated variables =",
      round((length(which(cortraining4 > 0.8, arr.ind = T))/2)*100/ dim(training4)[2], 2), 
        "%", 
        sep = " ")
```

```
## [1] "Percentage of correlated variables = 65.52 %"
```

Those results indicate that there is a large amont of correlation among the variables in **training4** dataset. Thus, to control for this large ammount of correlation and decrease the number of variables, **principal component analysis** was used in all preprocessing of the dataset.

The **training4** dataset was then divided in **training5** (70%) and **testing** (30%) datasets.


```r
set.seed(54321)
inTrain <- createDataPartition(training4$classe,
                               p = 0.70, 
                               list = FALSE)

training5 <- training4[inTrain,]
testing <- training4[-inTrain,]
```

##Scatterplot matrix of all 58 variables in **training5** dataset

```r
suppressWarnings(splom(training5, panel = panel.smoothScatter, raster = T))
```

![](harPML_files/figure-html/unnamed-chunk-9-1.png)

#Evaluation of the models for **Classe** prediction in HAR dataset

For all analysis, the **training5** dataset was splitted into 5-fold subdatasets for cross-validation.


```r
set.seed(54321)
fitmodelcontrol <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
```

Due to the characteristics of the data, three different algoritms were selected to be tested: **Random Forest**, **Generalized Boosted Regression** and **Linear Discriminant Analysis**

```r
set.seed(54321)
suppressPackageStartupMessages(suppressWarnings(
        fitrf <- train(classe ~ ., 
               "rf",
               preProcess="pca", 
               trControl = fitmodelcontrol, 
               data = training5)))
suppressPackageStartupMessages(suppressWarnings(
        fitgbm <- train(classe ~ ., 
                "gbm", 
                verbose = FALSE, 
                preProcess="pca", 
                trControl = fitmodelcontrol, 
                data = training5)))
suppressPackageStartupMessages(suppressWarnings(
        fitlda <- train(classe ~ .,
                "lda", 
                preProcess="pca", 
                trControl = fitmodelcontrol, 
                data = training5)))
```

The models were fitted with the **testing** dataset.


```r
set.seed(54321)
predrf <- predict(fitrf, testing)
predgbm <- predict(fitgbm, testing)
predlda <- predict(fitlda, testing)
```


A new dataset, **testnew**, was created from the predicted values of the former models and **classe** (**classetr** in **testnew**).

```r
classets <- testing$classe
testnew <- data.frame(predrf, predgbm, predlda, classets)
```

A new **Random Forest** model was fitted with the **testnew** dataset, which has the predicted values of the former models, and stored in **fit123**.


```r
set.seed(54321)
fit123 <- train(classets ~., method = "rf", data = testnew)
```

The **fit123** model was then predicit with the **testing** dataset.

```r
set.seed(54321)
preda123 <- predict(fit123, testing)
```

The following plots show the results of the prediction using all models.


```r
par(mfrow = c(2,2))
plot(predrf, testing$classe, 
     main = "Predicition with Random Forrest model",
     ylab = "Model Predicited values",
     xlab = "Original Classe values")

plot(predgbm, testing$classe, 
     main = "Predicition with Generalized Boosted Regression model",
     ylab = "Model Predicited values",
     xlab = "Original Classe values")

plot(predlda, testing$classe, 
     main = "Predicition with Linear Discriminant Analysis model",
     ylab = "Model Predicited values",
     xlab = "Original Classe values")

plot(preda123, testing$classe, 
     main = "Predicition with Random Forrest model
     on the combination of former models",
     ylab = "Model Predicited values",
     xlab = "Original Classe values")
```

![](harPML_files/figure-html/unnamed-chunk-16-1.png)

```r
par(mfrow = c(1,1))
```


Confusion matrix were created for all individual models and for the **fit123** model with the predicted values from the former models.


```r
set.seed(54321)
accrf <- confusionMatrix(predrf, testing$classe)
accgbm <- confusionMatrix(predgbm, testing$classe)
acclda <- confusionMatrix(predlda, testing$classe)
acc123 <- confusionMatrix(preda123, testing$classe)

accrf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1663   14    1    0    0
##          B    9 1116    8    0    0
##          C    2    9 1012   29    0
##          D    0    0    5  933    6
##          E    0    0    0    2 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9856          
##                  95% CI : (0.9822, 0.9884)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9817          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9934   0.9798   0.9864   0.9678   0.9945
## Specificity            0.9964   0.9964   0.9918   0.9978   0.9996
## Pos Pred Value         0.9911   0.9850   0.9620   0.9883   0.9981
## Neg Pred Value         0.9974   0.9952   0.9971   0.9937   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2826   0.1896   0.1720   0.1585   0.1828
## Detection Prevalence   0.2851   0.1925   0.1788   0.1604   0.1832
## Balanced Accuracy      0.9949   0.9881   0.9891   0.9828   0.9970
```

```r
accgbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1587   83    7    1    0
##          B   61  987   72    7    0
##          C   26   62  936   81    3
##          D    0    7   11  845   56
##          E    0    0    0   30 1023
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9138          
##                  95% CI : (0.9064, 0.9209)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.891           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9480   0.8665   0.9123   0.8766   0.9455
## Specificity            0.9784   0.9705   0.9646   0.9850   0.9938
## Pos Pred Value         0.9458   0.8758   0.8448   0.9195   0.9715
## Neg Pred Value         0.9793   0.9681   0.9812   0.9760   0.9878
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2697   0.1677   0.1590   0.1436   0.1738
## Detection Prevalence   0.2851   0.1915   0.1883   0.1562   0.1789
## Balanced Accuracy      0.9632   0.9185   0.9384   0.9308   0.9696
```

```r
acclda
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1301  142   19    0    0
##          B  296  714  221   61    0
##          C   77  254  705  127    0
##          D    0   29   59  607  263
##          E    0    0   22  169  819
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7045          
##                  95% CI : (0.6927, 0.7161)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6279          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7772   0.6269   0.6871   0.6297   0.7569
## Specificity            0.9618   0.8782   0.9057   0.9287   0.9602
## Pos Pred Value         0.8899   0.5526   0.6062   0.6336   0.8109
## Neg Pred Value         0.9157   0.9075   0.9320   0.9275   0.9461
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2211   0.1213   0.1198   0.1031   0.1392
## Detection Prevalence   0.2484   0.2195   0.1976   0.1628   0.1716
## Balanced Accuracy      0.8695   0.7525   0.7964   0.7792   0.8586
```

```r
acc123
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1663   14    1    0    0
##          B    9 1116    8    0    0
##          C    2    9 1012   29    0
##          D    0    0    5  933    6
##          E    0    0    0    2 1076
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9856          
##                  95% CI : (0.9822, 0.9884)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9817          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9934   0.9798   0.9864   0.9678   0.9945
## Specificity            0.9964   0.9964   0.9918   0.9978   0.9996
## Pos Pred Value         0.9911   0.9850   0.9620   0.9883   0.9981
## Neg Pred Value         0.9974   0.9952   0.9971   0.9937   0.9988
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2826   0.1896   0.1720   0.1585   0.1828
## Detection Prevalence   0.2851   0.1925   0.1788   0.1604   0.1832
## Balanced Accuracy      0.9949   0.9881   0.9891   0.9828   0.9970
```

The summary of the accuracy of all four models is shown in the next table.


```r
resa <- rbind(accrf$overall[1], accgbm$overall[1], acclda$overall[1], acc123$overall[1])
rownames(resa) <- c("RF", "GBM", "LDA", "All")
resa
```

```
##      Accuracy
## RF  0.9855565
## GBM 0.9138488
## LDA 0.7045030
## All 0.9855565
```

As the Random Forest model **fitrf** and the Random Forest model using all three former models **fit123** showed similar performance, due to its lesser complexity, the **fitrf** model was selected for futher analysis.


```r
plot(fitrf, main = "Accuracy of the random forest model fitrf")
```

![](harPML_files/figure-html/unnamed-chunk-19-1.png)

The expected out of sample error was estimated as (1 - the accuracy of the selected **fitrf** in the **testing** dataset). As the **testing** dataset was only used to evaluate the model after fitting, the expected out of sample error of 1.44% is a very good estimate.

#Reference

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


#Using the **fitrf** model to predict outcome in **test** dataset.

The **fitrf** model was used to predict the outcome in the **test** dataset as follows:


```r
predsubmission <- predict(fitrf, test)
predsubmission
```

```
##  [1] B A C A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```






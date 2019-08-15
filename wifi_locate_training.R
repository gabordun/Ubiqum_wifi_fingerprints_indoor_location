##########################################################################################
##                                                                                      ##
#############   Locate a mobile device's position                                       ##
##                                         upon its wifi fingerprints,                  ##
##                                            training the models                 ########
##              Author: Gabor Dunai                                                     ##
##              Version: 1.0                                                            ##
##              Date: 07.2019                                                           ##
##                                                                                      ##
##########################################################################################

####################  Part 0: directory, libraries, dataset   ################

#################### set directory, call libraries ###########################

setwd("A:/B/Ubiqum/module3/wifi")
library(dplyr)
library(ggplot2)
library(caret)
library(C50)
library(e1071)
library(gbm)
library(randomForest)
library(kknn)
library(mlbench)
library(export)

####################     optional: load saved environment ####################

            load("A:/B/Ubiqum/module3/wifi/wifi_locate_envir.RData")

####################          import data         ############################

training<-read.csv('A:/B/Ubiqum/module3/wifi/trainingData.csv')
validate<-read.csv('A:/B/Ubiqum/module3/wifi/validationData.csv')

####################    first check of the database        ###################
## training set
### classes

apply(training,2,class)

### frequencies

table(training$FLOOR)
table(training$BUILDINGID)
table(training$SPACEID)
table(training$RELATIVEPOSITION)
table(training$USERID)
table(training$PHONEID)
table(training$TIMESTAMP)

### number of different values

length(unique(training$LONGITUDE))
length(unique(training$LATITUDE))
length(unique(training$SPACEID))

### scatter plots

ggplot(training,aes(x=LONGITUDE, y=LATITUDE))+geom_point()
ggplot(training,aes(x=BUILDINGID, y=SPACEID))+geom_point()
ggplot(training,aes(x=FLOOR, y=SPACEID))+geom_point()
ggplot(training,aes(x=SPACEID, y=LATITUDE))+geom_point()
ggplot(training,aes(x=SPACEID, y=LONGITUDE))+geom_point()
ggplot(training,aes(x=SPACEID, y=RELATIVEPOSITION))+geom_point()
ggplot(training,aes(x=RELATIVEPOSITION, y=LATITUDE))+geom_point()
ggplot(training,aes(x=RELATIVEPOSITION, y=LONGITUDE))+geom_point()

##validation set
### classes

apply(validate,2,class)

### frequencies

table(validate$FLOOR)
table(validate$BUILDINGID)
table(validate$SPACEID)
table(validate$RELATIVEPOSITION)
table(validate$USERID)
table(validate$PHONEID)
table(validate$TIMESTAMP)

### number of unique values in each WAP's signal strength levels

freqie<-apply(validate,2,unique)
freqie<-lapply(freqie,length)
freqie<-as.numeric(freqie)
freqie<-freqie[-(521:529)]

### number of different values

length(unique(validate$LONGITUDE))
length(unique(validate$LATITUDE))

###plots

ggplot(validate,aes(x=LONGITUDE, y=LATITUDE))+geom_point()
barplot(freqie)

# delete plots

graphics.off()

##################        preprocess data        ##############################

# identification of irrelevant WAPs

features<-colnames(validate)
features<-features[-(521:529)]
SELECTION<-data.frame(cbind(features,freqie))
SELECTION$freqie=as.numeric(SELECTION$freqie)

quantile(freqie,0.6)

# filter out irrelevant WAPs

SELECTED<-filter(SELECTION, freqie>quantile(freqie,0.6))
helpfilter<-as.vector(SELECTED$features)
helpfilter2<-c(helpfilter,'LONGITUDE','LATITUDE','FLOOR')
filter_training<-select(training,helpfilter2)

# optional - identification of irrelevant WAPs

        nearzerovar<-data.frame(nearZeroVar(validate, freqCut= 99.25/0.75,saveMetrics = TRUE,names = TRUE))
        nearzerovar<-cbind(nearzerovar,colnames(validate))
        colnames(nearzerovar)[5]<-'colnames'
        nearzerovar_filt<-filter(nearzerovar,nearzerovar$nzv=='FALSE')

# optional - filter out irrelevant WAPs

        helpfilter<-as.vector(nearzerovar_filt$colnames)
        filter_training<-select(training,helpfilter)

# dataset for training the model

filter_training2<-filter_training

####################      Part 1: train predictive models    #################

###################             predict longitude           ##################

## optional - set floors

            filter_training2<-filter(filter_training2, FLOOR!='2')
            filter_training2<-filter(filter_training2, FLOOR!='3')
            filter_training2<-filter(filter_training2, FLOOR!='4')

##  setting sample, training, testing

set.seed(123)
rownum<-nrow(filter_training2)
a<-0.1*rownum

# setting sample

filter_training_longitude<-filter_training2
filter_training_longitude[,c('LATITUDE','FLOOR')]<-list(NULL)
Sample<-(filter_training_longitude[sample(1:nrow(filter_training2),floor(a),replace=FALSE), ])

# partioning to training and testing sets

inTraining <- createDataPartition(Sample$LONGITUDE, p = .7, list = FALSE)
trainingset <- Sample[inTraining, ]
testingset <- Sample[-inTraining, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 1)


## model kNN

Mod1_kNN_Long<-train(LONGITUDE~.,
                   data=trainingset, 
              method="knn",trControl=fitControl,
              tuneGrid = expand.grid(k = 1:25),
              tuneLength=10)
Mod1_kNN_Long
varImp(Mod1_kNN_Long)

## SVM, linear kernel

Mod2_SVM_Long<-svm(LONGITUDE~.,data=trainingset, kernel="linear",
                scale = FALSE,
                tolerance = 0.01,
                cost = c(1,10),
                cross = 3)
Mod2_SVM_Long

## Gradient boosted trees

Mod3_GBM_Long<-gbm(LONGITUDE~., data=trainingset, distribution = "gaussian",
                   n.trees = 100,
                   interaction.depth = 3, n.minobsinnode = 3,
                   shrinkage = 0.05,
                   bag.fraction = 0.5, train.fraction = 1, cv.folds = 10,
                   keep.data = TRUE, verbose = FALSE)
Mod3_GBM_Long

## Random forest

Mod4_Rf_Long<-randomForest(LONGITUDE~.,data=trainingset, mtry=25,ntree= 120,
                     trControl=fitControl)
Mod4_Rf_Long

## model weighted kNN

Mod5_kkNN_Long<-train.kknn(LONGITUDE~.,
                      trainingset,
                      kmax = 11,
                      ks = NULL,
                      distance = 2)
cv.kknn(LONGITUDE~., trainingset, kcv = 10)
Mod5_kkNN_Long

###############     model validations/ predicting longitude  ##################

TestkNN_Lo<-round(predict(Mod1_kNN_Long,testingset),0)
TestSVM_Lo<-round(predict(Mod2_SVM_Long,testingset),0)
TestGBM_Lo<-round(predict(Mod3_GBM_Long,testingset),0)
TestRF_Lo<-round(predict(Mod4_Rf_Long,testingset),0)
TestkkNN_Lo<-round(predict(Mod5_kkNN_Long,testingset),0)
TestMatrix_Lo<-data.frame(TestkNN_Lo,TestSVM_Lo,
                          TestGBM_Lo,TestRF_Lo,TestkkNN_Lo,round(testingset$LONGITUDE,0))

accuracy_long<-rbind(c('kNN','SVM','GBM','RF','kkNN'),
c(round(Metrics::accuracy(TestMatrix_Lo$TestkNN_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
round(Metrics::accuracy(TestMatrix_Lo$TestSVM_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
round(Metrics::accuracy(TestMatrix_Lo$TestGBM_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
round(Metrics::accuracy(TestMatrix_Lo$TestRF_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
round(Metrics::accuracy(TestMatrix_Lo$TestkkNN_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4)))

MAE_long<-rbind(c('kNN','SVM','GBM','RF','kkNN'),
    c(round(Metrics::mae(TestMatrix_Lo$TestkNN_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
    round(Metrics::mae(TestMatrix_Lo$TestSVM_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
    round(Metrics::mae(TestMatrix_Lo$TestGBM_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
    round(Metrics::mae(TestMatrix_Lo$TestRF_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
    round(Metrics::mae(TestMatrix_Lo$TestkkNN_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4)))

MAPE_long<-rbind(c('kNN','SVM','GBM','RF','kkNN'),
    c(round(Metrics::mape(TestMatrix_Lo$TestkNN_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
    round(Metrics::mape(TestMatrix_Lo$TestSVM_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
    round(Metrics::mape(TestMatrix_Lo$TestGBM_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
    round(Metrics::mape(TestMatrix_Lo$TestRF_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4),
    round(Metrics::mape(TestMatrix_Lo$TestkkNN_Lo, TestMatrix_Lo$round.testingset.LONGITUDE..0.),4)))

###################             predict latitude           ##################

##  setting sample, training, testing

set.seed(123)
rownum<-nrow(filter_training2)
a<-0.1*rownum

# setting sample

filter_training_latitude<-filter_training2
filter_training_latitude[,c('FLOOR','LONGITUDE')]<-list(NULL)
Sample<-(filter_training_latitude[sample(1:nrow(filter_training2),floor(a),replace=FALSE), ])

# partioning to training and testing sets

inTraining <- createDataPartition(Sample$LATITUDE, p = .7, list = FALSE)
trainingset <- Sample[inTraining, ]
testingset <- Sample[-inTraining, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 1)


## model kNN

  Mod21_kNN_lat<-train(LATITUDE~.,
                         data=trainingset, 
                         method="knn",trControl=fitControl,
                         tuneGrid = expand.grid(k = 1:25),
                         tuneLength=10)
  Mod21_kNN_lat
  varImp(Mod21_kNN_lat)

## SVM, linear kernel

Mod22_SVM_lat<-svm(LATITUDE~.,data=trainingset, kernel="linear",
                     scale = FALSE,
                     tolerance = 0.01,
                     cost = c(1,10),
                     cross = 3)
Mod22_SVM_lat

## Gradient boosted trees

Mod23_GBM_lat<-gbm(LATITUDE~., data=trainingset, distribution = "gaussian",
                     n.trees = 100,
                     interaction.depth = 3, n.minobsinnode = 3,
                     shrinkage = 0.05,
                     bag.fraction = 0.5, train.fraction = 1, cv.folds = 10,
                     keep.data = TRUE, verbose = FALSE)
Mod23_GBM_lat

## Random forest

Mod24_Rf_lat<-randomForest(LATITUDE~.,data=trainingset, mtry=40, tree=120,
                             trControl=fitControl)
Mod24_Rf_lat

## model weighted kNN

Mod25_kkNN_lat<-train.kknn(LATITUDE~.,
                             trainingset,
                             kmax = 11,
                             ks = NULL,
                             distance = 2)
cv.kknn(LATITUDE~., trainingset, kcv = 10)
Mod25_kkNN_lat

###############     model validations/ predicting latitude  ##################

TestkNN_lat<-round(predict(Mod21_kNN_lat,testingset),0)
TestSVM_lat<-round(predict(Mod22_SVM_lat,testingset),0)
TestGBM_lat<-round(predict(Mod23_GBM_lat,testingset),0)
TestRF_lat<-round(predict(Mod24_Rf_lat,testingset),0)
TestkkNN_lat<-round(predict(Mod25_kkNN_lat,testingset),0)
TestMatrix_lat<-data.frame(TestkNN_lat,TestSVM_lat,
                         TestGBM_lat,TestRF_lat,TestkkNN_lat,testingset$LATITUDE)

accuracy_lat<-rbind(c('kNN','SVM','GBM','RF','kkNN'),
     c(round(Metrics::accuracy(TestMatrix_lat$TestkNN_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::accuracy(TestMatrix_lat$TestSVM_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::accuracy(TestMatrix_lat$TestGBM_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::accuracy(TestMatrix_lat$TestRF_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::accuracy(TestMatrix_lat$TestkkNN_lat, TestMatrix_lat$testingset.LATITUDE),4)))

MAE_lat<-rbind(c('kNN','SVM','GBM','RF','kkNN'),
     c(round(Metrics::mae(TestMatrix_lat$TestkNN_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::mae(TestMatrix_lat$TestSVM_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::mae(TestMatrix_lat$TestGBM_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::mae(TestMatrix_lat$TestRF_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::mae(TestMatrix_lat$TestkkNN_lat, TestMatrix_lat$testingset.LATITUDE),4)))

MAPE_lat<-rbind(c('kNN','SVM','GBM','RF','kkNN'),
     c(round(Metrics::mape(TestMatrix_lat$TestkNN_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::mape(TestMatrix_lat$TestSVM_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::mape(TestMatrix_lat$TestGBM_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::mape(TestMatrix_lat$TestRF_lat, TestMatrix_lat$testingset.LATITUDE),4),
     round(Metrics::mape(TestMatrix_lat$TestkkNN_lat, TestMatrix_lat$testingset.LATITUDE),4)))

################### combined error for long & lat #########################

long_error<-c(as.numeric(MAE_long[2,]))
lat_error<-c(as.numeric(MAE_lat[2,]))
error_in_meters<-c(long_error^2+lat_error^2)^0.5
error_in_meters

###################             predict floor           ##################

##  setting sample, training, testing

set.seed(123)
rownum<-nrow(filter_training2)
a<-0.1*rownum

# setting sample

filter_training_floor<-filter_training2
filter_training_floor[,c('LATITUDE','LONGITUDE')]<-list(NULL)
Sample<-(filter_training_floor[sample(1:nrow(filter_training2),floor(a),replace=FALSE), ])

# partioning to training and testing sets

inTraining <- createDataPartition(Sample$FLOOR, p = .7, list = FALSE)
trainingset <- Sample[inTraining, ]
testingset <- Sample[-inTraining, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 1)


## model kNN

Mod11_kNN_floor<-train(FLOOR~.,
                     data=trainingset, 
                     method="knn",trControl=fitControl,
                     tuneGrid = expand.grid(k = 1:25),
                     tuneLength=10)
Mod11_kNN_floor
varImp(Mod11_kNN_floor)

## SVM, linear kernel

Mod12_SVM_floor<-svm(FLOOR~.,data=trainingset, kernel="linear",
                     scale = FALSE,
                     tolerance = 0.01,
                     cost = c(1,10),
                     cross = 3)
Mod12_SVM_floor

## Gradient boosted trees

Mod13_GBM_floor<-gbm(FLOOR~., data=trainingset, distribution = "gaussian",
                     n.trees = 100,
                     interaction.depth = 3, n.minobsinnode = 3,
                     shrinkage = 0.05,
                     bag.fraction = 0.5, train.fraction = 1, cv.folds = 10,
                     keep.data = TRUE, verbose = FALSE)
Mod13_GBM_floor

## Random forest

Mod14_Rf_floor<-randomForest(FLOOR~.,data=trainingset, mtry=40,ntree=120,
                           trControl=fitControl)
Mod14_Rf_floor

## model weighted kNN

Mod15_kkNN_floor<-train.kknn(FLOOR~.,
                           trainingset,
                           kmax = 11,
                           ks = NULL,
                           distance = 2)
cv.kknn(FLOOR~., trainingset, kcv = 10)
Mod15_kkNN_floor

###############     model validations/ predicting floor  ##################

TestkNN_f<-round(predict(Mod11_kNN_floor,testingset),0)
TestSVM_f<-round(predict(Mod12_SVM_floor,testingset),0)
TestGBM_f<-round(predict(Mod13_GBM_floor,testingset),0)
TestRF_f<-round(predict(Mod14_Rf_floor,testingset),0)
TestkkNN_f<-round(predict(Mod15_kkNN_floor,testingset),0)
TestMatrix_f<-data.frame(TestkNN_f,TestSVM_f,
                          TestGBM_f,TestRF_f,TestkkNN_f,testingset$FLOOR)

accuracy_f<-rbind(c('kNN','SVM','GBM','RF','kkNN'),
    c(round(Metrics::accuracy(TestMatrix_f$TestkNN_f, TestMatrix_f$testingset.FLOOR),4),
     round(Metrics::accuracy(TestMatrix_f$TestSVM_f, TestMatrix_f$testingset.FLOOR),4),
     round(Metrics::accuracy(TestMatrix_f$TestGBM_f, TestMatrix_f$testingset.FLOOR),4),
     round(Metrics::accuracy(TestMatrix_f$TestRF_f, TestMatrix_f$testingset.FLOOR),4),
     round(Metrics::accuracy(TestMatrix_f$TestkkNN_f, TestMatrix_f$testingset.FLOOR),4)))


####################      Part 2: train the best models on the entire dataset    #################

###################             predict longitude           ##################

# adjust dataset

filter_training_longitude<-filter_training2
filter_training_longitude[,c('LATITUDE','FLOOR')]<-list(NULL)

# partioning to training and testing sets

inTraining <- createDataPartition(filter_training_longitude$LONGITUDE, p = .8, list = FALSE)
trainingset <- filter_training_longitude[inTraining, ]
testingset <- filter_training_longitude[-inTraining, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 1)

## Random forest

  Mod_fin_Rf_Long<-randomForest(LONGITUDE~.,data=trainingset, mtry=25,ntree= 120,
                             trControl=fitControl)
  Mod_fin_Rf_Long

## model weighted kNN

Mod_fin_kkNN_Long<-train.kknn(LONGITUDE~.,
                           trainingset,
                           ks = 5,
                           distance = 2)
cv.kknn(LONGITUDE~., trainingset, kcv = 10)
Mod_fin_kkNN_Long

###############     model validations/ predicting longitude ##################

TestRF_Lo_fin<-round(predict(Mod_fin_Rf_Long,testingset),0)
TestkkNN_Lo_fin<-round(predict(Mod_fin_kkNN_Long,testingset),0)
TestMatrix_Lo_fin<-data.frame(TestRF_Lo_fin,TestkkNN_Lo_fin,round(testingset$LONGITUDE,0))

accuracy_long_fin<-rbind(c('RF','kkNN'),
          c(round(Metrics::accuracy(TestMatrix_Lo_fin$TestRF_Lo_fin, TestMatrix_Lo_fin$round.testingset.LONGITUDE..0.),4),
            round(Metrics::accuracy(TestMatrix_Lo_fin$TestkkNN_Lo_fin, TestMatrix_Lo_fin$round.testingset.LONGITUDE..0.),4)))

MAE_long_fin<-rbind(c('RF','kkNN'),
          c(round(Metrics::mae(TestMatrix_Lo_fin$TestRF_Lo_fin, TestMatrix_Lo_fin$round.testingset.LONGITUDE..0.),4),
            round(Metrics::mae(TestMatrix_Lo_fin$TestkkNN_Lo_fin, TestMatrix_Lo_fin$round.testingset.LONGITUDE..0.),4)))

MAPE_long_fin<-rbind(c('RF','kkNN'),
          c(round(Metrics::mape(TestMatrix_Lo_fin$TestRF_Lo_fin, TestMatrix_Lo_fin$round.testingset.LONGITUDE..0.),4),
            round(Metrics::mape(TestMatrix_Lo_fin$TestkkNN_Lo_fin, TestMatrix_Lo_fin$round.testingset.LONGITUDE..0.),4)))

###################             predict latitude           ##################

# adjust dataset

filter_training_latitude<-filter_training2
filter_training_latitude[,c('FLOOR','LONGITUDE')]<-list(NULL)

# partioning to training and testing sets

inTraining <- createDataPartition(filter_training_latitude$LATITUDE, p = .8, list = FALSE)
trainingset <- filter_training_latitude[inTraining, ]
testingset <- filter_training_latitude[-inTraining, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 1)


## Random forest

Mod_fin_Rf_lat<-randomForest(LATITUDE~.,data=trainingset, mtry=40,ntree=120,
                           trControl=fitControl)
Mod_fin_Rf_lat

## model weighted kNN

Mod_fin_kkNN_lat<-train.kknn(LATITUDE~.,
                           trainingset,
                           ks = 9,
                           distance = 2)
cv.kknn(LATITUDE~., trainingset, kcv = 10)
Mod_fin_kkNN_lat

###############     model validations/ predicting latitude  ##################

TestRF_lat_fin<-round(predict(Mod_fin_Rf_lat,testingset),0)
TestkkNN_lat_fin<-round(predict(Mod_fin_kkNN_lat,testingset),0)
TestMatrix_lat_fin<-data.frame(TestRF_lat_fin,TestkkNN_lat_fin,round(testingset$LATITUDE,0))

accuracy_lat_fin<-rbind(c('RF','kkNN'),
   c(round(Metrics::accuracy(TestMatrix_lat_fin$TestRF_lat_fin, TestMatrix_lat_fin$round.testingset.LATITUDE..0.),4),
     round(Metrics::accuracy(TestMatrix_lat_fin$TestkkNN_lat_fin, TestMatrix_lat_fin$round.testingset.LATITUDE..0.),4)))

MAE_lat_fin<-rbind(c('RF','kkNN'),
   c(round(Metrics::mae(TestMatrix_lat_fin$TestRF_lat_fin, TestMatrix_lat_fin$round.testingset.LATITUDE..0.),4),
     round(Metrics::mae(TestMatrix_lat_fin$TestkkNN_lat_fin, TestMatrix_lat_fin$round.testingset.LATITUDE..0.),4)))

MAPE_lat_fin<-rbind(c('RF','kkNN'),
   c(round(Metrics::mape(TestMatrix_lat_fin$TestRF_lat_fin, TestMatrix_lat_fin$round.testingset.LATITUDE..0.),4),
     round(Metrics::mape(TestMatrix_lat_fin$TestkkNN_lat_fin, TestMatrix_lat_fin$round.testingset.LATITUDE..0.),4)))

################### combined error for long & lat #########################

long_error_fin<-c(as.numeric(MAE_long_fin[2,]))
lat_error_fin<-c(as.numeric(MAE_lat_fin[2,]))
error_in_meters_fin<-c(long_error_fin^2+lat_error_fin^2)^0.5
error_in_meters_fin

###################             predict floor           ##################

filter_training_floor<-filter_training2
filter_training_floor[,c('LATITUDE','LONGITUDE')]<-list(NULL)

# partioning to training and testing sets

inTraining <- createDataPartition(filter_training_floor$FLOOR, p = .8, list = FALSE)
trainingset <- filter_training_floor[inTraining, ]
testingset <- filter_training_floor[-inTraining, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 1)

## Random forest

Mod_fin_Rf_floor<-randomForest(FLOOR~.,data=trainingset, mtry=40,ntree=120,
                             trControl=fitControl)
Mod_fin_Rf_floor

## model weighted kNN

Mod_fin_kkNN_floor<-train.kknn(FLOOR~.,
                             trainingset,
                             ks = 7,
                             distance = 2)
cv.kknn(FLOOR~., trainingset, kcv = 10)
Mod_fin_kkNN_floor

###############     model validations/ predicting floor  ##################

TestRF_f_fin<-round(predict(Mod_fin_Rf_floor,testingset),0)
TestkkNN_f_fin<-round(predict(Mod_fin_kkNN_floor,testingset),0)
TestMatrix_f_fin<-data.frame(TestRF_f_fin,TestkkNN_f_fin,round(testingset$FLOOR,0))

accuracy_f_fin<-rbind(c('RF','kkNN'),
    c(round(Metrics::accuracy(TestMatrix_f_fin$TestRF_f_fin, TestMatrix_f_fin$round.testingset.FLOOR..0.),4),
      round(Metrics::accuracy(TestMatrix_f_fin$TestkkNN_f_fin, TestMatrix_f_fin$round.testingset.FLOOR..0.),4)))

###############      save final models     #################################

saveRDS(Mod_fin_Rf_Long, "A:/B/Ubiqum/module3/wifi/Mod_fin_Rf_Long.RDS")
saveRDS(Mod_fin_Rf_lat, "A:/B/Ubiqum/module3/wifi/Mod_fin_Rf_lat.RDS")
saveRDS(Mod_fin_Rf_floor, "A:/B/Ubiqum/module3/wifi/Mod_fin_Rf_floor.RDS")
saveRDS(Mod_fin_kkNN_Long, "A:/B/Ubiqum/module3/wifi/Mod_fin_kkNN_Long.RDS")
saveRDS(Mod_fin_kkNN_lat, "A:/B/Ubiqum/module3/wifi/Mod_fin_kkNN_lat.RDS")
saveRDS(Mod_fin_kkNN_floor, "A:/B/Ubiqum/module3/wifi/Mod_fin_kkNN_floor.RDS")

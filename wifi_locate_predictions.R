##########################################################################################
##                                                                                      ##
#############   Locate a mobile device's position                                       ##
##                                         upon its wifi fingerprints,                  ##
##                                            predictions                         ########
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

#################### optional: load saved environment ########################

load("A:/B/Ubiqum/module3/wifi/.RData")

####################      optional: load models     ##########################

Mod_fin_Rf_Long<-readRDS("A:/B/Ubiqum/module3/wifi/Mod_fin_Rf_Long.RDS")
Mod_fin_Rf_lat<-readRDS("A:/B/Ubiqum/module3/wifi/Mod_fin_Rf_lat.RDS")
Mod_fin_Rf_floor<-readRDS("A:/B/Ubiqum/module3/wifi/Mod_fin_Rf_floor.RDS")
Mod_fin_kkNN_Long<-readRDS("A:/B/Ubiqum/module3/wifi/Mod_fin_kkNN_Long.RDS")
Mod_fin_kkNN_lat<-readRDS("A:/B/Ubiqum/module3/wifi/Mod_fin_kkNN_lat.RDS")
Mod_fin_kkNN_floor<-readRDS("A:/B/Ubiqum/module3/wifi/Mod_fin_kkNN_floor.RDS")

####################          import data         ############################

validate<-read.csv('A:/B/Ubiqum/module3/wifi/validationData.csv')

####################          preprocessing       ############################

# filter out irrelevant WAPs

filter_validate<-select(validate,helpfilter)

# filter out unnecessary features

filter_validate[,c('BUILDINGID','SPACEID','USERID','PHONEID','RELATIVEPOSITION','TIMESTAMP')]<-list(NULL)

####################          predictions         ############################

## longitude

Predict_long_rf<-round(predict(Mod_fin_Rf_Long,filter_validate),4)
Predict_long_kknn<-round(predict(Mod_fin_kkNN_Long,filter_validate),4)

## latitude

Predict_lat_rf<-round(predict(Mod_fin_Rf_lat,filter_validate),4)
Predict_lat_kknn<-round(predict(Mod_fin_kkNN_lat,filter_validate),4)

## floor

Predict_f_rf<-round(predict(Mod_fin_Rf_floor,filter_validate),0)
Predict_f_kknn<-round(predict(Mod_fin_kkNN_floor,filter_validate),0)

####################      predictions' accuracy   ############################

TestMatrix_long_val<-data.frame(Predict_long_rf,Predict_long_kknn,round(filter_validate$LONGITUDE,0))

MAE_long_val<-rbind(c('RF','kkNN'),
  c(round(Metrics::mae(TestMatrix_long_val$Predict_long_rf, TestMatrix_long_val$round.filter_validate.LONGITUDE..0.),4),
   round(Metrics::mae(TestMatrix_long_val$Predict_long_kknn, TestMatrix_long_val$round.filter_validate.LONGITUDE..0.),4)))

TestMatrix_lat_val<-data.frame(Predict_lat_rf,Predict_lat_kknn,round(filter_validate$LATITUDE,0))

MAE_lat_val<-rbind(c('RF','kkNN'),
  c(round(Metrics::mae(TestMatrix_lat_val$Predict_lat_rf, TestMatrix_lat_val$round.filter_validate.LATITUDE..0.),4),
   round(Metrics::mae(TestMatrix_lat_val$Predict_lat_kknn, TestMatrix_lat_val$round.filter_validate.LATITUDE..0.),4)))

TestMatrix_f_val<-data.frame(Predict_f_rf,Predict_f_kknn,round(filter_validate$FLOOR,0))

accuracy_f_val<-rbind(c('RF','kkNN'),
    c(round(Metrics::accuracy(TestMatrix_f_val$Predict_f_rf, TestMatrix_f_val$round.filter_validate.FLOOR..0.),4),
      round(Metrics::accuracy(TestMatrix_f_val$Predict_f_kknn, TestMatrix_f_val$round.filter_validate.FLOOR..0.),4)))

################### combined error for long & lat #########################

long_error_pred<-c(as.numeric(MAE_long_val[2,]))
lat_error_pred<-c(as.numeric(MAE_lat_val[2,]))
error_in_meters_pred<-c(long_error_pred^2+lat_error_pred^2)^0.5
error_in_meters_pred

####################      export the results      ############################

write.csv(cbind(Predict_lat_rf,Predict_long_rf,Predict_f_rf),
          'A:/B/Ubiqum/module3/wifi/randomforestpred.csv',row.names = FALSE,quote = FALSE)

write.csv(cbind(Predict_lat_kknn,Predict_long_kknn,Predict_f_kknn),
          'A:/B/Ubiqum/module3/wifi/weightedknnpred.csv',row.names = FALSE,quote = FALSE)

###################     visualize prediction      ############################

ggplot(data.frame(cbind(Predict_lat_rf,Predict_long_rf,Predict_f_rf)),
             aes(x=Predict_long_rf, y=Predict_lat_rf))+geom_point()

ggplot(data.frame(cbind(Predict_lat_kknn,Predict_long_kknn,Predict_f_kknn)),
       aes(x=Predict_long_kknn, y=Predict_lat_kknn))+geom_point()

# Ubiqum_wifi_fingerprints_indoor_location project

Predicting real world coordinates (longitude, latitude, floor) of certain places upon wlan access points RSSI (received signal strength indicator) levels with the assistance of machine learning methods.

I used a fast 20k observations to train and validate models such as kNN, support vector machine, gradient boosted trees, random forest and weighted kNN.

After the two best fitting models were choosen, according to their prediction accuracies, I trained these two models (i.e. random forest and weighted kNN) on the entire dataset. Finally, the trained models were validated by 10-times cross-validation method.

The trained models were used on a labelled validation set (which is strictly independent from the training set) to compare their predictions with the real world coordinates.

Somewhat surprisingly, on the validation set the RF model performed much better - while on the test set the weighted kNN model's forecast accuracy was better.



Language: R

Used packages for machine learning: caret, c50, e1071, gbm, randomForest, kknn

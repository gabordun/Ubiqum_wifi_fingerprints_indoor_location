# Ubiqum_wifi_fingerprints_indoor_location
Predicting real world coordinates (longitude, latitude, floor) upon wlan access points RSSI levels with the assistance of machine learning methods.

I used a fast 20k observations training set to train and validate regression models such as kNN, support vector machine, gradient boosted trees, random forest or weighted kNN.

After the two best fitting model-parameters set were choosen I trained these two models (i.e. random forest and weighted kNN)
on the entire dataset, employing cross validation.

The trained models were used on a validation set (which is strictly independent from the training set) to make the predictions
for the real world coordinates. Somewhat surprisingly, on the validation set the RF performed much better - altough the forecast accuracy
of the weighted kNN model was better on the test set.

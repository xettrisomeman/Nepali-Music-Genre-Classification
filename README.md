# BI-LSTM Based Nepali Music Genre Classification 


## Introduction
Code for classifying music genre using Bidirectional LSTM.


## Methods

1. Applied 13 MFCCs bands and saved the results in json
2. Created the model using 32 hidden neurons, output (4 neurons)
3. Adam with default learning rate and CrossEntropyLoss is used.
4. Trained the model for 50 epochs.

> The data is not saved as png file because we can encounter losses of data and change in information in the process. Spectrograms cannot be represented as images.


## Datasets
There is no audio dataset publicly available, so the data had to be created manually. There is about 400 samples of music , each 30 seconds of length. For searching the song youtube is used and for trimming the audio audacity (Open Source software) is used.


# Results
![Losses Comparison]("./assets/genreclassify/lossescomparison.jpg")
![Accuracy Comparison]("./assets/genreclassify/accuracycomparison.jpg")









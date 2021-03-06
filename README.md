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
There is no audio dataset publicly available, so the data had to be created manually. There is about 400 samples of music , each 30 seconds of length. For searching the song youtube is used and for trimming the audio audacity (Open Source software) is used. For training 75% of data is used and for testing remaining 25% is used.

# Results

Losses Comparison
![](genreclassify/assets/lossescomparison.jpg)


Accuracy Comparison
![](genreclassify/assets/accuracycomparison.jpg)


# Make Prediction
``` python
git clone -b test https://github.com/xettrisomeman/Nepali-Music-Genre-Classification
cd Nepali-Music-Classification
pip install -r requirements.txt
cd genreclassify
python predict.py --help
```



#### References
- [Mel-Frequency Cepstral Coefficients Explained](https://www.youtube.com/watch?v=4_SH2nfbQZ8)
- [Preparing dataset for Music genre Classification](https://www.youtube.com/watch?v=szyGiObZymo) 
- Poorjam, Amir Hossein. (2018). Re: Why we take only 12-13 MFCC coefficients in feature extraction?. Retrieved from: https://www.researchgate.net/post/Why_we_take_only_12-13_MFCC_coefficients_in_feature_extraction/5b0fd2b7cbdfd4b7b60e9431/citation/download.






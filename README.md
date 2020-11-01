# Profanity-Filter
link for dataset https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
Linear SVM was used to build the profanity algorithm.
The process followed is explained in this link  https://towardsdatascience.com/building-a-better-profanity-detection-library-with-scikit-learn-3638b2f2c4c2

#Running the model

We will be using scikit-learn's linear svm module and other main modules include nltk and genism. 

Preprocess function is used to perform the preprocessing for the data.
The link provided will help understand how processing is done. https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925

Joblib.load function is used to load the vectoriser and model files. Specify the path where the files are located.


Predict function, if the predicted values returns 1 its considered profane and if it returns 0 its considered not profane.

predict_prob gives us the probabilities of both occurances.


To improve the model, a bigger and better dataset can be used. Also performing LSTM algorithm will improve the algorithm performance.

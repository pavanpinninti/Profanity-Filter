# Profanity-Filter
link for dataset https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
Linear SVM was used to build the profanity algorithm.
The process followed is explained in this link  https://towardsdatascience.com/building-a-better-profanity-detection-library-with-scikit-learn-3638b2f2c4c2

# Training the Model
There are many columns in the raw data. We only need the comment text columns and Target Value. Target value ranges from 0 to 1 . For our classification problem I choose 0.5 as seperation value. Any value lower than 0.5 is changed to 0 and value greater than 0.5 is changed to 1.

#taking required columns in different dataframe for easier access

x_train = dftrain['comment_text']

#target column has values from 0 to 1, for convinience we have converted >0.5 values to 1 and <0.5 values to 0

y_train = np.where(dftrain['target'] >= 0.5, 1, 0)

x_test = dftest['comment_text']

# Preprocessing the text

This involves the following:
Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
Words that have fewer than 3 characters are removed.
All stopwords are removed.
Words are lemmatized — words in third person are changed to first person and verbs in past and future tenses are changed into present.
Words are stemmed — words are reduced to their root form.
We use the NLTK and gensim libraries to perform the preprocessing
# sample code
def lemmatize_stemming(text):

    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
#Tokenize and lemmatize

  def preprocess(text):

    result=[]
    
    for token in gensim.utils.simple_preprocess(text) :
    
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
        
            result.append(lemmatize_stemming(token))
            
    return result
    
# Vectorsation - Bag of words
I used scikit-learn's CountVectorizer class, which basically turns any text string into a vector by counting how many times each given word appears. This is known as a Bag of Words (BOW) representation. 

I use the fit_transform() method, which does 2 things:

Fit: learns a vocabulary by looking at all words that appear in the dataset.

Transform: turns each text string in the dataset into its vector form.

# Training - Linear SVM
The model I decided to use was a Linear Support Vector Machine (SVM), which is implemented by scikit-learn's LinearSVC class.

The CalibratedClassifierCV in the code  exists as a wrapper to give me the predict_proba() method, which returns a probability for each class instead of just a classification.

# Train the model
model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)

cclf = CalibratedClassifierCV(base_estimator=model)

cclf.fit(X, y_train)

# Saving the Model
joblib.dump(vectorizer, 'vectorizer.joblib')

joblib.dump(cclf, 'model.joblib')
# Running the model

We will be using scikit-learn's linear svm module and other main modules include nltk and genism. 

Preprocess function is used to perform the preprocessing for the data.
The link provided will help understand how processing is done. https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925

Joblib.load function is used to load the vectoriser and model files. Specify the path where the files are located.


Predict function, if the predicted values returns 1 its considered profane and if it returns 0 its considered not profane.

predict_prob gives us the probabilities of both occurances.


To improve the model, a bigger and better dataset can be used. Also performing LSTM algorithm will improve the algorithm performance.

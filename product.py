import numpy as np
import pandas as pd
from flask import Flask,request,jsonify
import pickle

dataset=pd.read_csv('ProductReview.csv')
X=dataset.iloc[:,0].astype(str).values
y=dataset.iloc[:,-1].values


for i in range (y.shape[0]):
    if(y[i]<=3):
        y[i]=0
    else:
        y[i]=1
    
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for j in range(X.shape[0]):
    review=re.sub('[^a-zA-Z]',' ',X[j])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    allstopwords=stopwords.words('english')
    
    review=[ps.stem(word) for word in review if not word  in set(allstopwords)]
    review=' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train)

y_predict=classifier.predict(X_test)
print(np.concatenate((y_predict.reshape(len(y_predict),1),y_test.reshape(len(y_test),1)),1 ))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_predict)
print(cm)
accuracy_score(y_test, y_predict)



with open('model.pkl','wb') as mod_file:
    pickle.dump(classifier, mod_file)

with open('select.pkl','wb') as sel_file:
    pickle.dump(cv, sel_file)

app=Flask(__name__)

@app.route('/predict', methods=['POST'])

def predict():
    data=request.json
    review3=data.get('Review')
    review3=re.sub('[^a-zA-Z]',' ',review3)
    review3=review3.lower()
    review3=review3.split()
    review3=[ps.stem(word) for word in review3 if not word  in set(allstopwords)]
    review3=' '.join(review3)
    with open('model.pkl','rb') as mod_file:
        ansClassify=pickle.load(mod_file)
    with open('select.pkl','rb') as sel_file:
        ansVectorize=pickle.load(sel_file)
    review3=ansVectorize.transform([review3]).toarray()
    answer=ansClassify.predict(review3)
    return jsonify({'Prediction':int(answer[0])})
if __name__== '__main__' :
    app.run(debug=True)
    
         
    


 
    

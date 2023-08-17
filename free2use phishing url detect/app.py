#importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')
from feature import generate_data_set
from sklearn.neighbors import KNeighborsClassifier

#This is a code for building a K-Nearest Neighbors (KNN) classification model using scikit-learn library in Python, and deploying it with Flask. The dataset used for this model is "phishing.csv", which is read into a Pandas dataframe. The index column is dropped from the dataframe, and the features and target variables are separated into X and y respectively.
#Next, the KNN model is instantiated with one nearest neighbor, and then trained on the data using the fit method. Finally, the Flask web framework is imported, and an instance of the Flask application is created. However, the rest of the Flask code is missing, so it is not clear what the purpose of the application is or how it is being used with the KNN model.

data = pd.read_csv("phishing.csv")
#droping index column
data = data.drop(['Index'],axis = 1)
# Splitting the dataset into dependant and independant fetature

X = data.drop(["class"],axis =1)
y = data["class"]

# instantiate the model
knn = KNeighborsClassifier(n_neighbors=1)

# fit the model 
knn.fit(X,y)
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", xx= -1)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        url = request.form["url"]
        x = np.array(generate_data_set(url)).reshape(1,30) 
        y_pred =knn.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = knn.predict_proba(x)[0,0]
        y_pro_non_phishing = knn.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
        # else:
        #     pred = "It is {0:.2f} % unsafe to go ".format(y_pro_non_phishing*100)
        #     return render_template('index.html',x =y_pro_non_phishing,url=url )
    return render_template("index.html", xx =-1)


if __name__ == "__main__":
    app.run(debug=True)
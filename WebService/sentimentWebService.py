from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)
api = Api(app)
# create new model object
# load trained classifier
clf_path = 'kNN'
with open(clf_path, 'rb') as f:
    model = pickle.load(f)
# load trained vectorizer
vec_path = 'vectorizer.pk'
with open(vec_path, 'rb') as f:
    vectorizer = pickle.load(f)
	
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        # vectorize the user's query and make a prediction
        uq_vectorized = vectorizer.transform(np.array([user_query]))
        prediction = model.predict(uq_vectorized)
        pred_proba = model.predict_proba(uq_vectorized)
        # Output 'SAD' or 'HAPPY' along with the score
        if prediction == 'HAPPY':
            pred_text = 'HAPPY'
        else:
            pred_text = 'SAD'
            
        # round the predict proba value and set to new variable
        confidence = np.round(pred_proba[0], 3)
        # create JSON object
        prediction= pd.Series(prediction).to_json(orient='values')
        confidence= pd.Series(confidence).to_json(orient='values')

        output = {'prediction': pred_text, 'confidence': confidence}
        
        return output
		
		
api.add_resource(PredictSentiment, '/')
  
if __name__ == '__main__':
    app.run(debug=True)

# sample GET request will be like:
#url = 'http://127.0.0.1:5000/'
#params ={'query': 'غذاشون واقعاً عالی بود'}
#response = requests.get(url, params)
#response.json()

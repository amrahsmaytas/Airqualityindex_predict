import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle


app = Flask(__name__)
model = pickle.load(open('randomForestRegressor.pkl','rb'))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction[0])
    output = round(prediction[0], 2) #gives output
    if (output in range(0,51):
        return render_template('good_home.html', prediction_text="Air Quality Index is: {}".format(output) #give whole output+text
    else if (output in range(51,100):
        return render_template('moderate_home.html', prediction_text="Air Quality Index is: {}".format(output) #give whole output+text
    else if (output in range(101,150):
        return render_template('Unhealthy_for_Sensitive_Groups.html', prediction_text="Air Quality Index is: {}".format(output) #give whole output+text
    else if (output in range(151,200):
        return render_template('Unhealthy_result.html', prediction_text="Air Quality Index is: {}".format(output) #give whole output+text
    else:
        return render_template('Very_Unhealthy.html', prediction_text="Air Quality Index is: {}".format(output)#give whole output+text




@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)

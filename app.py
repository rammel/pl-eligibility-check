from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
import numpy as np
import json

clf_model = joblib.load('gbm_clf_model_loan_prediction.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


# @app.route('/api', methods=['POST'])
# def make_prediction():
#     data = request.get_json(force=True)
#     #convert our json to a numpy array
#     one_hot_data = input_to_one_hot(data)
#     predict_request = gbr.predict([one_hot_data])
#     output = [predict_request[0]]
#     print(data)
#     return jsonify(results=output)

def input_to_one_hot(data):
    # initialize the target vector with zero values
    enc_input = np.zeros(13)
    # set the numerical input as they are
    enc_input[0] = data['Age']
    enc_input[1] = data['Experience']
    enc_input[2] = data['Income']
    enc_input[3] = data['Family']
    enc_input[4] = data['CCAvg']
    enc_input[5] = data['Mortgage']
    enc_input[6] = data['SecuritiesAccount']
    enc_input[7] = data['CDAccount']
    enc_input[8] = data['Online']
    enc_input[9] = data['CreditCard']
    ##################### Education #########################
    # get the array of education categories
    marks = ['1', '2', '3']
    cols = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',  'Mortgage', 
            'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard', 
            'Education_1', 'Education_2', 'Education_3']

    # redefine the the user inout to match the column name
    redefinded_user_input = 'Education_'+str(data['Education'])
    # search for the index in columns name list 
    edu_column_index = cols.index(redefinded_user_input)
    #print(mark_column_index)
    # fullfill the found index with 1
    enc_input[edu_column_index] = 1
   
    return enc_input

@app.route('/api',methods=['POST'])
def get_delay():
    result=request.form
    Age               = result['Age']
    Experience        = result['Experience']
    Income            = result['Income']
    Family            = result['Family']
    CCAvg             = result['CCAvg']
    Mortgage          = result['Mortgage']
    SecuritiesAccount = result['SecuritiesAccount']
    CDAccount         = result['CDAccount']
    Online            = result['Online']
    CreditCard        = result['CreditCard']
    Education         = result['Education']
    
    user_input = {'Age': Age, 
                  'Experience': Experience, 
                  'Income': Income, 
                  'Family': Family, 
                  'CCAvg': CCAvg, 
                  'Mortgage': Mortgage, 
                  'SecuritiesAccount': SecuritiesAccount, 
                  'CDAccount': CDAccount,                    
                  'Online': Online, 
                  'CreditCard': CreditCard, 
                  'Education': Education, 
                 }

    print(user_input)
    input_vector = input_to_one_hot(user_input)
    loan_eligibility_prediction = clf_model.predict([input_vector])[0]
    if loan_eligibility_prediction == 0:
      status = 'No'
    else:
      status = 'Yes'
    return json.dumps({'LoanEligibility': status});
    # return render_template('result.html',prediction=price_pred)

if __name__ == '__main__':
    app.run(port=8080, debug=True)







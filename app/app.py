from cgi import test
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import uuid
from joblib import load


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    
    request_type_str = request.method
    
    if request_type_str == 'GET':
        return render_template('index.html', href='static/base_pic.svg')
    else:
        text = request.form['text']
        randomStr = uuid.uuid4().hex
        path = "static/" + randomStr + ".svg"
        model = load('model.joblib')
        np_arr = floats_string_to_np_array(text)
        
        make_picture('AgesAndHeights.pkl', model, np_arr, path)
        return render_template('index.html', href=path)



def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
    data = pd.read_pickle(training_data_filename)
    ages = data['Age']
    data = data[ages > 0]
    ages = data['Age']
    heights = data['Height']
    
    x_new = np.array(list(range(19))).reshape(19, 1)
    preds = model.predict(x_new)
    
    fix = px.scatter(x=ages, y=heights, title='Altura vs Edad de las Personas', labels={'x':'Edad', 'y':'Altura (pulgadas)'})
    fix.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name='Model'))
    
    new_preds = model.predict(new_inp_np_arr)
    fix.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), 
                             y=new_preds, 
                             name='New Outputs', 
                             mode='markers', 
                             marker=dict(color='purple', size=20, line=dict(color='purple', width=2))
                            )
                 )
    
    fix.write_image(output_file, width=800, engine='kaleido')


def isFloat(s):
    try:
        float(s)
        return True
    except:
        return False

def floats_string_to_np_array(floats_str):
    floats = np.array([float(x) for x in floats_str.split(',') if isFloat(x)])
    return floats.reshape(len(floats), 1)


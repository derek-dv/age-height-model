from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
app = Flask(__name__)

def make_preds(model, inputs, out_file):
	model = load(model)
	data = pd.read_pickle('AgesAndHeights.pkl')
	ages = data['Age']
	data = data[ages > 0]
	ages = data['Age']
	heights = data['Height']
	age = np.array(ages).reshape(len(ages), 1)

	dp = np.linspace(0, 18, 19).reshape(19, 1)
	pred = model.predict(dp)
	lens = len(inputs)
	inputs = np.array(inputs).reshape(len(inputs), 1)
	preds = model.predict(inputs)
	print('plotting')

	fig = px.scatter(x=ages, y=heights, title='Heights vs Ages', labels={'x': 'Ages(inches)',
                                                                    'y': 'Heights'})
	fig.add_trace(go.Scatter(x=dp.reshape(19), y=pred, mode='lines', name='Model'))
	fig.add_trace(go.Scatter(x=inputs.reshape(lens), y=preds, mode='markers', name='Output', line=dict(color='purple', width=2)))
	fig.write_image(out_file, width=800, engine='kaleido')
	print('done')
	fig.show()
	

def make_float(n):
	def is_float():
		try:
			float(n)
			return True
		except:
			return False

	if is_float(): return float(n)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
		if request.method == 'GET': 			
			return render_template('index.html', href='static/out.svg')

		else:
			ran = uuid.uuid4().hex
			out_file = f'static/pred-{ran}.svg'
			inp1 = request.form['text1']
			inp2 = request.form['text2']
			inp3 = request.form['text3']
			float_inps = [make_float(inp1), make_float(inp2), make_float(inp3)]
			make_preds('model.joblib', float_inps, out_file)
			return render_template('index.html', href=out_file, inps=float_inps)

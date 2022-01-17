import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import argparse
import configparser
from flask import Flask, request
import json
from tensorflow import keras

app = Flask(__name__)

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('info.init')

args.host = conf.get('server','host')
args.port = conf.get('server','port')
args.features = json.loads(conf.get('settings','feature'))
args.features.remove('gateway_id')

scale_cols = ['stack_temp', 'ambient_temp', 'meoh_concentration', 'sp_condition', 'stack_power']

#MinMax Normalization
scaler = MinMaxScaler()
def normalization(data):
    nor = scaler.fit(data.astype(float))
    nor = scaler.transform(data)
    scaled_df = pd.DataFrame(nor)
    scaled_df.columns = scale_cols
    return scaled_df

def inverse_value(value):
	_value = np.zeros(shape=(len(value), len(scale_cols)))
	_value[:,0] = value[:,0]
	inv_value = scaler.inverse_transform(_value)[:,0]
	return inv_value

def to_supervised(train, n_input, n_out=12):
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return np.array(X), np.array(y)


@app.route("/predict", methods=['POST'])
def predict():
    receive_data = pd.DataFrame(json.loads(request.get_data()))
    receive_data['time'] = pd.to_datetime(receive_data['time'], format='%Y-%m-%d %H:%M:%S')
    gid = receive_data["gateway_id"][0]
    print("gid: {}".format(gid))

    receive_feature = receive_data[['time', 'stack_power', 'stack_temp', 'ambient_temp', 'meoh_concentration']]

    # Feature 
    percent = .25
    receive_feature['sp_condition'] = np.where(receive_feature['stack_power'].values < receive_feature[['stack_power']].quantile(percent)[0], 0, 1)

    # data 
    receive_feature.index = pd.to_datetime(receive_feature['time'], format='%Y-%m-%d %H:%M:%S')
    minutes = '5T'
    receive_feature = receive_feature.resample(minutes).mean().reset_index()
    receive_feature = receive_feature.dropna().tail(36)
    receive_feature = receive_feature.drop('time', axis=1)
    print('{} minutes average'.format(minutes))

    receive_nor = normalization(receive_feature)

    timesteps = 12
    size = int((len(receive_nor) / timesteps) * 1) * timesteps
    receive_df = receive_nor[0:size]
    receive_train = np.array(np.split(receive_df, len(receive_df) / timesteps))
    x_receive, y_receive = to_supervised(receive_train, timesteps)

    args.model_path = 'model/{}/'.format(gid)
    model_list = os.listdir(args.model_path)
    args.model = model_list[0]
    print(args.model_path + args.model)
    
    model = keras.models.load_model(args.model_path + args.model)

    prediction = model.predict(x_receive)
    prediction1 = inverse_value(prediction)
    prediction2 = pd.DataFrame({"pred_value": prediction1})

    print('pd2', prediction2)
    predict = prediction2["pred_value"].to_json()

    print("AI Model Prediction Success")

    return predict

if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=False)
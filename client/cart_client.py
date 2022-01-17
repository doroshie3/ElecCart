import pandas as pd
import numpy as np
import requests
import json
import sys
import configparser
import psycopg2
import time
from datetime import datetime
import argparse
from apscheduler.schedulers.background import BackgroundScheduler
from psycopg2.extensions import register_adapter, AsIs
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)
import warnings

warnings.simplefilter("ignore")
pd.set_option('mode.chained_assignment', None)

parser = argparse.ArgumentParser()
args = parser.parse_args("")

conf = configparser.ConfigParser()
conf.read('/home/p2g/bellk/fuelcell/client/info.init')

dbname = conf.get('DB', 'dbname')
host = conf.get('DB', 'host')
user = conf.get('DB', 'user')
password = conf.get('DB', 'password')
port = conf.get('DB', 'port')
table = json.loads(conf.get('DB','table')) #train table
print("<DB Info>")
print("dbname:", dbname + "\nhost:", host + "\nport:", port)

# =======================================CONFIGURATION=======================================*
# args.host = conf.get('server','host')
args.host = conf.get('server','local')
# args.host = conf.get('server','container_host')
args.port = conf.get('server','port')
args.url = 'http://' + args.host + ':' + args.port + '/predict'
# args.url = 'http://125.131.88.57' + ':' + args.port + '/predict'

args.minutes = conf.getint('settings','seconds')
args.count = conf.getint('settings','count')
args.error = conf.getfloat('settings','error')
args.opmode = conf.get('condition','opmode') 
args.stack_power = conf.get('condition','stack_power')
args.features = json.loads(conf.get('settings','feature'))

golf = 'course2'
args.golf = conf.get('place', golf) 
args.gid = json.loads(conf.get('gateway_id', golf))


print("<Golf Info>")
print("golf_name:", args.golf)
print("gateway:", args.gid)

select_gw = sys.argv[1]
# select_gw = args.gid[0]
print("select gw:", select_gw)
# ============================================================================================*

con = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
cursor = con.cursor()

def collect(gid):
    gw_cond = "gateway_id = '{}' AND operation_mode_status = {} AND stack_power !={} ORDER BY time DESC LIMIT {}".format(gid, args.opmode, args.stack_power,  args.count)
    gw_db = "SELECT * FROM {} WHERE {}".format(table[0], gw_cond)
    cursor.execute(gw_db)
    gw = pd.DataFrame(cursor.fetchall())
    gw.columns = [desc[0] for desc in cursor.description]

    gw = gw[['time', 'gateway_id', 'stack_power', 'stack_temp', 'ambient_temp', 'meoh_concentration', 'stack_current', 'stack_voltage']]
    gw['time'] = pd.to_datetime(gw['time'], format='%Y-%m-%d %H:%M:%S')
    gw['stack_power'] = gw['stack_current'] * gw['stack_voltage']
    gateway = gw.drop_duplicates(['time', 'gateway_id'], keep='first').reset_index(drop=True)

    print('request date: {}'.format(datetime.now()))
    print('start date: {}'.format(gw['time'][2159]))
    print('end date: {}'.format(gw['time'][0]))
    return gw


def serving(gid):
    gw = collect(gid)

    if gw.empty:
        print("gateway is empty!!")
    else:
        gw_feature = gw[args.features]
        gw_json = gw_feature.to_json(date_format='iso')
        req = requests.post(args.url, gw_json, verify=False)

        pred_json = req.json()
        print(req.status_code, "Receive Prediction Result")

        pred_df = pd.DataFrame(pred_json, index=['predict']).T
        pred_df['predict'] = pred_df['predict'].astype(float)
        pred_df['predict'] = pred_df['predict'].round(2)
        for i in range(len(pred_df)):
            if pred_df['predict'][i] < 0:
                pred_df['predict'][i] = 0.0

        gw_predict = pd.merge(gw.reset_index(drop=True), pred_df.reset_index(drop=True), left_index=True, right_index=True)
        print(gw_predict)
        gw_predict['sp_diff'] = round(gw_predict['predict'] - gw_predict['stack_power'],2)

        gw_predict['flag'] = 0
        for i in range(len(gw_predict)):
            if ((gw_predict['sp_diff'][i] > 0) & (gw_predict['sp_diff'][i] > gw_predict['predict'][i] * args.error)):
                gw_predict['flag'][i] = 1 # anomaly
            elif gw_predict['sp_diff'][i] < 0:
                gw_predict['flag'][i] = 2 # stack_power > sp_pred
            else:
                gw_predict['flag'][i] = 0 # normal

        gw_predict['flag'] = gw_predict['flag'].astype(float)
        gw_predict['status'] = 0
        gw_predict['status'] = gw_predict['status'].astype(float)

        cnt = 0
        for i in range(len(gw_predict)):
            if gw_predict.flag[i] == 1:
                cnt = cnt + 1
        if cnt == args.count:
            gw_predict['status'][0] = 1


        gw_predict = gw_predict[['time', 'gateway_id', 'stack_temp', 'ambient_temp', 'meoh_concentration', 'stack_power', 'predict', 'sp_diff', 'flag', 'status']]

        insert_serving = "INSERT INTO ads(time, gateway_id, stack_temp, ambient_temp, meoh_concentration, stack_power, predict, sp_diff, flag, status)"\
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)" \
                        "ON CONFLICT ON CONSTRAINT pri_key DO UPDATE SET " \
                        "time=EXCLUDED.time, gateway_id = EXCLUDED.gateway_id, stack_temp = EXCLUDED.stack_temp, " \
                        "ambient_temp=EXCLUDED.ambient_temp, meoh_concentration = EXCLUDED.meoh_concentration, stack_power = EXCLUDED.stack_power, " \
                        "predict = EXCLUDED.predict, sp_diff = EXCLUDED.sp_diff, flag = EXCLUDED.flag, status = EXCLUDED.status"

        predict_data = []
        for i in range(len(gw_predict)):
            predict_data.append(gw_predict.loc[i].tolist())

        cursor.executemany(insert_serving, predict_data)
        con.commit()
        print("Insert DB Success")

if __name__ == "__main__":
    # for gw in args.gid:
    serving(select_gw)


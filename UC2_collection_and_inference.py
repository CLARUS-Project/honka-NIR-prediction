import numpy as np
import requests
import sklearn
import joblib
import datetime
import json



def query_data_from_cloud(position,start_ts,end_ts):
    header = {"Accept": "application/json"}
    #data_received = json.dumps({"key": '{"x-api-key": "C6WYqPZZPu3iliFggLuxz6O0Oc125JRU2HXpjPLF"}'})
    #data_received_dejason = json.loads(data_received)
    header_auth = {"x-api-key": "key"}
    try:

        r = requests.request("get",
                             f"https://cloud.honkajokioy.fi/api/plant/timeseries_raw/source/p600/measurement/{position}?start={start_ts}&end={end_ts}&limit=10&newest=true",
                             headers=header_auth)
        loaded_content = (json.loads(r.content))
        sensor_val = float(loaded_content["results"][0]["value"])
    except Exception as err:
        print(err)
        sensor_val = 0
    return sensor_val


# inference function, later this need to be replaced with REST query that inferences the data to model executor,
# but they should be quite the same in input and output
def inference_data_to_models(model_n,in_cate,sensor_arr,ash,moisture,protein,fat):
    rf_model = joblib.load(f"UC2_models/Sensor_configuration/{model_n}")
    material_dict = {'material1': 0,
                     'material2': 1
                     }
    try:
        mat_id = material_dict[in_cate]
    except:
        mat_id = 1.5
    #rf_2406_TIC1_PDMEAS_model = joblib.load("UC2_models/Sensor_configuration/rf_2406_TIC1_PDMEAS.sav")
    #rf_2501_PIC1_PDMEAS_model = joblib.load("UC2_models/Sensor_configuration/rf_2501_PIC1_PDMEAS.sav")
    #rf_2501_SIC1_PDMEAS_model = joblib.load("UC2_models/Sensor_configuration/rf_2501_SIC1_PDMEAS.sav")
    #rf_2502_PIC1_PDMEAS_model = joblib.load("UC2_models/Sensor_configuration/rf_2502_PIC1_PDMEAS.sav")
    #rf_2502_SIC1_PDMEAS_model = joblib.load("UC2_models/Sensor_configuration/rf_2307_PIC1_PDMEAS.sav")
    #'Category', '2303_M1', '2307_M1', '2501_M1', '2501_M4', '2502_M4', '2413_M3', '2650_XT1', '2650_XT2', '2650_XT3', '2650_XT4'
    rf_data = rf_model.predict(np.array([mat_id,
                                         sensor_arr["2303_M1"],
                                         sensor_arr["2307_M1"],
                                         sensor_arr["2501_M1"],
                                         sensor_arr["2501_M4"],
                                         sensor_arr["2502_M4"],
                                         sensor_arr["2413_M3"],
                                         ash,
                                         moisture,
                                         protein,
                                         fat,
                                         ]).reshape(-1,11))
    print(rf_data[0])
    return rf_data[0]


def querry_all_input_sensor_data():

    current_ts = datetime.datetime.now()
    end_ts = (current_ts + datetime.timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    start_ts = (current_ts - datetime.timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    non_control_array = {}

    for non_controlled_pos in ['2303_M1', '2307_M1', '2501_M1', '2501_M4', '2502_M4', '2413_M3']:
        non_control_array[non_controlled_pos]=query_data_from_cloud(non_controlled_pos,start_ts,end_ts)
    print(non_control_array)
    return non_control_array
def collect_inferenced_data(mat_cate,non_control_input,ash_val,mois_val,prot_val,fat_val):
    model_sensor_dict = {"rf_2307_PIC1_PDMEAS.sav": "2307_PIC1_PDMEAS",
                         "rf_2406_TIC1_PDMEAS.sav": "2406_TIC1_PDMEAS",
                         "rf_2501_PIC1_PDMEAS.sav": "2501_PIC1_PDMEAS",
                         "rf_2501_SIC1_PDMEAS.sav": "2501_SIC1_PDMEAS",
                         "rf_2502_PIC1_PDMEAS.sav": "2502_PIC1_PDMEAS",
                         "rf_2502_SIC1_PDMEAS.sav": "2502_SIC1_PDMEAS"}
    pred_sensor_dict = {}
    for model_name in ["rf_2307_PIC1_PDMEAS.sav","rf_2406_TIC1_PDMEAS.sav","rf_2501_PIC1_PDMEAS.sav","rf_2501_SIC1_PDMEAS.sav","rf_2502_PIC1_PDMEAS.sav","rf_2502_SIC1_PDMEAS.sav"]:
        pred_sensor_dict[model_sensor_dict[model_name]] = inference_data_to_models(model_name,mat_cate,non_control_input,ash_val,mois_val,prot_val,fat_val)
    return pred_sensor_dict

sensor_input_data = querry_all_input_sensor_data()
#show sensor_input_data to users for modification if they want to change sensors or planning ahead?
result_dict = collect_inferenced_data(mat_cate='material1',
                                      non_control_input=sensor_input_data,
                                      ash_val=14,
                                      mois_val=5,
                                      prot_val=70,
                                      fat_val=11)
print(result_dict)
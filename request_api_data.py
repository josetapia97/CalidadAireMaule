from datetime import datetime, timedelta
import json
import requests
import os

HISTORY_API = 'http://api.openweathermap.org/data/2.5/air_pollution'
API_KEY='feacac8e99e71b7fb12a7b0e2b57df77'

def get_data(appid, lat, lon):
    now = datetime.utcnow() 
    params = {'lat': lat, 'lon': lon, 'appid': appid}
    reply = requests.get(HISTORY_API, params=params).json()
    return reply

def request_api_columns(grid_file):
    with open(grid_file) as file:
        grid_data=json.load(file)
    dt=datetime.now()
    dt_unix=int(dt.timestamp())
    for feature in grid_data['features']:
        coords=feature['geometry']['coordinates'][0]
        coord_hash=hash(tuple(coords))
        api_data=get_data(API_KEY,coords[1],coords[0])
        file_path='{0}{1}.json'.format(coord_hash,dt_unix)
        data_path=os.path.join(os.getcwd(),'data',file_path)

        output = dict()
        output["coords"] = coord_hash #hash de la tupla
        output["dt_u"] = dt_unix #dt de la consulta a la api
        output["lat"] = api_data["coord"]["lat"]
        output["lon"] = api_data["coord"]["lon"]
        output["aqi"] = api_data["list"][0]["main"]["aqi"]
        output["co"] = api_data["list"][0]["components"]["co"]
        output["no"] = api_data["list"][0]["components"]["no"]
        output["no2"] = api_data["list"][0]["components"]["no2"]
        output["o3"] = api_data["list"][0]["components"]["o3"]
        output["so2"] = api_data["list"][0]["components"]["so2"]
        output["pm2_5"] = api_data["list"][0]["components"]["pm2_5"]
        output["pm10"] = api_data["list"][0]["components"]["pm10"]
        output["nh3"] = api_data["list"][0]["components"]["nh3"]
        output["dt_m"] = api_data["list"][0]["dt"] #dtm es el datetime de la muestra en la pagin$

        with open(data_path,'w') as write_file:
            json.dump(output,write_file)

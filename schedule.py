import time
import threading
from request_api_data import *
from datetime import datetime, timedelta

def task():
    archivo = "./grid_data/grilla_maule_gruesa.geojson"
    start = datetime.now()
    request_api_columns(archivo)
    finish = datetime.now()
    print(finish - start)

def schedule():
    while 1:
        task()
        time.sleep(60*10)

# makes our logic non blocking
thread = threading.Thread(target=schedule)
thread.start()
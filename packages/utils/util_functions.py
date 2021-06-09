import datetime

def map_list(func, arr):
    return list(map(func, arr))

def get_artifacts_path(filename):
    date = datetime.today().strftime('%Y-%m-%d')
    dest = "data/artifacts/{}/{}".format( date,filename)
    return dest 

def get_today_date_formatted():
    date = datetime.today().strftime('%Y-%m-%d')
    return date
import pandas as pd
import pickle
from os import makedirs
from os.path import split 
from datetime import datetime

def load_pkl_from_path(path):
    path, filename = split(path)
    pkl = load_pkl(filename, '{}/'.format(path))
    return pkl

def load_pkl(filename, folder=''):
    # with open(folder + filename + '.pkl', 'rb') as pkl:
    #     return pickle.load(pkl, encoding="utf8")
    with open(folder + filename + '.pkl', 'rb') as pkl:
        return pickle.load(pkl) 

def inspect_pkl(filename, folder='', fn=None):
    """Inspect the contents of a pkl. 

    Params:
        fn {pkl -> None}: Function used to inspect the contents of a pkl
    """
    with open(folder + filename + '.pkl', 'rb') as pkl:
        pkl = (pickle.load(pkl))
        if fn:
            fn(pkl)
        else:
            print(pkl)

def save_pkl(filename, obj, folder=''):
    """Save pkl file 
    
    Arguments:
        filename {str} 
        obj {pkl object} -- the pickle object to save
    
    Keyword Arguments:
        folder {str} -- The folder to save the file in; it must be appended with '/' (default: {''})
    
    Returns:
        None
    """
    def write_to_pkl():
        with open(folder + filename + '.pkl', 'wb') as pkl:
            return pickle.dump(obj, pkl)
    if folder != '':
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    write_to_pkl()
    
def savefig(plt, folder, fname, use_pdf=False):
    if folder != '':
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    if use_pdf:
        plt.savefig(folder+fname+'.pdf', dpi=1200)
    else:
        plt.savefig(folder+fname)

def get_pkl(obj):
    return pickle.dumps(obj)

def store_results_dynamic(result, filename, root_folder):
    date = datetime.today().strftime('%Y-%m-%d')
    folder = "{}/{}/".format(root_folder, date)
    save_pkl(filename, result, folder)

def store_pic_dynamic(plt, fname, root_folder, use_pdf=False):
    date = datetime.today().strftime('%Y-%m-%d')
    folder = "{}/{}/".format(root_folder, date)
    savefig(plt, folder, fname, use_pdf)

def store_csv_dynamic(frame, filename, root_folder='results'):
    date = datetime.today().strftime('%Y-%m-%d')
    folder = "{}/{}/".format(root_folder, date)
    if folder != '':
        try:
            makedirs(folder)
        except FileExistsError:
            pass
    frame.to_csv(f'{root_folder}/{date}/{filename}.csv')
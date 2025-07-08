import copy
import pickle 

def select_params(d1, d2, depth=5000):
    for k in d2 :
        if k not in d1 :
            d1[k] = copy.copy(d2[k])
        elif type(d1[k]) is dict and type(d2[k]) is dict and depth>0 :
            select_params(d1[k], d2[k], depth=depth-1)
    return d1

def getp(key, d_list, default="error"):
    if type(d_list) is not list : d_list = [d_list] 
    for d in d_list :
        if key in d :
            return d[key]
    if default == "error" :
        print("error : the key ", key, " not found in d_list and no default value has been specified")
        raise
    return default

def print_dict(d, title=None, prefix=""):
    if title is not None :
        print("---------------------------", title)
    if type(d) is list :
        print(prefix+"--")
        for d_i in d :
            print_dict(d_i, None, prefix+"   ")
            print(prefix+"--")
    elif type(d) is dict :
        for k in d :
            type_k = type(d[k])
            if type_k is dict :
                print(prefix+k,":")
                print_dict(d[k], None, prefix+"   ")
            elif type_k in [str, int, float, bool]:
                print(prefix+k,":", d[k])
            else :
                print(prefix+k,":", type_k)
    if title is not None :
        print("---------------------------END", title)
                
def save_dict(d, savepath, force=False):
    if os.path.exists(savepath) and not force :
        print(savepath, "already exists, use force=True to overwrite")
    else :
        with open(savepath, 'wb') as f:
            pickle.dump(d, f)
        
def read_dict(d, savepath):
    with open(savepath, 'rb') as f:
        return pickle.load(f)
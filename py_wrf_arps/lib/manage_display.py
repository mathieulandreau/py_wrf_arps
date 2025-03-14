import numpy as np

def get_legend(legend, units, latex=True):
    string = legend
    if latex is None :
        return string
    if latex:
        string = "$"+string+"$"
        string = string.replace(' ', '\ ')
        string = string.replace('_AVG', '_{AVG}')
        string = string.replace('<', '\\langle ')
        string = string.replace('>', '\\rangle ')
    return string + " " + get_units(units, latex=latex)

def get_units(units, latex=True):
    if units in ["", " ", None] :
        return ""
    if latex :
        string = "("
        exponent = False
        for c in units :
            if c == "-" or c.isnumeric() :
                if exponent :
                    string += c
                else :
                    string += "$^{" + c
                    exponent = True
            else :
                if exponent :
                    exponent = False
                    string += c + "}$"
                else :
                    string += c
        
        if exponent :
            string += "}$"
        string += ")"
        return string
    else :
        return "("+units+")"

def display_var(var, name, pref = ""):
    """
    Description
        Display a variable or a list
    Parameters
        var : any variable
        name : str : name of the variable or txt to print
    Optional
        pref : str : prefix printed before the name
    Author(s)
        Mathieu LANDREAU
    """
    ncarac = 12
    if(isinstance(var, list)) : 
        nmax = 1
        n = len(var)
        if(nmax == 1 ):
            print(pref + name.ljust(ncarac) + " :", var[0], '(+' + str(n-nmax) + ')')
        else : 
            print(pref + name.ljust(ncarac) + " :")
            pref = pref + "  "
            for i in range(min(n, nmax)):
                print(pref, var[i])
            if(n > nmax):
                print(pref + '(+'+str(n-nmax)+')')
    else :
        print(pref + name.ljust(ncarac) + " :", var)
        
def print_arr(arr, decimals=5, nlen=7):
    """
    Description
        print an array
    Parameters
        arr : an array
    Optional
        decimals : int : see np.round
        nlen : int : number of character to print
    Author(s)
        Mathieu LANDREAU
    """
    n = len(arr)
    count = 0
    string = ""
    for i in range(n):
        string += str(np.round(arr[i], decimals)).ljust(nlen)+", "
        count += 1
        if count == 10 :
            print(string)
            count = 0
            string = "                                       "
    print(string)
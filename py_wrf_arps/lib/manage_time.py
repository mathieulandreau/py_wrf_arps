#!/usr/bin/env python3
import numpy as np
import datetime
import pandas as pd
import astral.sun


def get_date_list(date_list_in, slice_list=None, itime=None, max_time_correction=0):
    if slice_list is not None:
        return date_list_in[slice_list]
    elif itime is not None :
        return date_list_in[get_time_slice(itime, date_list_in, max_time_correction)]
    else :
        return date_list_in

    
def get_time_slice(date, date_list, max_time_correction = 0):
    """
    return :
    a slice or a list of indices
        that can be applied on date_list to get the desired list of date (sublist of date_list)
    
            itime : date to construct time_slice thanks to the method manage_time.get_time_slice, can be of type :
    date can be 
        datetime.datetime : a single date 
        np.datetime64 : a single date
        int : a single date, index of the file in self.date_list
        list : list of any type listed before
        tuple : (first_date, last_date, timestep), first_date and last_date can be any type liste before, timestep can be anything accepted by manage_time.to_timedelta(). timestep is optional.
        str : "ALL_TIMES" or anything accepted by manage_time.to_datetime (single date)
        None : first file
    """
    date_list = to_datetime(date_list)
    type_date = type(date)
    if date is None or (type_date is str and date == "") :
        return 0
    elif type_date in [list, np.ndarray] :
        temp = []
        for date_i in date :
            temp.append(get_time_slice(date_i, date_list, max_time_correction))
        return temp
    elif type_date is datetime.datetime :
        if not date in date_list :
            datetemp, diff, index = nearest_date(date, date_list, return_diff = True, return_index = True)
            if diff > max_time_correction :
                print(diff, max_time_correction)
                print("warning : converting ", date, " to the nearest known date : ", datetemp)
        else :
            index = int(np.squeeze(np.argwhere(np.array(date_list) == date)))
        return index
    elif type_date is np.datetime64 :
        return get_time_slice(to_datetime(date), date_list, max_time_correction)
    elif type_date is str :
        if date.upper() in ["ALL", "ALL_TIMES"] :
            return slice(len(date_list))
        else :
            try : 
                datetemp = to_datetime(date)
            except :
                print("error in get_time_slice : cannot get time slice from this string : ", date)
                raise
            return get_time_slice(datetemp, date_list, max_time_correction)
    elif type_date in [int, np.int64] :
        return int(date)
    elif type_date is slice :
        return date
    elif type_date is tuple :
        if type(date[0]) is int :
            return list(date)
        elif len(date) == 1 :
            start_index = 0
            end_date = to_datetime64(date[0])
            end_index = np.max(np.argwhere(date_list <= end_date+np.timedelta64(1, 'm')))
            step = 1
        else :
            start_date = to_datetime64(date[0])
            end_date = to_datetime64(date[1])
            start_index = np.min(np.argwhere(date_list >= start_date-np.timedelta64(1, 'm')))
            end_index = np.max(np.argwhere(date_list <= end_date+np.timedelta64(1, 'm')))
            if len(date) > 2 :
                step_temp = date[2]
                if type(step_temp) is int :
                    step = step_temp
                elif len(date_list) > 1 :
                    step_temp = to_timedelta(step_temp)
                    step = int(round(step_temp/(date_list[1] - date_list[0])))
                else :
                    step = 1
            else :
                step = 1
            step = max(1, step)
        return slice(start_index, end_index+1, step)
    else :
        print("error in get_time_slice : cannot get slice from : ", date, ", type = ", type_date)
        raise

        
def date_to_str(date, fmt="%Y-%m-%d_%H:%M:%S"):
    if fmt == "UTC": fmt = "%H%M"
    if fmt == "video": fmt = "%B %d, %H:%M UTC"
    date = to_datetime(date)
    if type(date) in [list, np.array, np.ndarray] :
        temp = []
        for date_i in date :
            temp.append(date_i.strftime(fmt))
        return temp
    else :
        return date.strftime(fmt)

def timedelta_to_str(delta, fmt="h", fmt_in=None):
    if type(delta) in [list, np.array, np.ndarray] :
        temp = []
        for delta_i in delta :
            temp.append(timedelta_to_str(delta_i, fmt))
        return temp
    else :
        delta = to_timedelta(delta, fmt=fmt_in)
        seconds = delta.seconds
        if fmt == "d" :
            return str(round(seconds/(3600*24),1))
        elif fmt == "h" :
            return str(round(seconds/3600,1))
        elif fmt == "m" :
            return str(seconds/60)
        elif fmt == "s" :
            return str(seconds)
        else :
            out_str = ""
            if "s" in fmt :
                seconds_supp = seconds%60
                out_str = str(seconds_supp) + "s"
            if "m" in fmt :
                minutes = seconds//60
                if "h" in fmt :
                    minutes_supp = minutes%60
                    out_str = str(minutes_supp) + "m" + out_str
                else :
                    out_str = str(minutes) + "m" + out_str
            if "h" in fmt :
                if "m" in fmt :
                    hours = seconds//3600
                else :
                    hours = seconds/3600
                if "d" in fmt :
                    hours_supp = hours%24
                    days = hours//24
                    out_str = str(days) + "d" + str(hours) + "h" + out_str
                else :
                    out_str = str(hours) + "h" + out_str
            return out_str
            
        
def print_timedelta64(delta, number=10, fmt=None):
    string = "("
    
    if type(delta) is not np.timedelta64 :
        delta = to_timedelta64(delta, fmt=fmt)
    if delta > np.array(0).astype('timedelta64[us]') :
        fac = 1
    else :
        fac = -1
        string += "-"
    delta = fac*delta
    if number > 0:
        days = delta.astype('timedelta64[D]').astype(int)
        if days > 0:
            string += str(days)+"d "
            number = number - 1
        if number > 0 :
            hours = delta.astype('timedelta64[h]').astype(int) - days*24
            if hours > 0 :
                string += str(hours)+"h "
                number = number - 1
            if number > 0 :
                minutes = delta.astype('timedelta64[m]').astype(int) - (days*24 + hours)*60
                if minutes > 0 :
                    string += str(minutes)+"m "
                    number = number - 1
                if number > 0 :
                    seconds = delta.astype('timedelta64[s]').astype(int)- ((days*24 + hours)*60 + minutes)*60
                    if seconds > 0 and number > 0:
                        string += str(seconds)+"s "
                        number = number - 1
    string += ")"
    return string

def timedelta_to_seconds(delta, fmt=None) :
    type_delta = type(delta)
    if type_delta is datetime.timedelta :
        return delta.total_seconds()
    elif type_delta in [list, np.array, np.ndarray] :
        temp = []
        for delta_i in delta :
            temp.append(timedelta_to_seconds(delta_i))
        return np.array(temp)
    else :
        return timedelta_to_seconds(to_timedelta(delta, fmt=fmt))

def timedelta_to_milliseconds(delta, fmt=None) :
    type_delta = type(delta)
    if type_delta is datetime.timedelta :
        return delta.seconds + 24*60*60*delta.days
    elif type_delta in [list, np.array, np.ndarray] :
        temp = []
        for delta_i in delta :
            temp.append(timedelta_to_seconds(delta_i))
        return np.array(temp)
    else :
        return timedelta_to_seconds(to_timedelta(delta, fmt=fmt))
        

def to_timedelta(delta, fmt=None) :
    type_delta = type(delta)
    if type_delta is datetime.timedelta :
        return delta
    elif type_delta is str :
        #"1d6h35m15s" "36h", ...
        delta_temp = delta
        if "d" in delta_temp :
            i = delta_temp.index("d")
            days = int(delta_temp[:i])
            delta_temp = delta_temp[i+1:]
        else :
            days = 0
        if "h" in delta_temp :
            i = delta_temp.index("h")
            hours = int(delta_temp[:i])
            delta_temp = delta_temp[i+1:]
        else :
            hours = 0
        if "m" in delta_temp :
            i = delta_temp.index("m")
            minutes = int(delta_temp[:i])
            delta_temp = delta_temp[i+1:]
        else :
            minutes = 0
        if "s" in delta_temp :
            i = delta_temp.index("s")
            seconds = int(delta_temp[:i])
            delta_temp = delta_temp[i+1:]
        else :
            seconds = 0    
        return datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    elif type_delta is np.timedelta64 :
        return datetime.timedelta(microseconds=int(delta.astype('timedelta64[ms]').astype("int")))
    elif type_delta in [int, float, np.int64, np.float64] :
        if fmt is None :
            print("error in manage_time.to_timedelta : with type : ", type_delta," a format is necessary ('h', 's', ...) , cannot convert : ", delta)
            raise
        elif fmt.lower() == "d" :
            return datetime.timedelta(days=delta)
        elif fmt.lower() == "h" :
            return datetime.timedelta(hours=delta)
        elif fmt.lower() == "m" :
            return datetime.timedelta(minutes=delta)
        elif fmt.lower() == "s" :
            return datetime.timedelta(seconds=delta)
        else :
            print("error in manage_time.to_timedelta : with type : ", type_delta," unknown format : ", fmt)
            raise 
    elif type_delta in [list, np.array, np.ndarray] :
        temp = []
        for delta_i in delta :
            temp.append(to_timedelta(delta_i, fmt=fmt))
        return temp
    else :
        print("error in manage_time.to_timedelta : unknown type : ", type_delta,", cannot convert : ", delta)
        raise

def to_datetime64(date, fmt=None):
    """
    Convert a date to the np.datetime_64 object
    
    date can be :
        datetime object
        np.datetime64 object : return itself
        string (fmt must be defined)
        list of any type above : return a list
        np.ndarray of datetime64 : return a list
    
    fmt : string (optional)
        must be defined if type(date) is string
        example : to_datetime("2020-05-17T23:15:00.000000", "%Y-%m-%dT%H:%M:%S.%f")
    """
    if type(date) is np.datetime64 :
        return date
    elif type(date) in [np.array, np.ndarray, list, pd.core.indexes.datetimes.DatetimeIndex]:
        temp = []
        for date_i in date :
            temp.append(to_datetime64(date_i, fmt))
        return np.array(temp).astype('datetime64')
    elif type(date) in [datetime.datetime, pd._libs.tslibs.timestamps.Timestamp]:
        return np.datetime64(date)
    elif type(date) is datetime.date :
        return to_datetime64(date.isoformat())
    elif type(date) is str :
        if fmt is None :
            fmt = guess_date_format(date)
        return  np.datetime64(to_datetime(date, fmt)) 
    else :
        print("error : unknown type ", type(date), " to convert to datetime64, date = ", date)
        raise

def to_day(date, fmt=None):
    """
    Description
        Convert a date to the nearest rounded day
    Parameters
        date : can be anything readable by to_datetime
    Optional
        fmt : string : see to_datetime
    Return
        same as to_datetime but at midnight for every date
    """
    if type(date) in [list, np.ndarray, np.array] :
        temp = []
        for date_i in date :
            temp.append(to_day(date_i, fmt))
        return temp
    else :
        return to_datetime(to_datetime(date).date())   
    
def to_hour(date, fmt=None):
    """
    Description
        Convert a date to hour
    Parameters
        date : can be anything readable by to_datetime
    Optional
        fmt : string : see to_datetime
    Return
        same as to_datetime but at midnight for every date
    """
    if type(date) in [list, np.ndarray, np.array] :
        temp = []
        for date_i in date :
            temp.append(to_hour(date_i, fmt))
        return temp
    else :
        return to_datetime(date).hour + to_datetime(date).minute/60 + to_datetime(date).second/3600

def to_datetime(date, fmt=None):
    """
    Convert a date to the datetime.datetime object
    
    date can be :
        datetime object : return itself
        np.datetime64 object
        string (fmt must be defined)
        list of any type above : return a list
        np.ndarray of datetime64 : return a list
    
    fmt : string (optional)
        must be defined if type(date) is string
        example : to_datetime("2020-05-17T23:15:00.000000", "%Y-%m-%dT%H:%M:%S.%f")
    """
    if type(date) in [list, np.ndarray, np.array] :
        temp = []
        for date_i in date :
            temp.append(to_datetime(date_i, fmt))
        return temp
    elif type(date) is datetime.date :
        return to_datetime(date.isoformat())
    elif type(date) is datetime.datetime :
        return date
    elif type(date) is np.datetime64 :
        temp = str(date.astype("datetime64[ms]"))
        return datetime.datetime.strptime(temp,"%Y-%m-%dT%H:%M:%S.%f")
    elif type(date) in [pd._libs.tslibs.timestamps.Timestamp, pd.core.indexes.datetimes.DatetimeIndex]:
        return to_datetime(to_datetime64(date))
    elif type(date) in [str, np.str_] :
        if fmt is None :
            fmt = guess_date_format(date)
        return datetime.datetime.strptime(date, fmt)
    else :
        print("error : unknown type ", type(date), " to convert to datetime : ", date)
        raise

def guess_date_format(date):
    """
    guess the format of a string date
    
    date : str, can be :
        2020.05.17
        2020.05.17.13
        2020.05.17.13.59
        2020.05.17.13.59.30
        2020.05.17.13.59.30.001025
        with any other character to replace the dots 
        for example : 2020-05-17T13:59:30.001025 can works too
    """
    if type(date) is not str :
        print("error : cannot guess the date format from :", date, "because it is not a string")
        raise
    if not( date[0:4].isnumeric() and date[5:7].isnumeric() and date[8-10].isnumeric() ):
        print("error cannot deduce a date format from : ", date)
        raise
    n = len(date)   
    if n == 10 :
        return "%Y" + date[4] + "%m" + date[7] + "%d"
    else :
        fmt = "%Y-%m-%d" + date[10]
        if n == 13 :
            return fmt + "%H"
        elif n == 16 :
            return fmt + "%H" + date[13] + "%M"
        elif n == 19 :
            return fmt + "%H" + date[13] + "%M" + date[16] + "%S"
        elif n == 26 :
            return fmt + "%H" + date[13] + "%M" + date[16] + "%S" + date[19] + "%f"
        else :
            print("error : cannot guess the date format from the string : ", date)
            raise
        
def nearest_date(date_in, list_of_date, return_diff = False, return_index=True):
    diff = abs(to_datetime64(date_in) - to_datetime64(list_of_date) )
    it = np.argmin(diff)
    result = (list_of_date[it],)
    if return_diff :
        result += (diff[it],)
    if return_index :
        result += (it,)
    return result

def is_nighttime(date, locInfo) :
    date = to_datetime(date)
    if type(date) is datetime.datetime :
        if date.tzinfo is None :
            date = date.replace(tzinfo=datetime.timezone.utc) 
        s = astral.sun.sun(locInfo.observer, date=date)
        if s["sunrise"] < s["sunset"] :
            return date < s["sunrise"] or date > s["sunset"]
        else :
            return date < s["sunrise"] and date > s["sunset"]
    elif type(date) in [list, np.ndarray, np.array] :
        temp = []
        for date_i in date :
            temp.append(is_nighttime(date_i, locInfo))
        return temp
    
def to_date_list(date1, date2, delta, fmt=None) :
    """
    Description
        Generate a list of date from date1 to date2 (included) with a step of delta
    Parameters
        date1, date2 : any type that can be input of to_datetime
        delta, fmt : any type that can be input of to_timedelta
    Returns 
        list of datetime.datetime objects 
    """
    date1 = to_datetime(date1)
    date2 = to_datetime(date2)
    delta = to_timedelta(delta, fmt)
    delta_tot = date2 - date1
    nt = delta_tot//delta + 1
    delta_list = delta*np.arange(nt)
    return date1 + delta_list

def is_regular(date_list):
    """
    Description
        find if the date_list has a constant delta
    Parameters
        date_list: a list of any type that can be input of to_datetime
    Returns 
        (boolean): True is delta is constant or len(list) in [0, 1]
    """
    date_list = to_datetime(date_list) #convert type of elements to datetime.datetime if not already
    if len(date_list) < 2 : return True
    diff = np.diff(date_list)
    return np.all(diff == diff[0])

def find_missing_date(date_list, delta=None, fmt=None, date1=None, date2=None) :
    """
    Description
        find the missing dates in a date_list that should contains every date between date1 and date2 with a constant delta
    Parameters
        date_list : a list of any type that can be input of to_datetime
    Optional
        date1, date2 : any type that can be input of to_datetime. If absent, choosing the min and the max dates of date_list (any date missing before the min and after the max won't be found)
        delta, fmt : any type that can be input of to_timedelta. If absent, choosing the minimal delta between two consecutive dates in date_list
    Returns 
        complete_date_list : list of datetime.datetime : the expected complete_date_list
        check : list of booleans, same length as complete_date_list : True if the expected dates are present in date_list.
    """
    date_list = to_datetime(date_list) #convert type of elements to datetime.datetime if not already
    date_list.sort() #sort in ascending order
    date1 = date_list[0] if date1 is None else to_datetime(date1)
    date2 = date_list[-1] if date2 is None else to_datetime(date2)
    if delta is None :
        delta = np.min(np.diff(np.unique(np.array(date_list))))
    complete_date_list = to_date_list(date1, date2, delta, fmt)
    check = [e in date_list for e in complete_date_list]
    if np.all(np.array(check)) :
        print("no missing date")
    else :
        dc = np.diff(np.array(check).astype("int"))
        print("the missing date(s) are : ")
        for i, d in enumerate(dc) :
            if d == -1 :
                if i < len(dc)-1 and dc[i+1] == 0 :
                    print("from", complete_date_list[i+1], end='')
                else :
                    print(complete_date_list[i+1])
            if d == 1 :
                if i > 0 and dc[i-1] == 0 :
                    print(" to", complete_date_list[i])
                elif i == 0 : 
                    print(complete_date_list[i])
    return complete_date_list, check
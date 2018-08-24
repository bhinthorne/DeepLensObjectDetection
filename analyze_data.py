'''
Import Statements
'''

#%%time

import sagemaker
import boto3
import pandas as pd
import ast
import datetime
from sagemaker import get_execution_role
import matplotlib.pyplot as plt
role = get_execution_role()


# In[4]:
'''
Downloading all the data from S3 and putting it into a dataframe
'''

##Downloads Detection Data from S3 Bucket and Puts it into a dataframe
def download_json_data():
    bucket_name = 'YOUR-BUCKET-NAME'
    keys = []
    times = []
    confs = []
    objects = []
    devices = []
    indecies = []
    client = boto3.client('s3')
    s3objects = client.list_objects_v2(Bucket=bucket_name)
    for object in s3objects['Contents']:
        # only want the objects that are the json records
        if object['Key'].startswith('json/record'):
            obj = client.get_object(Bucket=bucket_name, Key=object['Key'])
            ##Retrieve the body and time of the json files as strings
            data = obj['Body'].read().decode('utf-8')
            #Cast the data to a dictionary
            data_dict = ast.literal_eval(data)
            # A list of detections from the json file
            detections = data_dict["objects"]
            for i in range(len(detections)):
                times.append(data_dict["timestamp"])
                devices.append(data_dict["device_id"])
                single_obj = detections[i]
                objects.append(single_obj["object"])
                indecies.append(single_obj["index"])
                confs.append(single_obj["confidence"])
            
    ##Create a data frame
    df = pd.DataFrame()
    df['time'] = times
    df['device'] = devices
    df['object'] = objects
    df['index'] = indecies
    df['confidence'] = confs
    return df

detections = download_json_data()
print(detections)

'''
Sorting the data in various ways: by device, by day, by hour
TODO: Condense code and methods, lots of repeated code
'''

##A method returning a dataframe containing only the detections for a specified device
def sort_by_device(detections,device):
    times = []
    confs = []
    objects = []
    devices = []
    indecies = []
    for i in range(len(detections)):
        if detections.iloc[i]['device'] == device:
            times.append(detections.iloc[i]['time'])
            devices.append(detections.iloc[i]['device'])
            objects.append(detections.iloc[i]['object'])
            indecies.append(detections.iloc[i]['index'])
            confs.append(detections.iloc[i]['confidence'])
    df = pd.DataFrame()
    df['time'] = times
    df['device'] = devices
    df['object'] = objects
    df['index'] = indecies
    df['confidence'] = confs
    return df


##A method returning a dataframe containing only the detections for a specified day
def sort_by_day(detections,day):
    times = []
    confs = []
    objects = []
    devices = []
    indecies = []
    for i in range(len(detections)):
        if detections.iloc[i]['time'][:8] == day:
            times.append(detections.iloc[i]['time'])
            devices.append(detections.iloc[i]['device'])
            objects.append(detections.iloc[i]['object'])
            indecies.append(detections.iloc[i]['index'])
            confs.append(detections.iloc[i]['confidence'])
    df = pd.DataFrame()
    df['time'] = times
    df['device'] = devices
    df['object'] = objects
    df['index'] = indecies
    df['confidence'] = confs
    return df


## helper method, returns list of days from the total detections
def get_days(detections):
    days = []
    first_day = True
    for time in detections['time']:
        day = time[:8]
        if first_day:
            days.append(day)
        elif search_days(days,day):
            days.append(day)
        first_day = False
    return days

## helper method, returns list of hours for a certain day
def get_hours(detections):
    hours = []
    first_hour = True
    for time in detections['time']:
        hour = int(time[9:11])
        if first_hour:
            hours.append(hour)
        elif search_days(hours,hour):
            hours.append(hour)
        first_hour= False
    range_hours = []
    for i in range(min(hours),max(hours)+1):
        range_hours.append(str(i))
    return range_hours  

## helper method for get_hours and get_days, checking if a value is already in the list
def search_days(days, time):
    new_day = True
    for day in days:
        if time == day:
            new_day = False
    return new_day

# In[6]:
'''
Methods to plot according to certain constraints like day or hour
TODO: Condense code, lots of repeated code
'''

##Creates a bar plot for the entire data set with detections per day
def plot_by_day(detections, days):
    dets_per_day = {}
    for day in days:
        day_count = 0
        for time in detections['time']:
            if time[:8] == day:
                day_count += 1
        dets_per_day[day] = day_count
    names = []
    values = []
    for key in dets_per_day:
        names.append(key)
        values.append(dets_per_day[key])
    plt.bar(names, values)
    plt.xlabel('Day')
    plt.ylabel('Number Detections')
    plt.suptitle('Total Detections Per Day')
    plt.show()
    

##Creates a bar plot for a single day with detections per hour
def plot_by_hour(detections, hours, day):
    dets_per_hour = {}
    for hour in hours:
        hour_count = 0
        for time in detections['time']:
            if time[9:11] == hour:
                hour_count+=1
        dets_per_hour[hour] = hour_count
    names = []
    values = []
    for key in dets_per_hour:
        names.append(key)
        values.append(dets_per_hour[key])
    plt.bar(names, values)
    plt.xlabel('Hour')
    plt.ylabel('Number Detections')
    plt.suptitle('Total Detections Per Hour on Day: ' + day)
    plt.show()


def show(detections):
    days = get_days(detections)
    plot_by_day(detections, days)
    for day in days:
        sorted_times = sort_by_day(detections, day)
        hours = get_hours(sorted_times)
        plot_by_hour(sorted_times, hours, day)

def show_by_device(detections, devices):
    for device in devices: 
        print("Showing Data For Device: " + device)
        device_data = sort_by_device(detections, device)
        show(device_data)

def show_all(detections, devices):
    print("Showing Data For All Devices")
    show(detections)
    show_by_device(detections, devices)

# In[7]:
## A method to report the confidence scores 
def report_confidence(detections):
    mean_conf = detections.loc[:,"confidence"].mean()
    return mean_conf

def report_by_device(detections, devices):
    for device in devices:
        sorted_dev = sort_by_device(detections, device)
        print('For Device ' + device + ': ' + str(report_confidence(sorted_dev)))

def confidence_reports(detections, devices):
    print('Confidence Scores Over All Days: ')
    print('All Devices: '+ str(report_confidence(detections)))
    report_by_device(detections, devices)
    
    days = get_days(detections)
    for day in days:
        sorted_times = sort_by_day(detections, day)
        print('\nConfidence Scores for Day: ' + day)
        print('All Devices: ' + str(report_confidence(sorted_times)))
        report_by_device(sorted_times, devices)
        
##Show all plots and all confidence reports
## devices should be a list of devices the s3 bucket has collected data from
devices = ["Seattle01"]
detections = download_json_data()
show_all(detections, devices)
confidence_reports

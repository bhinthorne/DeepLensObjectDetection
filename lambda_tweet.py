from __future__ import print_function

import json
import urllib
import tweepy
import random
import boto3
import ast
import datetime

_CONSUMER_KEY = "YOUR KEY"
_CONSUMER_SECRET = "YOUR SECRET KEY"
_ACCESS_TOKEN = "YOUR TOKEN"
_ACCESS_TOKEN_SECRET = "YOUR SECRET TOKEN"

messages = ["Coffee Time!", 
            "Having a great day, enjoying some great coffee!", 
            "This coffee deserves a thumbs up!"]

print('Loading function')

def get_device():
    bucket_name = 'YOUR-BUCKET-NAME'
    client = boto3.client('s3')
    s3objects = client.list_objects_v2(Bucket=bucket_name, StartAfter = 'latest/')
    for object in s3objects['Contents']:
        if object['Key'].startswith('latest/'):
            obj = client.get_object(Bucket=bucket_name, Key=object['Key'])
            ##Retrieve the body and time of the json files as strings
            data = obj['Body'].read().decode('utf-8')
            #Cast the data to a dictionary
            data_dict = ast.literal_eval(data)
            device = data_dict["device_id"]
            return device
        
def construct_message(messages):
    now = datetime.datetime.now()
    device = get_device()
    index = random.randint(0,len(messages)-1)
    if now.minute<10:
        minute = "0" + str(now.minute)
    else: 
        minute = str(now.minute)
    header = "Botchitecture report at " + str(now.hour) + ":" + minute + " (UTC): "
    message = header + messages[index] + " #botchitecture #" + device
    return message

def tweet(CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_TOKEN_SECRET):
    
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    message = construct_message(messages)
    api.update_status(message)


def lambda_handler(event, context):
    try:
        tweet(_CONSUMER_KEY,_CONSUMER_SECRET,_ACCESS_TOKEN,_ACCESS_TOKEN_SECRET)

    except Exception as e:
        print(e)
        raise e

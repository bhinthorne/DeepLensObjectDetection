import boto3
import ast
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#This method downloads json files, the frames with bounding boxes,
#and the raw images
#it also stores the data from the json files in a dictionary with key of 
#a filename and object of another dictionary with the annotation data
def download_training_data():
    bucket_name = 'thumbs-up-output-3'
    client = boto3.client('s3')
    s3 = boto3.resource('s3')
    s3objects = client.list_objects_v2(Bucket=bucket_name)
    counter = 0
    annotation_data={}
    for object in s3objects['Contents']:
        if object['Key'].startswith('frames_resized/r'):
            obj_1 = s3.Object(bucket_name,object['Key'])
            image_filename = object['Key'].replace('frames_resized/','')
            #print(image_filename)
            obj_1.download_file('training_images/' + image_filename)
        
        if object['Key'].startswith('raw_images/r'):
            obj_3 = s3.Object(bucket_name,object['Key'])
            raw_image_filename = object['Key'].replace('raw_images/','')
            #print(raw_image_filename)
            obj_3.download_file('raw_images/' + raw_image_filename)
        
        if object['Key'].startswith('training_annotation/r'):
            obj_2 = s3.Object(bucket_name,object['Key'])
            json_filename = object['Key'].replace('training_annotation/','')
            #print(json_filename)
            obj_2.download_file('training_annotations/'+ json_filename)
            obj = client.get_object(Bucket=bucket_name, Key=object['Key'])
            ##Retrieve the body and time of the json files as strings
            data = obj['Body'].read().decode('utf-8')
            #Cast the data to a dictionary
            data_dict = ast.literal_eval(data)
            annotation_data[json_filename] = data_dict
            #print(data_dict['file'])
    return annotation_data
            
## I don't know why this is only showing one image right now
## It is supposed to show every image with it's corresponding data for comparison
def show_data():            
    data = download_training_data()
    counter = 0
    for key in data:
        image_name = data[key]['file'].replace('retrain','resized')
        #print(image_name)
        image_path = os.getcwd()+'/training_images/'+image_name
        img=mpimg.imread(image_path)
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        print(data[key])
        print("\n")
        plt.show()

show_data()
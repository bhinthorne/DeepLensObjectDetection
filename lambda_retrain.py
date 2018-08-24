#*****************************************************
#                                                    *
# Copyright 2018 Amazon.com, Inc. or its affiliates. *
# All Rights Reserved.                               *
#                                                    *
#*****************************************************
""" A sample lambda for object detection"""
from threading import Thread, Event
import os
import json
import numpy as np
import awscam
import cv2
import greengrasssdk
import mo

from botocore.session import Session
import datetime
import time

Write_To_FIFO = True

class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()

# Function to write to S3
# The function is creating an S3 client every time to use temporary credentials
# from the GG session over TES 
def write_image_to_s3(img, output, time, file_name, devices, resized_img):
    # Create an IoT client for sending to messages to the cloud.
    client = greengrasssdk.client('iot-data')
    iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
    session = Session()
    s3 = session.create_client('s3')
    device = devices[os.environ['AWS_IOT_THING_NAME']]
    record = 'json/record_'+ device + '_' +time+'.json'
    #path to a resized image
    resized_image = 'frames_resized/resized_' + device + '_' + time + '.jpg'
    #latest record uploaded to it's own directory
    latest = 'latest/latest.json'
    # You can contorl the size and quality of the image
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    _, jpg_data = cv2.imencode('.jpg', img, encode_param)
    _, resized_data = cv2.imencode('.jpg',resized_img,encode_param)
    response = s3.put_object(Body=jpg_data.tostring(),Bucket='YOUR-BUCKET-NAME',Key=file_name)
    response2 = s3.put_object(Body=json.dumps(output),Bucket='YOUR-BUCKET-NAME',Key=record)
    response3 = s3.put_object(Body=json.dumps(output),Bucket='YOUR-BUCKET-NAME',Key=latest)
    response4 = s3.put_object(Body=resized_data.tostring(),Bucket='YOUR-BUCKET-NAME',Key=resized_image)
    
    #client.publish(topic=iot_topic, payload="Response: {}".format(response))
    client.publish(topic=iot_topic, payload="Response: {}".format(response2))
    client.publish(topic=iot_topic, payload="Data pushed to S3")

    image_url = 'https://s3.amazonaws.com/YOUR-BUCKET-NAME/'+file_name
    return image_url
    
##This method writes the training data to the s3 bucket
##Write_to_s3 was getting to large and confusuing so separating this method
def write_training_data(raw_file_name, raw_img, train_annotation):
    client = greengrasssdk.client('iot-data')
    iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
    session = Session()
    s3 = session.create_client('s3')
    ##path to raw_image
    raw_image_path = 'raw_images/'+raw_file_name + '.jpg'
    #path to json annotation
    json_annotation_path = 'training_annotation/' + raw_file_name + '.json'
    
    # You can contorl the size and quality of the image
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    _, raw_data = cv2.imencode('.jpg', raw_img, encode_param)
    raw_image_response = s3.put_object(Body=raw_data.tostring(),Bucket='YOUR-BUCKET-NAME',Key=raw_image_path)
    annotation_response = s3.put_object(Body=json.dumps(train_annotation),Bucket='YOUR-BUCKET-NAME',Key=json_annotation_path)

##A method to create a annotation file from an inference in a frame
##to be reused for further training
def create_training_json(filename, annotations):
    training_data = {}
    training_data['file'] = filename + '.jpg'
    ##image size will always be the same
    training_data['image_size'] = [{"width": 512,"height": 512,"depth": 3}]
    training_data['annotations'] = annotations
    ##Will always be the same
    training_data['categories'] = [{"class_id": 0,"name": "thumbs_up"}]
    return training_data
    

def greengrass_infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        lapse = 0
        interval = 10
        # This object detection model is implemented as single shot detector (ssd), since
        # the number of labels is small we create a dictionary that will help us convert
        # the machine labels to human readable labels.
        model_type = 'ssd'
        output_map = {1: 'thumbs-up'}
        # Create an IoT client for sending to messages to the cloud.
        client = greengrasssdk.client('iot-data')
        iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()
        # The sample projects come with optimized artifacts, hence only the artifact
        # path is required.
        model_name = 'model_algo_1'
        #aux_inputs = {--epoch: 30}
        #error, model_path = mo.optimize(model_name, 512, 512, 'MXNet', aux_inputs)
        #if not error:
            #raise Exception('Failed to optimize model')
        model_path = '/opt/awscam/artifacts/model_algo_1.xml'
        # Load the model onto the GPU.
        client.publish(topic=iot_topic, payload='Loading object detection model')
        model = awscam.Model(model_path, {'GPU': 1})
        client.publish(topic=iot_topic, payload='Object detection model loaded')
        ## Debiging AWS_IOT_THING_NAME
        client.publish(topic=iot_topic, payload = os.environ['AWS_IOT_THING_NAME'])
        # Set the threshold for detection
        detection_threshold = 0.90
        # The height and width of the training set images
        input_height = 512
        input_width = 512
        # A dictionary to identify device id's
        devices = {}
        devices['deeplens_HoVip9KQTXiC3UFub47lJA'] = "Seattle01"
        devices['deeplens_bmTWwitIRUi_mASjZASUHA'] = "Chicago01"
        devices['deeplens_Hgs6kj_yQASF2x-3fOxCHA'] = "Chicago02"
        
        
        # Do inference until the lambda is killed.
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (input_height, input_width))
            
            ##Store a raw resized image before inference to be used for retraining later
            raw_training = cv2.resize(frame,(input_height,input_width))
            
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a ssd model,
            # a simple API is provided.
            parsed_inference_results = model.parseResult(model_type,
                                                         model.doInference(frame_resize))
            # Compute the scale in order to draw bounding boxes on the full resolution
            # image.
            yscale = float(frame.shape[0]/input_height)
            xscale = float(frame.shape[1]/input_width)
            # Dictionary to be filled with labels and probabilities for MQTT
            cloud_output = {}
            # Index to keep track of multiple detections in a frame
            index = 0
            # Get the detected objects and probabilities
            # A dictionary containing data for a single detection in a frame
            frame_time = time.strftime("%Y%m%d-%H%M%S")
            detection = {}
            detection["device_id"] = devices[os.environ['AWS_IOT_THING_NAME']]
            detection["timestamp"] = frame_time
            # A list that will contain the information of the objects detected in the frame
            objects_det = []
            # A boolean recording if a detection was made or not
            detection_made = False
            
            ##Set a list of annotations to be outputted in json file for training
            annotations = []
            
            for obj in parsed_inference_results[model_type]:
                # A dictionary to contain the info from an object detected
                object = {}
                
                # A dictionary containing annotation info for retraining
                annotation = {}
                
                if obj['prob'] > detection_threshold:
                    
                    # Set the annotation data for retraining
                    annotation['class_id'] = 0
                    annotation['left'] = int(obj['xmin'])
                    annotation['top']= int(obj['ymin'])
                    annotation['width'] = abs(int(obj['xmax'])-int(obj['xmin']))
                    annotation['height'] = abs(int(obj['ymax'])-int(obj['ymin']))
                    ## append to the list of annotations
                    annotations.append(annotation)
                    
                    #Set detection_made to true
                    detection_made = True
                    # Add bounding boxes to full resolution frame
                    xmin = int(xscale * obj['xmin']) \
                          + int((obj['xmin'] - input_width/2))
                    ymin = int(yscale * obj['ymin']) \
                           + int((obj['ymin'] - input_height/2)) 
                    xmax = int(xscale * obj['xmax']) \
                           + int((obj['xmax'] - input_width/2))
                    ymax = int(yscale * obj['ymax']) \
                           + int((obj['ymax'] - input_height/2))
                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.rectangle method.
                    # Method signature: image, point1, point2, color, and tickness.
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 10)
                    # Amount to offset the label/probability text above the bounding box.
                    text_offset = 15
                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.putText method.
                    # Method signature: image, text, origin, font face, font scale, color,
                    # and tickness
                    cv2.putText(frame, "{}: {:.2f}%".format(output_map[obj['label']],
                                                               obj['prob'] * 100),
                                (xmin, ymin-text_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 165, 20), 6)
                    ##This is putting bounding boxes for images resized to 512
                    cv2.rectangle(frame_resize,(int(obj['xmin']),int(obj['ymin'])),(int(obj['xmax']), int(obj['ymax'])),(255,165,20),10)
                    cv2.putText(frame_resize, "{}: {:.2f}%".format(output_map[obj['label']],
                                                               obj['prob'] * 100),
                                (int(obj['xmin']), int(obj['ymin'])-text_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 165, 20), 6)
                    # Store label and probability to send to cloud
                    #cloud_output[output_map[obj['label']]] = obj['prob']
                    ##Add to the dictionary of cloudoutput the index of the detection
                    #cloud_output['index'] = index
                    # set detection data for the object
                    object["index"] = index
                    object["object"] = output_map[obj['label']]
                    object["confidence"] = obj['prob']
                    # append the object to the list of detections for the frame
                    objects_det.append(object)
                    index += 1
            # Add the detections to the dictionary for the frame
            detection["objects"] = objects_det
            
            # add a link to the image to detections
            img_file_name = 'images/image_'+detection["device_id"]+'_'+frame_time+'.jpg'
            link = 'https://s3.amazonaws.com/thumbs-up-output-3/'+img_file_name
            detection["link_to_img"] = link
            
            #a filename for the raw image to be used for trianing later
            raw_file_name = 'retrain_'+detection["device_id"]+'_'+frame_time
            
            # Upload to S3 to allow viewing the image in the browser, only if a detection was made
            if detection_made and lapse>interval:
                #Create the json for retraining
                training_json = create_training_json(raw_file_name, annotations)
                #Upload the retraining data
                write_training_data(raw_file_name, raw_training, training_json)
                #Upload the inference data
                image_url = write_image_to_s3(frame, detection, frame_time, img_file_name, devices, frame_resize)
                #Publish success messages
                client.publish(topic=iot_topic, payload='{{"img":"{}"}}'.format(image_url))
                client.publish(topic=iot_topic, payload=json.dumps(detection))
                lapse = 0
            else:
                client.publish(topic=iot_topic, payload="NO DETECTIONS MADE")
                lapse += 1
            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))

greengrass_infinite_infer_run()

# Detecting a Thumbs Up with Amazon DeepLens

In this project we set out to train and deploy a custom object detection model on Amazon DeepLens to recognize human gestures. We decided to first start with the simple and widely understood "thumbs up". We followed the process all the way from collecting the intial data set to recording inferences from the deployed model on the DeepLens.

## **Before We Start**

----

Before diving into training an object detection model, it helps to have a background in the basics of machine learning and convolutional nerual nets (CNNs). A great resource to get started if you are new to machine learning and object detection is Michael Nielsen's free [online text book](http://neuralnetworksanddeeplearning.com/index.html).

There are a few things to set up before moving forward with training. We decided to train our model using Amazon SageMaker on AWS. You can set up an AWS account [here](https://aws.amazon.com/free/?sc_channel=PS&sc_campaign=acquisition_US&sc_publisher=google&sc_medium=ACQ-P%7CPS-GO%7CBrand%7CSU%7CCore%7CCore%7CUS%7CEN%7CText&sc_content=Brand_Account_bmm&sc_detail=%2Baws%20%2Baccount&sc_category=core&sc_segment=280392801059&sc_matchtype=b&sc_country=US&sc_kwcid=AL!4422!3!280392801059!b!!g!!%2Baws%20%2Baccount&s_kwcid=AL!4422!3!280392801059!b!!g!!%2Baws%20%2Baccount&ef_id=Wyf25QAABWB@RUiB:20180730230333:s). The next step is to set up and get familiar with [Amazon SageMaker](https://aws.amazon.com/sagemaker/). The final step is to register an Amazon DeepLens device, which you can get started with [here](https://aws.amazon.com/deeplens/).

A note for first time AWS users (as I was one myself): it may take some time to get familiar with using these services. We read documentation from many different sources and went through a variety of tutorials before we were able to train our first custom model succesfully. Amazon SageMaker has some built in tutorials that are good, and another resource we followed is this [github repository](https://github.com/mahendrabairagi/DeeplensWorkshop). Hopefully, however, you will be successful in following this tutorial from start to finish. 

## **Collecting the Data**

----

In order to obtain our data set we gathered images from two different sources. The first was pulling images of thumbs up from Google, the second was taking our own pictures of thumbs up in an office space. The specific goal of our project was to deploy the DeepLens in an office environment, so we felt it necessary to have pictures from such an environment in our dataset. On the other hand, it is important to have a variety of types of images. Getting a diverse dataset in size, background color, perspective, and other variables can help improve training. Our most recent data set contained 375 images, about 100 were taken by us in an office space and the rest were pulled from Google images.  

Once we have our raw images, we can start the preprocessing of the data set. Amazon SageMaker offers two types of training with their built in object detection algorithm described [here](https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html?shortFooter=true). The first of which (actually recomended by Amazon) is using Apache MXnet RecordIO files. So far, however, we have been successful using the second form of data supported, raw images (.jpg or .png) and corresponding json files containing the annotations for each image in the data set. Our data preprocessing pipeline therefore went as follows:

1. Gather raw images into the same folder.
2. Annotate these images, creating corresponding xml files for each image.
3. Convert these xml files to a single csv file containing all annotations
4. Convert this csv file into individual json annoation files for each image

### **Annotating The Images**

Once you have your images gathered, the next step is the annotation process. To label our images we cloned [this github repository](https://github.com/tzutalin/labelImg), and ran the script:

``` 
python3 labelIMg.py
```

From here there is a user interface in which you can open the folder containing your images and begin annotating. After we drew our first bounding box we created the label "thumbs_up", which we used as the label for the rest of our images. Once we drew the bounding box around each object of interest in the image, we saved the annoatations in the same folder as the images (annotations will save as an xml file). Once you have gone through every image in the directory, the annoation is complete.

A note on image annotation: annoating images very carefully is worth your time. Although the process is very tedious, it helps to make sure that your bounding boxes are tightly situated around your object of interest. This will make sure the training is happening on the features you are interested in (i.e. the features of your object) and not the surrounding area. At first our annoatations were a bit sloppy, once we went back an re-annotated our images, our training results improved greatly. Some shortcuts to expadite the process: 'w' to start a new bounding box, cmd + s to save.

### **Converting xml to csv**

The next step in the precprocessing of the data is to convert our newly created xml annotations to a single csv file containing annotations. We used the script, xml_to_csv.py, which can be found publicly available online. The version we used is below. To change it to fit your data set, simply change "Image Directory" to the directory containing your images and xml annotations, and "Output CSV Name" to the name you want your output csv file to be.

``` python
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
        image_path = os.path.join(os.getcwd(), <Image Directory>)
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(<Output CSV Name>, index=None)
        print('Successfully converted xml to csv.')


main()
```

### **Converting CSV to JSON files**

The final step is to take the csv file we created that stores our annotations and create corresponding json files for each image. The specific content and format of these json files are specified in [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection.html?shortFooter=true). In order to convert our csv file to the corresponding json files needed for the object detection algorithm on SageMaker, we wrote our own script called csv_to_json.py. This script was written specifically for our purposes to take the specific csv files we generated and create the specific format of json files needed for training on Amazon SageMaker. This script is detailed below. In order to make it work for your data set, simply change the parameters when create_files() is called to be the path to your csv file and the name of your desired output folder. Be aware however that this file is specfically written for our purposes of only detecting a thumbs up. If you intend to work with more than one object, you may have to make some changes to this script before running.

``` python
import csv
import json

#loads all file names into a list for comparison later
def loadFileNames(file_name):
    names = []
    with open(file_name) as rfile:
        reader = csv.DictReader(rfile)
        for row in reader:
            names.append(row['filename'])
    names.append('ENDFILE')
    return names

#Returns true if next filename is the same as current
def checkNextName(index, names):
    if names[index] == names[index+1]:
        return True
    else:
        return False

#Returns true if the previous filename is the same as current
def checkPrevName(index, names):
    ##If it is the first file return false
    if index == 0:
        return False
    elif names[index] == names[index-1]:
        return True
    else:
        return False

def create_files(file_name, output_folder):
    names = loadFileNames(file_name)
    with open(file_name) as rfile:
        reader = csv.DictReader(rfile)
        counter = 0
        for row in reader:
            ##Reading Data
            filename = row['filename']
            width = int(row['width'])
            height = int(row['height'])
            depth = 3
            class_name = row['class']
            ##Since we only have one class, class_id will always be zero.
            class_id = 0
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            box_width = abs(xmax-xmin)
            box_height = abs(ymax-ymin)
            #Done Reading Data
            #If the previous name is not the same, clear data and load the new data
            if not checkPrevName(counter, names):
                ## Clearing Data
                data = {}
                ## Loading Data
                data['file'] = filename
                data['image_size'] = []
                data['image_size'].append({
                'width':width,
                'height':height,
                'depth':depth
                })
                data['annotations'] = []
                data['annotations'].append({
                    'class_id':class_id,
                    'left':xmin,
                    'top':ymin,
                    'width':box_width,
                    'height':box_height
                })

                data['categories'] = []
                data['categories'].append({
                    'class_id':class_id,
                    'name':class_name
                })
            #If the previous file name is the same, append only annotation data
            if checkPrevName(counter, names):
                ##Appending Annotations
                data['annotations'].append({
                    'class_id':class_id,
                    'left':xmin,
                    'top':ymin,
                    'width':box_width,
                    'height':box_height
                })
            #If the next name is not the same then we want to write the data into a json file
            if not checkNextName(counter, names):
                ## error handling for .jpeg vs. .jpg file names
                if filename[-4:] == "jpeg":
                    new_name = filename[:-5]
                else:
                    new_name = filename[:-4]
                output_loc = output_folder+new_name+'.json'
                with open(output_loc, 'w') as outfile:
                    json.dump(data, outfile)
            counter += 1

create_files(<Your CSV Filename>, <output_folder/>)
```

You should now have a folder containing all the json files with annotations for each image! The final step to preprocess the data is to split your images and annoatations into train and validation folders. First, create folders train, validation, train_annotation, and validation_annoatation. Then split up your images and json files into train and validation categories. You want many more training images than validation images. We had 359 images for training and 16 for validation. Put your training images into the train directory, and the corresponding json files into the train_annoation directory. Do the same with your validation data for the validation and validation_annoataion directories. Your data is now all preprocessed! The next step is to move onto AWS to begin training.

## **Uploading to AWS**

----

At this point we have all of our data configured locally on our computer and the next step is to upload the needed files to AWS and move forward with the training process. We completed the registration and configuration of our AWS account in the "Before We Start" section. To begin training with AWS there are two things we need to do: create an S3 storage bucket to store our data, and create a notebook instance to run our training script.

### **Creating S3 Bucket**

To create an S3 bucket, navigate to the S3 storage services and select "Create Bucket". Enter the bucket name, our name for the first model that worked was "deeplens-sagemaker-thumbs-up-v1". Be sure to select the region that is the same as the IAM role you are using, and where your notebook instance will be running. To correspond with the notebook instance we used, we created the following file structure in our S3 bucket. We first made the folder "DEMO-ObjectDetection". Inside of DEMO-ObjectDetection, we hit "upload" and drag and dropped the four folders of data we created locally earlier: train, validation, train_annoation, and validation_annotation. We now have all the data for our custom model uploaded into an S3 Bucket.

### **Creating a notebook instnace for training**

To create a notebook instance for training, navigate to SageMaker from AWS services and go to the Notebook instances tab. Chose "Create Notebook Instance" and give your notebook isntance a name. We left all other settings as they were, but for the IAM role we used an "AmazonSageMaker-ExecutionRole" (which I believe we created before?).

For our training, we used an already existing SageMaker notebook and modified it to fit our needs. From the "SageMaker" examples in the notebook instance, in "Introduction to Amazon algorithms", we selected "Use" for the notebook titled "object_detection_image_json_format.ipynb". This should create a directory in the notebook instance titled "object_detection_pascalvoc_coco_%datecreated" (feel free to rename this). Inside of this directory, open the file "object_detection_image_json_format.ipynb". This is the file we will modify to complete our training.

## **Modifying an Example Notebook for Training**

----

### **Set Up**

At this point we have to make changes to this notebook to accomodate our data set. In the first section titled "Set Up" the only change we made is setting 'bucket = YOUR BUCKET NAME' to point to the bucket where our training data is stored. For us this looked like:

``` python
bucket = 'deeplens-sagemaker-thumbs-up-v1'
```

We then ran the three cells under the "Set Up" section. The next section titled "Data Preperation" is not needed for our training. This section is downloading the PascalVOC dataset that the example is built to train on. Since we have already created our own data, we do not need to train on this data. If we were to train on PascalVOC data rather than our own custom data, we could just run the notebook straight without any preprocessing or uploading of the data.

### **Upload to S3**

We then move down to the "Upload to S3" section of the notebook. In the first cell of this notebook we deleted the middle block of code, which is uploading the PascalVOC data to S3. We have already manually uploaded our data to S3, so this step is not needed for us. The first cell under "Upload to S3" in our notebook then contained the following code:

``` python
%%time

train_channel = prefix + '/train'
validation_channel = prefix + '/validation'
train_annotation_channel = prefix + '/train_annotation'
validation_annotation_channel = prefix + '/validation_annotation'

s3_train_data = 's3://{}/{}'.format(bucket, train_channel)
s3_validation_data = 's3://{}/{}'.format(bucket, validation_channel)
s3_train_annotation = 's3://{}/{}'.format(bucket, train_annotation_channel)
s3_validation_annotation = 's3://{}/{}'.format(bucket, validation_annotation_channel)
```

We left the next cell as is. s3_output_location creates a folder in the s3 storage bucket where the trained model will be stored. We then ran these two cells under "Upload to S3".

### **Training**

Now moving on to the section titled training, this is where we create our model with hyperparameters and begin the training. The first cell creates the sagemaker.estimator.Estimator object which will be used for training. We did not make any changes to this cell, thus its contents were:

``` python
od_model = sagemaker.estimator.Estimator(training_image,
                                         role,
                                         train_instance_count=1,
                                         train_instance_type='ml.p3.2xlarge',
                                         train_volume_size = 50,
                                         train_max_run = 360000,
                                         input_mode = 'File',
                                         output_path=s3_output_location,
                                         sagemaker_session=sess)
```

The next cell contains the hyperparameters for training. Hyperparamters can have a large impact on the success of a training job. Understanding convolutional neural networks and how the algorithms work can help understand how changing certain hyperparameters can affect the success of training a model. A description of the hyperparameters used by SageMaker's built in training algorith can be found [here](https://docs.aws.amazon.com/sagemaker/latest/dg/object-detection-api-config.html?shortFooter=true).

We left the base network to be resnet-50 for our first iteration of training. We decided to use a pretrained model for our traianing because training from scratch usually takes much longer to be successful, so transfer learning seemed like the way to go. We set our first mini_batch_size to 10. Batch size is how many images are fed to the network before the weights and biases are updated. So a batch size of 10 means that the CNN will see 10 images before updating weights and biases. Once all training images have been fed through the network, that is an epoch. We ran for 30 epochs for our first training iteration. We set our learning rate to .004, this is something that can be adjusted to affect training greatly. A larger learning rate means that the weights and baises are changed more dramatically with each batch. A smaller learning rate means that they are changed by less with each batch. lr_scheduler_step and factor determine the rate of decay of the learning rate. As training progresses the convention is to usually lower the learning rate. The way I understand this is as you approach the ideal weights and biases, smaller and more precise steps will be more beneficial then larger and broader steps towards the ideal model. We left image_shape and label_width as their default values in the notebook for our first training job. We had 359 training examples. Make sure this value is exactly how many training samples you have, that is excluding the validation samples. The hyperparameters we ran our first training job with are here:

``` python
od_model.set_hyperparameters(base_network='resnet-50',
                             use_pretrained_model=1,
                             num_classes=1,
                             mini_batch_size=10,
                             epochs=30,
                             learning_rate=0.004,
                             lr_scheduler_step='10',
                             lr_scheduler_factor=0.1,
                             optimizer='sgd',
                             momentum=0.9,
                             weight_decay=0.0005,
                             overlap_threshold=0.5,
                             nms_threshold=0.45,
                             image_shape=512,
                             label_width=350,
                             num_training_samples=359)
```

After we had our hyperperamters set, we ran that cell. The cell below it creates data channels to feed to the model for training, we ran that cell without making any changes. This brings us to the cell with contents:

``` python
od_model.fit(inputs=data_channels, logs=True)
```

Once you run this cell, the training job will start. Depending on how much data you have, how many epochs you run, and your hyperparamters, the length of this cell's runtime will vary. For our first specified hyperparameters, the cell ran for about 7-10 minutes. When you run the cell, you should start to see something like this show up: ![image](training_shot.jpg). At first the string of dots will show up and begin progressing, eventually a lot of red text will show up documenting the training job. Once the training job is complete, you should see a message that looks like this:  

```
===== Job Complete =====
Billable seconds: 556
```

### **Hosting**

The way Amazon SageMaker does inference is they create an endpoint for hosting. To do this we ran the command:

``` python
object_detector = od_model.deploy(initial_instance_count = 1,
                                 instance_type = 'ml.m4.xlarge')
```

This command will take a few minutes to run, you will know it is finished when you see dashes and they end with an exclamation point. The end will look like "---!".

### **Inference**

Now that we have our endpoint created, we can run inference on our model! We made some modifications to the built in inferences that were in the example notebook. The first thing we did was create a folder in our notebook instance called "validation". Within this folder we uploaded images we wanted to test the model with. The next thing we did was run the cell with the method "visualize detection". This method is a helper method to display the images with the inference bounding boxes. We then modified the  next cell to go through our validation images and make inferences on them, displaying the images with bounding boxes and printing their confidence scores and coordinates. The contents of this cell are:

``` python
object_categories = ['thumbs-up']
# Setting a threshold 0.30 will only plot detection results that have a confidence score greater than 0.30.
threshold = 0.3

import os
import json

for file_name in os.listdir(os.getcwd()+'/validation'):
    with open('validation/'+file_name, 'rb') as image:
        f = image.read()
        b = bytearray(f)
        ne = open('n.txt','wb')
        ne.write(b)
    object_detector.content_type = 'image/jpeg'
    results = object_detector.predict(b)
    detections = json.loads(results)
    this = detections['prediction']
    print(file_name + ":")
    for l in this:
        if l[1]>threshold:
            print(l)
    visualize_detection('validation/' + file_name, this, object_categories, threshold)
# Visualize the detections.
```

The variable object_categories should be a list of the names of the objects you are trying to detect. We only had one object, the 'thumbs-up'. The threshold variable defines at what confidence score you wish to make detections. 0.3 means that anything with a confidence score greater than 0.3 will be detected. The rest of the method goes through every image in the validation directory and runs inferences on them. For each object it will print the detections as a list: [object_id, conf_score, xmin, ymin, xmax, ymax]. It will then display the image with the bounding boxes drawn onto it.

Below I have included a summary of the results from our first verion of training:

| Expected Predictions | True Predictions | False Predictions | Missed Predictions | Avg. True Conf | Avg. False Conf | Time | 
|:--------------------:|:----------------:|:-----------------:|:------------------:|:--------------:|:---------------:|:----:|
|30|26|7|4|0.677|0.431|550|

## **Tuning Hyperparameters**

----

Even after succesful training, we decided that the first model we trained needed to be improved. We set out to tune our model by changing the hyperparameters, but leaving the data set the same. As a reminder, our data set has 359 training images and 16 test images for a total of 375 images. In addition, to stay consistent all of our inferences had a threshold of 0.3. This means only inferences with a confidence score of larger than 0.3 were recorded.

### **Version 2**

 We increased the number of epochs because we hypothesized that 30 epochs was too short of a training time, and training for longer would therefore strengthen our model. We decided to increase mini_batch_size because we hypothesized that a batch size of 10 was reducing the speed of the training and so could inhibit our results. We therefore increased mini_batch_size to 15. Lastly, we changed image_shape from 512 to 300 because we thought a shape of 300 would be a better representation of our training data.

``` python
od_model.set_hyperparameters(base_network='resnet-50',
                             use_pretrained_model=1,
                             num_classes=1,
                             mini_batch_size=15,
                             epochs=60,
                             learning_rate=0.004,
                             lr_scheduler_step='10',
                             lr_scheduler_factor=0.1,
                             optimizer='sgd',
                             momentum=0.9,
                             weight_decay=0.0005,
                             overlap_threshold=0.5,
                             nms_threshold=0.45,
                             image_shape=300,
                             label_width=350,
                             num_training_samples=359)
```

Results:

| Expected Predictions | True Predictions | False Predictions | Missed Predictions | Avg. True Conf | Avg. False Conf | Time |
|:--------------------:|:----------------:|:-----------------:|:------------------:|:--------------:|:---------------:|:----:|
|30|20|3|10|0.604|0.473|950|

The results from this version of training were much worse then the first iteration. The results had many more detections missed, lower confidence scores for true detections, and made almost all the same false detections as the first model.

### **Version 3**

In our next version of training we attempted to debug why version 2 went so poorly. We hypothesized it was either changing the image_shape or the mini_batch_size. To find out, we ran version 3 of training with image_shape set back to 512. Thus our hyperparameters for training looked like this:

``` python
od_model.set_hyperparameters(base_network='resnet-50',
                             use_pretrained_model=1,
                             num_classes=1,
                             mini_batch_size=15,
                             epochs=60,
                             learning_rate=0.004,
                             lr_scheduler_step='10',
                             lr_scheduler_factor=0.1,
                             optimizer='sgd',
                             momentum=0.9,
                             weight_decay=0.0005,
                             overlap_threshold=0.5,
                             nms_threshold=0.45,
                             image_shape=512,
                             label_width=350,
                             num_training_samples=359)
```

Results:

| Expected Predictions | True Predictions | False Predictions | Missed Predictions | Avg. True Conf | Avg. False Conf | Time | 
|:--------------------:|:----------------:|:-----------------:|:------------------:|:--------------:|:---------------:|:----:|
|30|19|5|11|0.766|0.553|950|

These results were similar to version 2, and still much worse than version 1. We can conclude from these results that changing the mini_batch_size from 10 to 15 did infact affect training in a negative way for our data set.

### **Version 4**

For the next version of training, changed the mini_batch_size back to 10. This means we were left with the same settings as version 1 of training, except the epochs doubled from 30 to 60.

``` python
od_model.set_hyperparameters(base_network='resnet-50',
                             use_pretrained_model=1,
                             num_classes=1,
                             mini_batch_size=10,
                             epochs=60,
                             learning_rate=0.004,
                             lr_scheduler_step='10',
                             lr_scheduler_factor=0.1,
                             optimizer='sgd',
                             momentum=0.9,
                             weight_decay=0.0005,
                             overlap_threshold=0.5,
                             nms_threshold=0.45,
                             image_shape=512,
                             label_width=350,
                             num_training_samples=359)
```

Results:

| Expected Predictions | True Predictions | False Predictions | Missed Predictions | Avg. True Conf | Avg. False Conf | Time | 
|:--------------------:|:----------------:|:-----------------:|:------------------:|:--------------:|:---------------:|:----:|
|30|29|8|1|0.875|0.454|950|

The results from this training were the best we have had yet. They were much improved from the versions 2 and 3 with different hyper parameters, and much more accurate than version 1, which had the same hyper parameters but less epochs than this version. The only concerning part about these results is the number of false predictions. While looking at the pictures however, many of the false predictions came on one image that seemed to be an outlier. In addition, this concern can be countered by the fact that the difference between the average confidence score for true predictions is so much larger than the average confidence score for false predictions. This means that if we were to raise our prediction threshold, the we could eliminate many of the false predictions while still retaining the majority of the true predictions.

### **Version 5**

Since we observed that increasing epochs with these hyperparamters improved the results of training, we decided to continue to increase the number of epochs and train again. For the next version of training, we increased the number of epochs to 90.

``` python
od_model.set_hyperparameters(base_network='resnet-50',
                             use_pretrained_model=1,
                             num_classes=1,
                             mini_batch_size=10,
                             epochs=90,
                             learning_rate=0.004,
                             lr_scheduler_step='10',
                             lr_scheduler_factor=0.1,
                             optimizer='sgd',
                             momentum=0.9,
                             weight_decay=0.0005,
                             overlap_threshold=0.5,
                             nms_threshold=0.45,
                             image_shape=512,
                             label_width=350,
                             num_training_samples=359)
```

Results:

| Expected Predictions | True Predictions | False Predictions | Missed Predictions | Avg. True Conf | Avg. False Conf | Time | 
|:--------------------:|:----------------:|:-----------------:|:------------------:|:--------------:|:---------------:|:----:|
|30|27|11|3|0.879|0.560|1350|

These results were overall not as succesful as version 4. There were two more missed predictions, three more false predictions, and a significantly higher average for false confidence scores, while only a slight increase in true confidence scores. It is possible that we have increased the epoch size too much, and the model has started to overfit.

### **Version 6**

So far we have observed that increasing batch_size from 10 to 15 detrimented training. Of the three different epoch lengths we have tried, we have observed that 60 epochs is optimal for our training. In the next version of training, we tried reducing mini_batch_size from 10 to 8, and set epochs back to 60.

``` python
od_model.set_hyperparameters(base_network='resnet-50',
                             use_pretrained_model=1,
                             num_classes=1,
                             mini_batch_size=8,
                             epochs=60,
                             learning_rate=0.004,
                             lr_scheduler_step='10',
                             lr_scheduler_factor=0.1,
                             optimizer='sgd',
                             momentum=0.9,
                             weight_decay=0.0005,
                             overlap_threshold=0.5,
                             nms_threshold=0.45,
                             image_shape=512,
                             label_width=350,
                             num_training_samples=359)
```

Results:

| Expected Predictions | True Predictions | False Predictions | Missed Predictions | Avg. True Conf | Avg. False Conf | Time | 
|:--------------------:|:----------------:|:-----------------:|:------------------:|:--------------:|:---------------:|:----:|
|30|26|8|4|0.818|0.580|950|

These results were not as good as version 4. This helps us conclude that the optimal mini_batch_size for our training is 10.

## **Summary of Results**

----

This first table shows the hyperparameters that changed between the versions of training

Training Version| mini_batch_size | image_shape | epochs |
|:-------------:|:---------------:|:-----------:|:------:|
|1|10|512|30|
|2|15|300|60|
|3|15|512|60|
|4|10|512|60|
|5|10|512|90|
|6|8|512|60|

This second table is a combined summary of all results from each version of training

Training Version| Expected Predictions | True Predictions | False Predictions | Missed Predictions | Avg. True Conf | Avg. False Conf | Time |
|:-------------:|:--------------------:|:----------------:|:-----------------:|:------------------:|:--------------:|:---------------:|:----:|
|1|30|26|7|4|0.677|0.431|550|
|2|30|20|3|10|0.604|0.473|950|
|3|30|19|5|11|0.766|0.553|950|
|4|30|29|8|1|0.875|0.454|950|
|5|30|27|11|3|0.879|0.560|1350|
|6|30|26|8|4|0.818|0.580|950|

### **Future Training Ambitions**

Although our training process was successful enough for our purposes, there are still some things we would like to look into to try and improve our model. Each model will have unique hyperparameters that optimize it's specific training. Although there are generalizations that can be made for how to tune hyperparemeters for any model, for any specific model the best way to optimize is essentially trial and error in training. In the six versions of training above we only touched slightly on adjusting mini_batch_size, epochs, and image_shape. We would love to try further trainings with slightly adusting these hyperparameters and changing the results. There are other hyperparameters we would like to experiment with as well. Learning rate is something that can have a great affect on training, so collecting data with a lower or higher learning rate could help us find the optimal model. In addition, experimenting with the lr_scheduler step and factor to change the learning rate could help optimize our model as well. Furthermore, we would like to look into how modifying our data set will affect training. We could train with more data, or new data, and see how our results change. As an extension, training with more than one class would also give us a better understanding of how to scale the project in the future. If we wish to detect more gestures than just a thumbs up, we could learn whether we can train an entire dataset of gestures or only one gesture at a time.

## **Deploying to DeepLens**

----

Now that we have successfully trained an object detection model for thumbs up, our next goal was to deploy this model for inference on the DeepLens. In the "Before We Start" section you should have already registered a DeepLens so it should be good to go.

### **Creating a Project**

To deploy a model to the DeepLens, you must create a Project that contains a model and a lambda function. The first step is to import your model to AWS DeepLens. First navigate to the AWS DeepLens console, and from the dashboard under "Resources" click on "Models". Within models select the orange button in the top right that says "Import Model". Now choose the import source to be "Amazon SageMaker trained model". Under model settings select the training job ID of the model you wish to upload. This is the ID of the SageMaker endpoint you creaed during the training process. "Model name" can be set to whatever you wish to identify your model. Make sure "model framework" is set to whatever you used to train your model. If you followed this tutorial for training and used the default SageMaker notebook that framework is MXNet. Once you have these settings configured, hit "Import model" and your model is good to go. 

Now that the model is loaded, the next step is to configure a lamda function for the project. First navigate to the AWS Lambda service and to "Functions", then select "Create function". We are going to use a blue print for our lambda function so select the "Blueprints" option and search for "greengrass-hello-world". Select "greengrass-hello-world" and you will be taken to a "Basic information" page. Name your function whatever you wish, ours was "thumbs-up-detection". For the role chose an existing role and chose "service-role/AWSDeepLensLambdaRole". You can then scroll down and hit "Create Function". Once your function is created, find it and select it on the "Functions" page. This will take you to a page containing the section "Function code". We need to modify "greengrassHelloWorld.py" to contain code to make inferences for our model. Replace the code in greengrassHelloWorld.py with the code detailed below:

```python
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

def greengrass_infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
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
        # Set the threshold for detection
        detection_threshold = 0.60
        # The height and width of the training set images
        input_height = 512
        input_width = 512
        # Do inference until the lambda is killed.
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (input_height, input_width))
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
            # Get the detected objects and probabilities
            for obj in parsed_inference_results[model_type]:
                if obj['prob'] > detection_threshold:
                    # Add bounding boxes to full resolution frame
                    xmin = int(xscale * obj['xmin']) \
                           + int((obj['xmin'] - input_width/2))
                    ymin = int(yscale * obj['ymin'])
                    xmax = int(xscale * obj['xmax']) \
                           + int((obj['xmax'] - input_width/2))
                    ymax = int(yscale * obj['ymax'])
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
                    # Store label and probability to send to cloud
                    cloud_output[output_map[obj['label']]] = obj['prob']
            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
            # Send results to the cloud
            client.publish(topic=iot_topic, payload=json.dumps(cloud_output))
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))

greengrass_infinite_infer_run()

```
 
There are a few things to notice or change in this code to adjust it for your model. The first is the output map, for our model this is set to {1:'thumbs-up'}. This map is a dictionary where they keys are the object IDs and the values are the corresponding labels. If you have only 1 object you can simply change 'thumbs-up' to whatever object you are trying to detect. If you have more than one object then structure you output map to look like: {1:'object1', 2:'object2',3:'object3'} and so on for however many objects your model is trained on. 

The next thing to notice is the "model_name" variable. Your sagemaker trained model should output three files: a hyperparms.json file, another .json file, and a .params file. For our model, these three files were hyperparams.json, model_algo_1-0000.params, and model_algo_1-symbol.json. As a reminder, to view these files navigate to the s3 bucket where your SageMaker training job outputted and navigate to the output folder. Then download and unzip "model.tar.gz" and you will see these files. You should have three similar files to ours, but the prefix may not be "model_algo_1". This prefix is what you need to set your "model_name" to. Notice for us, it is "model_algo_1".

The next thing to notice is the "model_path" variable. This path will point to an xml file that we will create once we have deployed the project onto the DeepLens. No inference will work until we have created this xml file, so don't expect the inferences to run right away when you deploy the project. For now, set the model path to "model_path = /opt/awscam/artifacts/'model_name'.xml", except replace 'model_name' with whatever you set to be the name of your model. 

The final thing to change is "input_height" and "input_width" which should be the height and width of your training images, which you set in your hyperparameters on SageMaker. For our model they were set to 512. You can also change "detection_threshold" to whatever value you want it to be between 0 and 1. The DeepLens will only record inferences that have a confidence score above this threshold. 

Once you have made all these changes save your lambda function. Once it is saved, under "Actions" chose publish. This will publish the first version of your lamda function.

Now that you have imported a model and created a lamda function we can now create a project. In the AWS DeepLens console, go to the Projects tab and select "Create new project". Then select "Create new blank project" and give your project a name. Then add the model you just imported and the lambda function you just created. Then hit "Create".

### **Deploying the Project to the DeepLens**

Now that the project is created we can upload it to the DeepLens. First make sure that your DeepLens is online and connected to the same Wifi network as your computer. If you go to the "Devices" tab in the AWS DeepLens console you should see your DeepLens with the "Device status" as "Online". Once your DeepLens is online, then return to the Projects tab. Select the project you just created and hit "Deploy to device". Then select your DeepLens, hit review, and then deploy. You should see a blue bar on the top of the screen displaying the progress of the deployment. It will track the downloading of the model and then display a green success message when the project has been uploaded to the device. Remember you will not be able to see any inferences after this first deployment, we first need to optimize the model for the DeepLens. 

Now open up a terminal in the DeepLens. You can either ssh into the DeepLens or connect to a monitor and work with a keyboard and mouse. Once you have a terminal open, change directories into /opt/awscam/artifacts. If your project was sucessfully deployed you should see the same three files in there that were outputted by your SageMaker training job. The next step is to optimize the model and create the xml file our lambda function points to for inference. Feel free to remove the project from the device at this point, the files will remain loaded on the DeepLens and we will be redeploying the project later.

### **Optimizing the Model for DeepLens**

To create an xml file for inference we have to use an AWS module called "mo" and run an mo.optimize function. This function takes our json files outputted from the training job, now deployed onto the DeepLens, and creates an xml file used for inference. After spending a long time struggling and trying to get the optimizer working on our model, we were informed that an SSD model trained on SageMaker cannot be directly deployed or optimized on the DeepLens. We first had to run the artifacts from the model training through a "deploy.py" script from this [github repo](https://github.com/apache/incubator-mxnet/tree/master/example/ssd).

To run deploy.py we first cloned the github repo from the DeepLens terminal with the command: 

```bash
sudo git clone https://github.com/apache/incubator-mxnet
```

Once we cloned the directory we then moved "model_name"-0000.params and "model_name"-symbol.json into the same directory as deploy.py. Make sure that you move these files, rather than making a copy. Making a copy may cause you to run into an issue of ownership of the files when downloading the odel later. Once the files are in the directory with deploy.py, we can then run deploy.py with the following command:

```bash
python3 deploy.py --network='resnet50' / 
    --epoch = 0 /
    --prefix = 'model_algo_1' / 
    --data-shape = 512 / 
    --num-class = 1 /
```

Be sure to adjust the paramaters to fit your trained model. A note about the epoch parameter, this has to do with the suffix of digits at the end of "model_name"-0000.params. We trained for 60 epochs with our model, so at first I thought logically epoch would be set to 60. I then got an error saying there was no such file "model_algo_1-0060.params". Since our file was "model_algo_1-0000.params" I set epoch to 0 and it worked. So make sure to set epoch to whatever the number is at the end of your .params file. If you run that command successfully you should see a message that looks like:

Saved Model: 'model_name'.params
Saved Model: 'model_name'.json

Your model artifacts are now ready to be optimized. We then moved the .params and .json files back to their original directory in /opt/awscam/artifacts. Again be sure to move these files, do not copy them. Now we can use the mo.optimize function mentioned earlier to generate the xml file. Return to the home directory in the DeepLens terminal, and open python with the sequence of commands:

```bash
cd ~
sudo python
```

If you wish to see the details about the model optimizer refer to the [aws documentation](https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-model-optimizer-api.html). We have provided the commands that worked for us. Once you have python opened, run these commands to optimize your model and generate the xml file:

```python
import mo
model_name = 'your_model_name'
input_height = your_input_height
input_width = your_input_width
mo.optimize(model_name, input_height, input_width)
```

By default this will generate an xml file with the name 'your_model_name'.xml in the /opt/awscam/artifacts directory. If you go back and look at the lambda function we created earlier, this is the file we are pointing to that contains our model. You now have a model optimized for inference on the DeepLens!

## **Viewing the Output**

Now that you have a model optimized and present on the DeepLens, you can now make inferences with your model. To be sure everything is set up correctly, go back to the DeepLens console and redeploy your project to the device. Once again make sure your device is connected to the same network as your computer before deployment. You will know your project is successfully deployed once all three blue lights on your DeepLens are illuminated.

### **Viewing Video Stream**

There are now a few options to view the output from the DeepLens. You can either connect your DeepLens to a monitor and use mplayer to view the output, ssh into the DeepLens and use mplayer to view the output on your own computer, or view the output directly in the DeepLens console through your browser. We have found that viewing the output directly in the console has been the most reliable. Details about viewing output streams can be seen [here](https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-viewing-device-output-on-device.html). To view the output directly in your browser, in your DeepLens console click on the "View Output" option and follow the steps provided to upload the streaming certificate and view the stream. Your browser may say that the connection is unsafe, or that it doesn't trust the address, but just continue to the stream anyways. 

Once you have successfully opened the stream viewing page, you will see two tabs: project stream, and live stream. Live stream is simply the video coming from your device without any model present. The project stream is where you will see inference results from your device. Our experience with the output stream has been nowhere near perfect, but we were excited to even see the stream with any inferences taking place. When we first tried to view output we could view the live stream, but our output stream was just a blank white screen. We fixed this when we realized our lambda function was not pointing to the xml file for our model. If you are having the same issue, double check your lambda and the location of your model xml file.

Whe viewing our output stream successfully it is still nowhere near perfect. Our stream was extremely laggy and seemed to have very low fps. We asked on an Amazon Forum if this was expected and the short answer is yes, the stream will not be perfect and you should expect some delay. With some deliberate placement of a thumbs up in the frame, however, we could see that our model was actually making correct inferences. One other issue we ran into is the positioning of the bounding box. The resizing of frames for inferencen and for the stream caused the bounding boxes to be offset when a detection of a thumb was being made. Since our deployment purposes are not super reliant on the stream output, we're not too concerned with the output bounding boxes, but we are still working to adjust them appropriately.

### **Viewing Cloudwatch Logs**

If you want to see your inferences and a form that is not a video stream, you can see them in the Cloudwatch Logs. If you go to your DeepLens console and scroll down to device settings you will see "Lambda Logs" with a little link to the logs next to it. If you click on this link you will be taken to a CloudWatch where you can navigate to your logs. If you find your lamda function, and then click on your DeepLens device, you will be taken to a stream of logs that will report on your DeepLens project. You should see messages about the model being loaded and you can also see print outs of your model's inference. The logs will also report any errors or warnings in your model, so when debugging our lamda function the logs were the most helpful way to view output. You may notice that the CloudWatch logs are organized so that the most recent logs are at the bottom, causing you to continuously scroll if you wish to see the most recent logs. This can be counteracted if you adjust the date and time of the logs you wish to view, by clicking on the blue timestamp in the top right corner of the logs.

If you successfully deploy a project to the DeepLens, but the device doesn't seem to be online, the CloudWatch logs are the place to look for any error you may be getting. Even if your model is not making any inferences, it should publish the error from the lamda function to the logs. 

## **Extension: Storing and Processing Data in S3**

----

Once we had our DeepLens successfully making inferences on thumbs up, the next step was to decide what we were going to do with the data recorded from the inferences. We decided to modify our lamda function so that when an inference is made the DeepLens publishes the inference image and a correspodning json file with the detection data to an S3 bucket. We could then process the data from this S3 bucket to store it in a form that can be easily interperated and analyzed. An updated verison of our lambda function and an outline of the changes we made from the originial version are included below. 

### **Configuring S3 Bucket**

In order to write any data to an S3 bucket from a lambda function you must make sure your s3 bucket and IAM roles have the correct permissions. First thing to check is if the role you used for the lambda fucntion has s3 access. To check this go to the IAM console, and select the role you used for your lambda function. For us this looked something like "AWSDeepLensLambdaRole". In the permissions for that role, make sure that you grant "Full3Access". 

### **Updated Lambda Function**

```python
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
def write_image_to_s3(img, output, time, file_name):
    # Create an IoT client for sending to messages to the cloud.
    client = greengrasssdk.client('iot-data')
    iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
    session = Session()
    s3 = session.create_client('s3')
    record = 'json/record_at_' +time+'.json'
    # You can contorl the size and quality of the image
    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
    _, jpg_data = cv2.imencode('.jpg', img, encode_param)
    response = s3.put_object(Body=jpg_data.tostring(),Bucket='thumbs-up-output',Key=file_name)
    response2 = s3.put_object(Body=json.dumps(output),Bucket='thumbs-up-output',Key=record)
    #client.publish(topic=iot_topic, payload="Response: {}".format(response))
    client.publish(topic=iot_topic, payload="Response: {}".format(response2))
    client.publish(topic=iot_topic, payload="Data pushed to S3")

    image_url = 'https://s3.amazonaws.com/thumbs-up-output/'+file_name
    return image_url

def greengrass_infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
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
        detection_threshold = 0.60
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
            for obj in parsed_inference_results[model_type]:
                # A dictionary to contain the info from an object detected
                object = {}
                if obj['prob'] > detection_threshold:
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
            img_file_name = 'images/image_at_'+frame_time+'.jpg'
            link = 'https://s3.amazonaws.com/thumbs-up-output/'+img_file_name
            detection["link_to_img"] = link
            # Upload to S3 to allow viewing the image in the browser, only if a detection was made
            if detection_made:
                image_url = write_image_to_s3(frame, detection, frame_time, img_file_name)
                client.publish(topic=iot_topic, payload='{{"img":"{}"}}'.format(image_url))
                client.publish(topic=iot_topic, payload=json.dumps(detection))
            else:
                client.publish(topic=iot_topic, payload="NO DETECTIONS MADE")
            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)
    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in object detection lambda: {}'.format(ex))

greengrass_infinite_infer_run()
```

The first thing to notice is the addition of the "write_image_to_s3" method. This method is built to upload both an image and a json record to a place in an s3 bucket that we specify. It wiill be called whenever an inference is made so that our inference data is stored in a place that is accessable for later. The parameter file_name is the name of the image file, while the name of the json file is defined within the method. Each file has a timestamp associated with it so we can determine at what time the detection was made. 

The rest of the changes come in greengrass_infinite_infer_run().The first change in this method is the addition of the "devices" dictionary. This is a dictionary built with the keys of unique deeplens ids that are returned by os.environ['AWS_IOT_THING_NAME']. The values are the much cleaner DeepLens names we set in registration. This dictionary is to help identify within the lambda function which Deeplens is making inferences if the project were to be deployed on multiple devices at once. We currently have three devices, thus our dictionary was defined as: 

```python
devices = {}
devices['deeplens_HoVip9KQTXiC3UFub47lJA'] = "Seattle01"
devices['deeplens_bmTWwitIRUi_mASjZASUHA'] = "Chicago01"
devices['deeplens_Hgs6kj_yQASF2x-3fOxCHA'] = "Chicago02"
```

The next change we made is to rebuild the way the inference data is structured. We thus defined a dictionary called "detection" which is meant to contain the data for all detections made within a single frame. The contents of the detection dictionary have the following structure:
```json
{
    "link_to_img": "https://s3.amazonaws.com/thumbs-up-output/images/image_at_20180816-183152.jpg", 
    "timestamp": "20180816-183152", 
    "objects": [
        {
            "index": 0, 
            "confidence": 0.998046875, 
            "object": "thumbs-up" }, 
        {
            "index": 1, 
            "confidence": 0.994140625, 
            "object": "thumbs-up" }
            ], 
    "device_id": "Seattle01"
}
```

"device_id" and "timestamp" are set just after the declaration of the dictionary, being reset for every frame of inference. The rest of the values are set within the if statement that determines if the detection made was above the threshold. Notice that they key for "objects" is actually a list of sub-dictionaries. These sub dictionaries contain an index to keep track of multiple detections in a frame, the confidence score for the detection, and the object detected. "link_to_img" is a link to the corresponding image for the dictionary of detections. We also added a boolean "detection_made" that is originally set to false at the beginning of every frame, but is set to true within the if statement determining if there was a detection above the threshold. We then write the images and data to s3 with the following statement: 

```python
    if detection_made:
                image_url = write_image_to_s3(frame, detection, frame_time, img_file_name)
                client.publish(topic=iot_topic, payload='{{"img":"{}"}}'.format(image_url))
                client.publish(topic=iot_topic, payload=json.dumps(detection))
```

Now you have an s3 bucket containing images and data for all the inferences from the DeepLens!

## **Double Extension: Tweeting Updates**

----

Our goal was to deploy the DeepLens to have users interact with the camera to give feedback about some prompted question. The next step in this workflow was to come up with a system to allow the user to see realtime responses from their interaction with the camera! Our solution to this was to create a twitter account and tweet the live updates. In order to do this we created another lambda function that, when triggered by upload to our output s3 bucket, sends out a tweet signifying an inference. This way the user could see a response from their interaction with the camera within seconds of the thumbs up event. 

### **Tweeting in Python**

Python has a library meant for tweeting called 'tweepy'. To tweet using tweepy, however, you first have to create a Twitter account and a Twitter App. Twitter policies have recently changed, so this process requires an application to fill out in order to obtain the Keys and Tokens you need to tweet. Once you set this up, however, the process of tweeting from a python file is pretty simple. First make sure that you have installed the tweepy module, 

```python
import tweepy

_CONSUMER_KEY = 'YOUR KEY'
_CONSUMER_SECRET = 'YOUR SECRET KEY'
_ACCESS_TOKEN = 'YOUR TOKEN'
_ACCESS_TOKEN_SECRET = 'YOUR SECRET TOKEN'

def tweet(CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_TOKEN_SECRET):
    
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    api.update_status("Hello Twitter!")
```

As long as you have tweepy installed and you have successfully retrieved the proper keys and tokens for your account, if you run this tweet method it should tweet the message "Hello Twitter!" from your registered account. The next step is to do this within an aws lambda function.

### **Configuruing the lamda function**

Unlike the previous lambda function for the deeplens object detection, we are creating this tweeting lambda function from scratch. The first thing to do is navigate to the lambda console in AWS and select "Create function" and chose to do it from scratch. Name the function whatever you wish, ours was "thumbs-up-tweeter", and chose the runtime to be python 2.7. For our role we used an existing role we had for basic lambda functionality. Whatever role you chose, be sure that it has permission for full Amazon S3 Access so it can access your S3 buckets for the output. Once you have all this configured choose "Create Function". 

Once you have the function created, the first thing to do is create the trigger by S3. Within your lambda function under the 'Designer' section you should see an option to add triggers. Select 'S3', and then move on to the configuration of your trigger. Select whatever bucket you wish to trigger the lambda function, but be aware that it may not let you have a single bucket trigger more than one lambda function. So make sure that you only have one bucket assigned to trigger one lambda function. We had our function triggered by any event but you can set it to "PUT" or "delete" or whatever you wish. The prefix and suffix fields can also be specified if you wish for the function to only be triggered by an upload to a certain folder.









```python
from __future__ import print_function

import json
import urllib
import tweepy
import random
import boto3
import ast
import datetime

_CONSUMER_KEY = "36QP9is9GEG4uuo3AY2te6Gmf"
_CONSUMER_SECRET = "wJJy0RVilvEl5teOsWwL05woyoOLwXSLaz4DuBq0oXnxHMRZXG"
_ACCESS_TOKEN = "1030514716903460867-aFhaQUuMpTFlQUq7Wsfys6csYAy7wJ"
_ACCESS_TOKEN_SECRET = "2JHt0oVvfkuqF2mMBW8IHglqWiwUNugKAq4mkZgc5CVr3"

messages = ["Coffee Time!", 
            "Having a great day, enjoying some great coffee!", 
            "This coffee deserves a thumbs up!"]

print('Loading function')

def get_device():
    bucket_name = 'thumbs-up-output-3'
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
    #print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    #bucket = event['Records'][0]['s3']['bucket']['name']
    #key = urllib.unquote_plus(event['Records'][0]['s3']['object']['key'].encode('utf8'))
    try:
        #response = s3.get_object(Bucket=bucket, Key=key)
        #print("CONTENT TYPE: " + response['ContentType'])
        tweet(_CONSUMER_KEY,_CONSUMER_SECRET,_ACCESS_TOKEN,_ACCESS_TOKEN_SECRET)
        #return response['ContentType']

    except Exception as e:
        print(e)
        #print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e

```

 

![image](https://github.com/bhinthorne/DeepLensObjectDetection/blob/master/results_1.jpg)


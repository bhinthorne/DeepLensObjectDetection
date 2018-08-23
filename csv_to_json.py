import csv
import json

#Just loads all file names into a list for comparison later
def loadFileNames(file_name):
    names = []
    with open(file_name) as rfile:
        reader = csv.DictReader(rfile)
        for row in reader:
            names.append(row['filename'])
    names.append('ENDFILE')
    #print(names)
    return names
#Returns true if next name is the same
def checkNextName(index, names):
    if names[index] == names[index+1]:
        return True
    else:
        return False
#Returns true if the previous name is the same
def checkPrevName(index, names):
    ##If it is the first file return false
    if index == 0:
        return False
    elif names[index] == names[index-1]:
        return True
    else:
        return False

def create_files(file_name, folder):
    names = loadFileNames(file_name)
    with open(file_name) as rfile:
        reader = csv.DictReader(rfile)
        counter = 0
        for row in reader:
            #doWrite = True
            #isFirstFile = True
            ##Reading Data
            filename = row['filename']
            width = int(row['width'])
            height = int(row['height'])
            depth = 3
            class_name = row['class']
            class_id = 0
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            box_width = abs(xmax-xmin)
            box_height = abs(ymax-ymin)
            ##Done Reading Data
           
            '''
                If the previous name is not the same, clear data and load the new data
            '''
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
            '''
                If the previous file name is the same, append annotation data
            '''
            if checkPrevName(counter, names):
                ##Appending Annotations
                data['annotations'].append({
                    'class_id':class_id,
                    'left':xmin,
                    'top':ymin,
                    'width':box_width,
                    'height':box_height
                })
            '''
                If the next name is not the same then we want to write the data
            '''    
            if not checkNextName(counter, names):
                if filename[-4:] == "jpeg":
                    new_name = filename[:-5]
                else:
                    new_name = filename[:-4]
                output_loc = 'json_files_test'+new_name+'.json'
                with open(output_loc, 'w') as outfile:
                    json.dump(data, outfile)
            counter += 1

#create_files('train_up_labels.csv', 'json_files/train')
create_files('thumb_labels.csv', 'json_files_test')
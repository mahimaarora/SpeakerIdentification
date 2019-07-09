
# IMPORTING LIBRARIES
import os
import shutil
from tqdm import tqdm
import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import time

# Adding Datapaths
datapath = 'data/Dataset'
extra = 'data/extras'
train_path = 'data/data_train'
test_path = 'data/data_test'
validation_path = 'data/data_validation'


# speakers, n = np.load('details.npy',allow_pickle=True)


# GETTING LABELS OF THE TRAINING DATA
def get_labels(path):
    labels = os.listdir(path)
    return labels, len(labels)
speakers, n = get_labels(datapath)

# Saving the names of speakers and their labels
details=[speakers,n]
np.save('data/details.npy', details)
print('Number of speakers: ',n)

# CREATING DIRECTORIES TO STORE DATA
data_sources = [train_path,test_path,validation_path,extra]
for source in data_sources:
    try:
        # Create target Directory
        os.mkdir(source)
        print("Directory " , source ,  " Created ") 
    except FileExistsError:
        print("Directory " , source ,  " already exists")


# If existing directories, then clearing it
for source in data_sources:
    fileList = os.listdir(source)
    for fileName in fileList:
        os.remove(source+"/"+fileName)


""" Creating the Train, Validation and Test csv files containing the audio filename and the speaker associated with that audio
    For each Speaker:
        Training Samples:   15
        Validation Samples:  5
        Testing Samples:     5
"""

def organise_data_csv(path):
    validation = [['SampleName','Speaker']]
    test = [['SampleName','Speaker']]
    train =[['SampleName','Speaker']]
    training_samples = [[] for _ in range(n)]
    for id_ in speakers:
        c = 0
        val = []
        test_id = []
        train_id = []
        remaining = []
        speaker_path = path+'/'+id_
        videos = os.listdir(speaker_path)
        
        random.shuffle(videos)
        for v in videos:
            files = os.listdir(speaker_path+'/'+v) 
            random.shuffle(files)
            if len(files) >= 3:
                if len(train_id) < 15:
                    c +=1
                    audio_clip = speaker_path+'/'+v+'/'+files.pop(0), train_path +'/'+id_+'.'+str(c)+'.wav'
                    shutil.move(audio_clip[0],audio_clip[1])
                    train_id.append([audio_clip[1],id_])
                if len(val) < 5:
                    c +=1
                    audio_clip = speaker_path+'/'+v+'/'+files.pop(0), validation_path +'/'+id_+'.'+str(c)+'.wav'
                    shutil.move(audio_clip[0],audio_clip[1])
                    val.append([audio_clip[1],id_])
                if len(test_id) < 5:
                    c +=1
                    audio_clip = speaker_path+'/'+v+'/'+files.pop(0), test_path +'/'+id_+'.'+str(c)+'.wav'
                    shutil.move(audio_clip[0],audio_clip[1])
                    test_id.append([audio_clip[1],id_])
            else:
                if len(train_id) < 15:
                    c +=1
                    audio_clip = speaker_path+'/'+v+'/'+files.pop(0), train_path +'/'+id_+'.'+str(c)+'.wav'
                    shutil.move(audio_clip[0],audio_clip[1])
                    train_id.append([audio_clip[1],id_])
            while (len(files) != 0):
                c +=1
                audio_clip = speaker_path+'/'+v+'/'+files.pop(0), extra +'/'+id_+'.'+str(c)+'.wav'
                shutil.move(audio_clip[0],audio_clip[1])
                remaining.append(audio_clip[1])
            shutil.rmtree(speaker_path+'/'+v)
                
        random.shuffle(remaining)
        while (len(train_id) < 15):
            if len(remaining) > 0:
                c+=1
                audio_clip = remaining.pop(0), train_path +'/'+id_+'.'+str(c)+'.wav'
                shutil.move(audio_clip[0],audio_clip[1])
                train_id.append([audio_clip[1],id_])
        while (len(test_id) < 5):
            if len(remaining) > 0:
                c+=1
                audio_clip = remaining.pop(0), test_path +'/'+id_+'.'+str(c)+'.wav'
                shutil.move(audio_clip[0],audio_clip[1])
                test_id.append([audio_clip[1],id_])
        while (len(val) < 5):
            if len(remaining) > 0:
                c+=1
                audio_clip = remaining.pop(0), validation_path +'/'+id_+'.'+str(c)+'.wav'
                shutil.move(audio_clip[0],audio_clip[1])
                val.append([audio_clip[1],id_])
                
            
        validation.extend(val)
        test.extend(test_id)
        training_samples[speakers.index(id_)].extend(train_id)
        train.extend(train_id)
        shutil.rmtree(path+'/'+id_)
        fileList = os.listdir(extra)
        for fileName in fileList:
             os.remove(extra+"/"+fileName)
        
        
    with open('data/train.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(train)
    csvFile.close()    
    
    with open('data/test.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(test)
    csvFile.close()
    
    with open('data/validation.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(validation)
    csvFile.close() 
    
    return training_samples,test, validation
start = time.time()      
training_samples ,testing_samples, validation_samples = organise_data_csv(datapath)
end = time.time()
print('Time for',n,'speakers', (end-start)/60,'mins')

# Saving the training samples speaker-wise 
np.save('data/training_samples.npy', training_samples)

# training_samples = np.load('training_samples.npy', allow_pickle=True).tolist()

sum([len(i) for i in training_samples]) == 15*n






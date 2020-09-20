import subprocess
import shutil

# In[ ]:

##############################################
#        ADJUST THOSE DIRECTORIES            #
##############################################

# Note: Labels have to be in the same directory as images
datasetDirectory = "/kaggle/input/preparepandadataset/data"
workingDirectory = "/kaggle/darknet"
outputDirectory = "/kaggle/working/weights"
pretrainedCustomFile = "/kaggle/input/trainpandadatasetyolov4darknet/weights/custom-yolov4-detector_last.weights"


# In[ ]:


import os
imageDirectory = os.path.join(datasetDirectory, "images")
trainTXT = os.path.join(datasetDirectory, "train.txt")
valTXT = os.path.join(datasetDirectory, "val.txt")
classesTXT = os.path.join(datasetDirectory, "annos.names")

if os.path.exists(imageDirectory) and os.path.exists(trainTXT) and os.path.exists(valTXT) and os.path.exists(classesTXT):
    print("Dataset files found!")
    print(f"Images: {int(len(os.listdir(imageDirectory)) / 2)}")


# ## Install CUDA

# In[ ]:


subprocess.run(["apt-get", "install", "-y", "nvidia-cuda-toolkit", "nvidia-cuda-dev"])


# ## Check CUDA

# In[ ]:


subprocess.run(['/usr/local/cuda/bin/nvcc', '--version'])
subprocess.run(['nvidia-smi'])


# In[ ]:


# Change the number depending on what GPU is listed above, under NVIDIA-SMI > Name.
# Tesla K80: 30
# Tesla P100: 60
# Tesla T4: 75
subprocess.run(['env', 'compute_capability=60'])


# ## Install OpenCV

# In[ ]:


subprocess.run(['apt-get', 'install', 'libopencv-dev', '-y'])


# ## SetUp Darknet

# In[ ]:


subprocess.run(['git', 'clone', 'https://github.com/AlexeyAB/darknet.git', workingDirectory])


# In[ ]:


customCfg = os.path.join(workingDirectory, "cfg/yolov4-custom.cfg")
objCfg = os.path.join(workingDirectory, "cfg/yolo-obj.cfg")


# In[ ]:

shutil.copy(customCfg, objCfg)


# In[ ]:


subprocess.run(['conda', 'install', 'gdown', '-y'])


# In[ ]:

subprocess.run(['gdown', 'https://drive.google.com/uc?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp'])
shutil.copy("yolov4.conv.137", workingDirectory)


# In[ ]:


subprocess.run(["sed", "-i" "'s/OPENCV=0/OPENCV=1/g'", os.path.join(workingDirectory, "Makefile")])
subprocess.run(["sed", "-i" "'s/GPU=0/GPU=1/g'", os.path.join(workingDirectory, "Makefile")])
subprocess.run(["sed", "-i" "'s/CUDNN=0/CUDNN=1/g'", os.path.join(workingDirectory, "Makefile")])
subprocess.run(["sed", "-i", "'s/ARCH= -gencode arch=compute_30,code=sm_30 \\\\/ARCH= -gencode arch=compute_70,code=sm_70 \\\\/g'", os.path.join(workingDirectory, "Makefile")])
#!sed -i 's/NVCC=nvcc/NVCC=\/usr\/local\/cuda\/bin\/nvcc/g' Makefile
subprocess.run(['make', workingDirectory])


# ### Setup Custom Dataset

# In[ ]:


subprocess.run(['mkdir', outputDirectory])


# In[ ]:


def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

num_classes = file_len(classesTXT)
print("writing config for a custom YOLOv4 detector detecting number of classes: " + str(num_classes))

with open(os.path.join(workingDirectory, 'data/obj.data'), 'w') as out:
  out.write(f"classes = {num_classes}\n")
  out.write(f'train = {trainTXT}\n')
  out.write(f'valid = {valTXT}\n')
  out.write(f"names = {classesTXT}\n")
  out.write(f'backup = {outputDirectory}')


# In[ ]:


subprocess.run(['wget', "-O", os.path.join(workingDirectory, "cfg/yolov4-custom1.cfg"), "https://raw.githubusercontent.com/roboflow-ai/darknet/master/cfg/yolov4-custom1.cfg"])
subprocess.run(['wget', "-O", os.path.join(workingDirectory, "cfg/yolov4-custom2.cfg"), "https://raw.githubusercontent.com/roboflow-ai/darknet/master/cfg/yolov4-custom2.cfg"])
subprocess.run(['wget', "-O", os.path.join(workingDirectory, "cfg/yolov4-custom3.cfg"), "https://raw.githubusercontent.com/roboflow-ai/darknet/master/cfg/yolov4-custom3.cfg"])
subprocess.run(['wget', "-O", os.path.join(workingDirectory, "cfg/yolov4-custom4.cfg"), "https://raw.githubusercontent.com/roboflow-ai/darknet/master/cfg/yolov4-custom4.cfg"])
subprocess.run(['wget', "-O", os.path.join(workingDirectory, "cfg/yolov4-custom5.cfg"), "https://raw.githubusercontent.com/roboflow-ai/darknet/master/cfg/yolov4-custom5.cfg"])


# In[ ]:


import os

#Instructions from the darknet repo
#change line max_batches to (classes*2000 but not less than number of training images, and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
#change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
if os.path.exists(os.path.join(workingDirectory, 'cfg/custom-yolov4-detector.cfg')): os.remove(os.path.join(workingDirectory, 'cfg/custom-yolov4-detector.cfg'))


with open(os.path.join(workingDirectory, 'cfg/custom-yolov4-detector.cfg'), 'a') as f:
  f.write('[net]' + '\n')
  f.write('batch=64' + '\n')
  #####smaller subdivisions help the GPU run faster. 12 is optimal, but you might need to change to 24,36,64####
  f.write('subdivisions=32' + '\n')
  f.write('width=416' + '\n')
  f.write('height=416' + '\n')
  f.write('channels=3' + '\n')
  f.write('momentum=0.949' + '\n')
  f.write('decay=0.0005' + '\n')
  f.write('angle=0' + '\n')
  f.write('saturation = 1.5' + '\n')
  f.write('exposure = 1.5' + '\n')
  f.write('hue = .1' + '\n')
  f.write('\n')
  f.write('learning_rate=0.001' + '\n')
  f.write('burn_in=1000' + '\n')
  ######you can adjust up and down to change training time#####
  ##Darknet does iterations with batches, not epochs####
  max_batches = max(num_classes*2000, 6000)
  #max_batches = 2000
  f.write('max_batches=' + str(max_batches) + '\n')
  f.write('policy=steps' + '\n')
  steps1 = .8 * max_batches
  steps2 = .9 * max_batches
  f.write('steps='+str(steps1)+','+str(steps2) + '\n')

#Instructions from the darknet repo
#change line classes=80 to your number of objects in each of 3 [yolo]-layers:
#change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.

  with open(os.path.join(workingDirectory, 'cfg/yolov4-custom2.cfg'), 'r') as f2:
    content = f2.readlines()
    for line in content:
      f.write(line)    
    num_filters = (num_classes + 5) * 3
    f.write('filters='+str(num_filters) + '\n')
    f.write('activation=linear')
    f.write('\n')
    f.write('\n')
    f.write('[yolo]' + '\n')
    f.write('mask = 0,1,2' + '\n')
    f.write('anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401' + '\n')
    f.write('classes=' + str(num_classes) + '\n')

  with open(os.path.join(workingDirectory, 'cfg/yolov4-custom3.cfg'), 'r') as f3:
    content = f3.readlines()
    for line in content:
      f.write(line)    
    num_filters = (num_classes + 5) * 3
    f.write('filters='+str(num_filters) + '\n')
    f.write('activation=linear')
    f.write('\n')
    f.write('\n')
    f.write('[yolo]' + '\n')
    f.write('mask = 3,4,5' + '\n')
    f.write('anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401' + '\n')
    f.write('classes=' + str(num_classes) + '\n')

  with open(os.path.join(workingDirectory, 'cfg/yolov4-custom4.cfg'), 'r') as f4:
    content = f4.readlines()
    for line in content:
      f.write(line)    
    num_filters = (num_classes + 5) * 3
    f.write('filters='+str(num_filters) + '\n')
    f.write('activation=linear')
    f.write('\n')
    f.write('\n')
    f.write('[yolo]' + '\n')
    f.write('mask = 6,7,8' + '\n')
    f.write('anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401' + '\n')
    f.write('classes=' + str(num_classes) + '\n')
    
  with open(os.path.join(workingDirectory, 'cfg/yolov4-custom5.cfg'), 'r') as f5:
    content = f5.readlines()
    for line in content:
      f.write(line)

print("file is written!") 
print("Amount of batches: " + str(max_batches))


# In[ ]:


subprocess.run(['cp', os.path.join(workingDirectory, 'cfg/custom-yolov4-detector.cfg'), outputDirectory])


# ## Train

# In[ ]:


weightsFile = os.path.join(workingDirectory, "yolov4.conv.137")
if os.path.exists(pretrainedCustomFile):
    weightsFile = pretrainedCustomFile
    
print(f"Using weights file: {weightsFile}")


# In[ ]:


# START TRAIN
subprocess.run(["chmod", "+x", os.path.join(workingDirectory, "build", 'darknet')])
# Warning: This call may cause an exception!
subprocess.run([os.path.join(workingDirectory, "build", 'darknet'), 'detector', 'train', os.path.join(workingDirectory, 'data/obj.data'), os.path.join(workingDirectory, 'cfg/custom-yolov4-detector.cfg'), weightsFile, '-dont_show'])


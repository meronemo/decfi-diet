!pip install ultralytics
!pip install roboflow

from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="FOuEdvU2HLPz5zXhQJHe")
project = rf.workspace("sadge").project("korean-food_yolov5-86zan") #korean food image dataset
dataset = project.version(1).download("yolov8")  #yolov8 format

#path to data.yaml file
data_yaml = dataset.location + "/data.yaml"

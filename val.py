import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'E:\YOLOv11-RGBT-master\YOLOv11-RGBT-master\runs\Drone(v9)\yolo11s-RGBT-midfusion-all2\weights\best.pt')
    model.val(data='E:/YOLOv11-RGBT-master/YOLOv11-RGBT-master/ultralytics/cfg/datasets/Dronevhicle.yaml',
              split='val',
              imgsz=640,
              batch=32,
              use_simotm="RGBT",
              channels=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/Drone_v8_test_r20',
              name='LLVIP_r20-yolov8n-no_pretrained',
              )
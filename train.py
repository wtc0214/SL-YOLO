import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'F:\YOLOv11-RGBT-master\ultralytics\cfg\models\v8\yolov8.yaml')
    #model.load('F:/YOLOv11-RGBT-master/best.pt') # loading pretrain weights
    model.train(data="F:/YOLOv11-RGBT-master/ultralytics/cfg/datasets/ASL.yaml",
                cache=False,
                imgsz=640,
                epochs=900,
                batch=32,
                close_mosaic=5,
                workers=4,
                device='0',
                optimizer='SGD',  # using SGD
                patience=20,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGB",
                channels=3,
                project='COCO_yixue_',
                name='detection_ex-all',
                )
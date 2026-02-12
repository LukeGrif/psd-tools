(marine-ml) luke-griffin@luke-griffin-Precision-7680:~/psd-tools$ yolo detect train data=dataset/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640

(marine-ml) luke-griffin@luke-griffin-Precision-7680:~/psd-tools$ yolo detect predict   model=runs/detect/train/weights/best.pt   source=image_to_test/0126_1_ADS_0934.png   imgsz=832   conf=0.05

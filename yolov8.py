from ultralytics import YOLO

if __name__ == '__main__':
    #Train
    # Use a YAML configuration file to create the model and train it from scratch.
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-RepC3.yaml')#our
    model.train(cfg="ultralytics/cfg/default.yaml",data="ultralytics/datasets/traffic.yaml",
                epochs=600,batch=32,workers=4)

    # validation
    model = YOLO('runs/weights/best.pt')
    model.val(**{'data': 'ultralytics/datasets/traffic.yaml'})
    #
    # # inference
    model = YOLO('runs/weights/best.pt')
    model.predict(source='ultralytics/datasets/traffic/test/images/', **{'save': True,'visualize': True})

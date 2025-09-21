

<details open>
<summary>Install</summary>

Pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
pip install -r requirements.txt
```



<details open>
<summary>Usage</summary>
YOLOv8 may also be used directly in a Python environment:
​    

#### Train


```python
from ultralytics import YOLO

if __name__ == '__main__':
    #Train
    # Use a YAML configuration file to create the model and train it from scratch.
    model = YOLO('ultralytics/cfg/models/v8/yolov8n-RepC3.yaml')#our
    model.train(cfg="ultralytics/cfg/default.yaml",data="ultralytics/datasets/traffic.yaml",
                epochs=600,batch=32,workers=4)

```

#### validation

```
# validation
model = YOLO('runs/weights/best.pt')
model.val(**{'data': 'ultralytics/datasets/traffic.yaml'})
```

#### inference

```
# # inference
model = YOLO('runs/weights/best.pt')
model.predict(source='ultralytics/datasets/traffic/test/images/', **{'save': True,'visualize': True})
```



### datasets

**The data supporting this study's findings are available at GitHub and Zenodo:**
**-** https://github.com/yuhongxia21/CCTSDB2021/releases/tag/V1.0.0

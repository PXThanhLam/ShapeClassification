# ShapeClassification
Require <br/>
* torch >= 1.6 <br/>
* torchvision compatible with torch <br/>
 
Usage
* To extarct fetaure
 ```python:
 from utils import ShapeRecogModel
 model = ShapeRecogModel(model_path = 'path/to/your_model')
 img_embd = model.extract_feature(image_path = test_path)
 # or
 # img_embd = model.extract_feature(image = your_numpy_image)
 ```
Model is store at '/data/lamphan/ShapeClassification/checkpoint_0005.pth.tar', ubntu69


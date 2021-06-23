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
Model is store at '/data/lamphan/ShapeClassification/checkpoint_0005.pth.tar', ubuntu69

# Training
* Code is taken from https://github.com/facebookresearch/moco , with some minor change on augmentationand datal loading part.
* Training command :
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --moco-t 0.2 --aug-plus --cos \
  [your imagenet-folder with train and val folders]
```


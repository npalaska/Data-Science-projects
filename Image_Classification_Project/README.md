
## Common Use case for train.py:
*python train.py flowers --save_dir "checkpoint.pth" --hidden_units 512 --arch "vgg16" --learning_rate 0.0001 --epochs 10*
*python train.py flowers --save_dir "checkpoint.pth" --hidden_units 512,256 --arch "vgg16" --learning_rate 0.01 --epochs 10*

## Common Use case for predict.py:
*python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 3*

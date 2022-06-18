# RetinexCFNet

RetinexCFNet for Lowlight Image Enhancement or Image Dehazing.

## Requirements

```shell
python 3.7
onnxruntime
opencv-python
```

## Prediction

```
cd RetinexCFNet/
pip install -r requirements.txt
```
You can download weights from [Google Drive](https://drive.google.com/file/d/1VJR9YebFpUpEdUzk4tLodwGBK5wPLF8w/view?usp=sharing).

For lowlight image enhancement:

```
python infer_demo_onnx.py --model_path weights/RetinexCFNet.onnx --image_path test_images/ --save_dir img_prediction/result_RetinexCFNet
```

For image dehazing:

```
python infer_demo_onnx.py --model_path weights/RetinexCFNet_dehaze.onnx --image_path dehaze_images/test_data/ --save_dir img_prediction/result_RetinexCFNet_dehaze
```

Note: the predi

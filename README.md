# RetinexCFNet

RetinexCFNet for Lowlight Image Enhancement or Image Dehazing.

## Requirements

```shell
python 3.7
onnxruntime
opencv-python
```

Install onnxruntime-gpu instead of onnxruntime to support GPU prediction.

```
pip install onnxruntime-gpu
```

## Prediction

```
cd RetinexCFNet/
pip install -r requirements.txt
```
You can download weights from [Google Drive](https://drive.google.com/file/d/1VJR9YebFpUpEdUzk4tLodwGBK5wPLF8w/view?usp=sharing).

***For Lowlight Image Enhancement:***

```
python infer_demo_onnx.py --model_path weights/RetinexCFNet.onnx --image_path lowlight_images/test_data/ --save_dir results/result_RetinexCFNet
```

You can also download the prediction results of LOL dataset and LIME dataset from [Google Drive](https://drive.google.com/drive/folders/1bcAglKl1HAsv1BZzbSqH_AP86qcYrNwU?usp=sharing).

***For Image Dehazing:***

```
python infer_demo_onnx.py --model_path weights/RetinexCFNet_dehaze.onnx --image_path dehaze_images/test_data/ --save_dir results/result_RetinexCFNet_dehaze
```

You can also download the prediction results of O-HAZE dataset, I-HAZE dataset and NH-HAZE dataset from [Google Drive](https://drive.google.com/drive/folders/1bcAglKl1HAsv1BZzbSqH_AP86qcYrNwU?usp=sharing).

## Performance Evaluation

***Lowlight Image Enhancement***

Quantitative comparison on LOL dataset about MSE, PSNR, SSIM, LPIPS and DISTS. The highlighted in bold represents the best results.

![image-20220618232530140](/home/yzw/.config/Typora/typora-user-images/image-20220618232530140.png)

***Image Dehazing***

Quantitative comparison on O-HAZE dataset, I-HAZE dataset and NH-HAZE dataset about PSNR and SSIM. The highlighted in bold represents the best results.

![image-20220618233706718](/home/yzw/.config/Typora/typora-user-images/image-20220618233706718.png)

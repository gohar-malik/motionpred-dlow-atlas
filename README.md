# motionpred-dlow-atlas
This repository provides code for the ECE 740 course project, 3D Human Motion Prediction on Edge. In this project, we develop a 3d human motion prediction model for the Huawei Atlas 200 DK. 

## Datasets
Please follow the data preprocessing steps ([DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)) inside the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo. Place the prepocessed data data_3d_h36m.npz (Human3.6M) and data_3d_humaneva15.npz (HumanEva-I) under the data folder.

## Pretrained models
The pretrained models are provided in the `checkpoints` dir.

## Inference on Huawei Atlas 200 DK
To perform inference using the model on the device, run the following script.

```bash
python3 eval_om.py
```

There are three quantization configurations used for the model deployment. These configurations are explained in the below table. To use any of these configuratios, please modify the model argument in line 59 of the `eval_om.py` file, [here](https://github.com/gohar-malik/motionpred-dlow-atlas/blob/c0cb23a2ed4ca69f6b086c91630b165ec8786b2d/eval_om.py#L59)

| Name        | Description           | 
| ------------- |:-------------:| 
| force_fp32      | All the weights are quantized to float32 | 
| force_fp32      | Weights of some important layers are quantized to float32, others to float16      |   
| allow_fp32_to_fp16 |  All the weights are quantized to float16      |   

## Acknowledgement
The model used in this project and the code is based on the [DLow](https://github.com/Khrylx/DLow) repo. Special thanks to the authors.

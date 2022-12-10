# motionpred-dlow-atlas
This repository provides code for the ECE 740 course project, Low-Latency and Power-Efficient 3D Human Motion Prediction for Edge Applications. In this project, we develop a 3d human motion prediction model for the Huawei Atlas 200 DK. 

## Datasets
Prepared Human3.6M dataset can be downloaded directly from this [GDrive link](https://drive.google.com/file/d/1b1WztVxqjvPs0aSUdrnbo5QiovXT8y2X/view?usp=share_link). Place the data file,  `data_3d_h36m.npz` (Human3.6M) under the `data` folder. 
If you want to see the how this file was prepared, please follow the steps for Human3.6M from ([DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)) inside the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo. 

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
| force_fp32 | All layers quantized to 32 bit floating point | 
| mix_precision | Only initial layers quantized to 16 bit, rest quantized to 32 bit |   
|force_fp16 | All layers quantized to 16 bit floating point |   

## Acknowledgement
The model used in this project and the code is based on the [DLow](https://github.com/Khrylx/DLow) repo. Special thanks to the authors.

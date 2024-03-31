
<!-- 关于本项目 -->
# MvDeFormer
Torch implementation for achieving accurate recognition of similar gestures under high-strength human activity interference, where a Multi-view De-interference Transformer (MvDeFormer) network is proposed as follows:
<img src="images/MvDeFormer.png" alt="MvDeFormer-logo" width="833" height="475">

For indoor scenarios, human activities such as walking, running, or other unrelated behaviors very likely appear around the user, which would seriously affect the accuracy of gesture recognition. What's more, similar gestures are very difficult to distinguish with same number of segments or partially identical movements. So, in our experiments,  we considered the two issues mentioned above. Specifically, to deal with the strong human activity interference, we design a DeFormer module to capture the useful gesture features by learning different patterns between gestures and interference, thereby reducing the impact of interference. Then, we develop a hierarchical multi-view fusion module to first extract the enhanced features within each view, and effectively fuse them across various views for final recognition. The images of seven similar gestures and the recognition results by MvDeFormer are as follows:
<p align="center">
  <img src="images/gesture.png" alt="Gesture" width="440" style="vertical-align: middle;"/>
  <img src="images/ConfusionMatrix.png" alt="ConfusionMatrix" width="280" style="vertical-align: middle;"/>
</p>

<p align="right">(<a href="#top">Reture to top</a>)</p>



## Setup

### Prerequisites
- Linux or Windows
- NVIDIA GPU + CUDA CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

### Getting Started
- Install torch and dependencies from https://github.com/torch/distro
- Install torch packages tqdm, sklearn, einops and linformer
```bash
pip install tqdm
pip install -U scikit-learn
pip install einops
pip install linformer
```
### Dataset Preparation

To get started, please follow these steps:

1. **Store the Dataset**: Place your dataset in the `data_picture` directory. Ensure that all the data you intend to use is properly organized within this folder.

2. **Splitting the Dataset**: Run the `process_radar_data` script to split your dataset into training, validation, and test sets with a ratio of 7:1:2. This will help in evaluating the model's performance accurately.

Open a terminal or command prompt and navigate to the directory containing the `process_radar_data` script. Run the following command:

```bash
python process_radar_data.py
```
## Train

**Run the Training Script**: Execute the `train.py` script from your terminal or command prompt to start the training process. You can run the script using the following command:
```bash
   python train.py
```
Model Parameters: After the training process is complete, the trained model parameters will be saved in the model_parameter directory. Ensure this directory exists or the script will create it for you.

## Test

After training your model, you can test its performance using the `test.py` script. Follow these steps to conduct the test:

**Run the Testing Script**: Execute `test.py` from your terminal or command prompt to begin the testing process. Use the following command:
 ```bash
   python test.py
```
Model Loading: The script will automatically load the model parameters from the model_parameter directory. Ensure that your trained model parameters are correctly saved in this directory before running the test.

Test Output: Upon completion, the script will output the loss and accuracy of the model on the test dataset. This information will help you evaluate the model's performance.

## Citation
If you use this code for your research, please cite our paper.

<p align="right">(<a href="#top">Reture to top</a>)</p>




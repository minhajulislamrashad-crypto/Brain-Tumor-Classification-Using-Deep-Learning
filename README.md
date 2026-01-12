##### **# Brain Tumor Classification Using Deep Learning**



###### **## Project Overview**



This project implements a lightweight Convolutional Neural Network (CNN) to classify brain MRI images into four classes: glioma, meningioma, pituitary tumor, and no tumor. The pipeline includes dataset loading and preprocessing, model training, evaluation, and a simple single-sample prediction/visualization.



The goal is to demonstrate a correct and reproducible application of supervised CNN-based classification, not to propose a novel architecture.



\## Dataset

The project uses the following Kaggle dataset:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset



Expected folder structure:





data/

&nbsp;  Training/

&nbsp;     glioma/

&nbsp;     meningioma/

&nbsp;     pituitary/

&nbsp;     notumor/

&nbsp;  Testing/

&nbsp;     glioma/

&nbsp;     meningioma/

&nbsp;     pituitary/

&nbsp;     notumor/





If the dataset is not included in the submission (often excluded due to size), download it and place it into the `data/` directory exactly as shown.



\## Environment Requirements

\- Python 3.9+ (tested with Python 3.11)

\- PyTorch

\- torchvision

\- numpy

\- matplotlib

\- scikit-learn (optional, if used for extra metrics)

\- CUDA-capable NVIDIA GPU (optional)



###### **## Setup (Windows)**



1\) Create a virtual environment:

```powershell

python -m venv .venv



2\. Activate Virtual Environment

.venv\\Scripts\\activate



3\. Install Dependencies

pip install -r requirements.txt



###### **Project Structure:**



Brain\_Tumor\_ML/

&nbsp; data/                  # dataset directory (may be excluded from zip if too large)

&nbsp; outputs/               # saved model weights and optional logs

&nbsp; src/

&nbsp;   \_\_init\_\_.py

&nbsp;   data\_loader.py       # dataloaders + preprocessing transforms

&nbsp;   model.py             # CNN model definition (create\_model)

&nbsp;   optimizer.py         # loss function + optimizer setup

&nbsp;   train.py             # training loop (train\_model)

&nbsp;   evaluate.py          # evaluation loop (evaluate\_model)

&nbsp;   predict.py           # prediction + visualization helpers

&nbsp; Main.ipynb             # main notebook that runs the pipeline end-to-end

&nbsp; requirements.txt

&nbsp; README.md

&nbsp; pyproject.toml         # optional (not required to run if using requirements.txt)





###### **How to Run:**



Start Jupyter:



powershell:

jupyter lab

>>

Open and run:

Main.ipynb



**This notebook:**



* loads the dataset using src/data\_loader.py
* 
* creates the CNN using src/model.py
* 
* sets loss/optimizer using src/optimizer.py
* 
* trains using src/train.py and saves outputs/model.pt
* 
* evaluates using src/evaluate.py
* 
* runs one sample prediction using src/predict.py



**Outputs:**



* outputs/model.pt : saved model weights after training
* 
* Console logs: training loss per epoch, test loss, and test accuracy
* 
* Optional: a displayed sample MRI with predicted class and confidence



**Notes**



GPU acceleration is optional. If CUDA is available and a CUDA-enabled PyTorch is installed, training will run faster.



This project does not include external clinical validation.



Transformers and reinforcement learning are not used because this task is standard supervised image classification.



###### **License**



Academic and educational use only.


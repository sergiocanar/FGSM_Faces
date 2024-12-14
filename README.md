# Project ML for Engineering - Group 1

## Members
- Juan Camilo Rojas Hernández (202121526)  
- Sergio Andrés Canar Lozano (202020383)  
- Wilman Sánchez Hernández (202116779)  

---

## Adversarial Machine Learning: **Fast Gradient Sign Method (FGSM) in Face Recognition**

---

## Repository Preparation

To use this repository, you need to install the required dependencies. We recommend creating an [Anaconda](https://www.anaconda.com/download/success) environment and installing the necessary libraries within it. Follow the steps below:

1. Create the environment (here, named `fgsm_faces`):
   ```bash
   conda create -n fgsm_faces python=3.10.2


2. Activate the environment:

```bash
  conda activate fgsm_faces

```

3. Install the libraries in ``requirements.txt``.

```bash
  pip install -r requirements.txt

```

4. Download the data and model weights to use for the **FGSM**. So, the first thing you need to do is download the [data](https://drive.google.com/drive/folders/1cZ_pz10iFS_ydMxQ8kpAqV5EbJ4wmbF_?usp=sharing) and unzip it so that you have the following folder structure:

```
FGSM_faces/data/
 │
 ├───face_recognition
 │ ├───Faces
 │ │ └──└─Akshay Kumar_0.jpg
 │ │ └───Akshay Kumar_1.jpg
 │ │ └──...
 │ ├───Original Images
 │ │ └───Akshay Kumar
 │ │ └└──Alexandra Daddario
 │ │ └──...
 │ ├───data_txt.txt
 │ │ ├───dataset_faces.csv
 │ ├───Dataset.csv
```

Now for the weights of the models used:

```
FGSM_faces/
 │
 ├────models
 │ └───resnet_celeb_faces.pth
 │ └└──resnet18_celeb_faces.pth
 │ └└───resnet101_celeb_faces.pth

```

**All models were trained in a GPU Nvidia GeForce RTX 4060 de 8 Gb**.

## Use the repository

The file containing the **FGSM** implementation is `Group1_FGSM.ipynb`. By running this file, you will find a detailed description of how this *Adversarial Machine Learning* method works, as well as an overview of the importance of this area of research in *Machine Learning* and *Artificial Intelligence*.

To perform the FGSM attack with a different model, you must modify two variables: `model` and `pretrained_model`. These variables define the model that will be subjected to the attack. You can change them according to the model you want to test, but make sure that the models and their weights match to avoid errors when loading the weights. Below is the section of the code where you will need to make these changes.

<div style=‘text-align: center;’>
  <img src="./resources/weights.png" alt=‘pesos’ width=‘1000’>
</div>

All set! You're free to change the model and experiment with any model you want. 

### Results

Here are the results we obtained for the three models we tested, showing the effect of epsilon ($\epsilon$) on the model's performance. This highlights the importance of studying the vulnerability of ML and AI models.

<div style="text-align: center;">
  <img src="./resources/results_subplots.png" alt="weights" width="1000">
</div>

Now, these are the qualitative results of our project, where we can see that a minimal perturbation in the input images causes the model to produce incorrect predictions.

<div style="text-align: center;">
  <img src="./results/examples_resnet18.png" alt="weights" width="1000">
</div>

### Analysis

As we can observe, in all three models, as ϵ increases, accuracy drops dramatically, starting with low values of ϵ. This indicates that the models are sensitive even to small perturbations in the input data. In fact, looking at the qualitative results, we can see that with an ϵ of 0.0001, misclassifications of faces begin to occur, and these increase as epsilon grows. This low tolerance of the models to small perturbations makes them much less effective against adversarial attacks or noise.

In fact, it is important to note that all the evaluated models exhibit a high vulnerability to adversarial perturbations, as even the most complex architectures, such as ResNet-101, do not show significant advantages over ResNet-18 or the base version in terms of resistance to these attacks.

This brief analysis underscores the importance of implementing strategies to enhance the robustness of models against adversarial attacks, especially since many of these models are used in critical real-life applications, such as disease detection in medical imaging, autonomous driving, security in surveillance systems, and financial data processing.



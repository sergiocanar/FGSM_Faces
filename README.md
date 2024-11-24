# Proyecto ML for Engineering - Grupo 1

## Integrantes
- Juan Camilo Rojas Hernández (202121526)  
- Sergio Andrés Canar Lozano (202020383)  
- Wilman Sánchez Hernández (202116779)  

---

# Machine Learning Adversarial: **Fast Gradient Sign Method (FGSM) en el Reconocimiento de Rostros**

## Preparación del repositorio

Para usar este repositorio, es necesario que descargues los datos y los pesos del modelo a utilizar para el **FGSM**. Entonces, lo primero que deberás hacer es descargar los [datos](https://drive.google.com/drive/folders/1cZ_pz10iFS_ydMxQ8kpAqV5EbJ4wmbF_?usp=sharing) y descomprimirlos para que tengas la siguiente estructura de carpetas:

```
FGSM_faces/data/
 │
 ├───face_recognition
 │      ├──Faces
 │      │      └──Akshay Kumar_0.jpg
 │      │      └──Akshay Kumar_1.jpg
 │      │      └──...
 │      ├──Original Images
 │      │      └──Akshay Kumar
 │      │      └──Alexandra Daddario
 │      │      └──...
 │      ├──data_txt.txt
 │      ├──dataset_faces.csv
 │      ├──Dataset.csv
```

Ahora para los pesos de los modelos usados:

```
FGSM_faces/
 │
 ├───models
 │      └──resnet_celeb_faces.pth
 │      └──resnet18_celeb_faces.pth
 │      └──resnet101_celeb_faces.pth

```
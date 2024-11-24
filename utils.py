import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(data_csv, output_dir):
    df = pd.read_csv(data_csv, sep=',', header=None)
    df = df.drop([0], axis=0)  # Elimina la prima riga

    unique_labels = np.unique(df[1].values)
    print(len(unique_labels))
    
    rows = [] # Dictionary list
    
    for _, row in df.iterrows():
        img = row[0]
        label = np.where(unique_labels == row[1])[0][0]
        rows.append({'img': img, 'label': label})
    
    new_df = pd.DataFrame(rows)
        
    new_df.to_csv(output_dir, sep=',', index=False, header=False)
    
    print('Identity file created at: {}'.format(output_dir))
    
def subplots_img(img_lt, title, subtitle_lt,save_path):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    fig.suptitle(title)
    for i, img in enumerate(img_lt):
        
        if 'accuracies' in img:
            img = plt.imread(os.path.join(results_dir, img))
            ax[i].imshow(img)
            ax[i].set_title(subtitle_lt[i])
            ax[i].axis('off')
        else:
            continue
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    results_dir = os.path.join(this_dir, 'results')
    data_csv = os.path.join(data_dir, 'face_recognition', 'Dataset.csv')
    output_dir = os.path.join(data_dir, 'face_recognition', 'dataset_faces.csv')
    output_dir_img = os.path.join(this_dir, 'resources', 'results_subplots.png')
    img_lt = os.listdir(results_dir)
    subtitle_lt = ['ResNet', 'ResNet-18', 'ResNet-101']
    
    get_data(data_csv, output_dir)
    subplots_img(img_lt, 'Resulados de FGSM en diferentes modelos', subtitle_lt,output_dir_img)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
def get_data(data_csv, output_dir):
    try:
        df = pd.read_csv(data_csv, sep=',', header=None)
        df = df.drop([0], axis=0)  # Drop the first row if it contains headers
        unique_labels = np.unique(df[1].values)
        
        rows = []  # Dictionary list for the new DataFrame
        for _, row in df.iterrows():
            img = row[0]
            label = np.where(unique_labels == row[1])[0][0]
            rows.append({'img': img, 'label': label})
        
        new_df = pd.DataFrame(rows)
        new_df.to_csv(output_dir, sep=',', index=False, header=False)
        print(f'Identity file created at: {output_dir}')
    except Exception as e:
        print(f"Error processing data: {e}")

def plot_dataset_samples(n_samples, images_path_lt, labels_lt, save_path):
    try:
        rows = int(np.sqrt(n_samples))
        cols = n_samples // rows
        fig, ax = plt.subplots(rows, cols, figsize=(10, 10))
        fig.suptitle('Dataset Samples')
        
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                img = plt.imread(images_path_lt[idx])
                ax[i, j].imshow(img)
                ax[i, j].set_title(labels_lt[idx])
                ax[i, j].axis('off')
        
        plt.savefig(save_path)
        plt.close()
        print(f'Sample plot saved at: {save_path}')
    except Exception as e:
        print(f"Error plotting dataset samples: {e}")
        
def get_dataset_distribution(dataset_csv, output_dir):
    try:
        df = pd.read_csv(dataset_csv, sep=',', header=None)
        df = df.drop([0], axis=0)  # Drop the first row if it contains headers
        unique_labels = np.unique(df[1].values)
        labels_count = df[1].value_counts()
        print(f'Unique labels: {unique_labels}')
        print(f'Labels count: {labels_count}')
        hist = labels_count.plot(kind='bar', title='Dataset Distribution')
        hist.set_xlabel('Labels')
        hist.set_ylabel('Count')
        plt.savefig(output_dir)
        plt.close()
        print(f'Dataset distribution plot saved at: {output_dir}')
        
    except Exception as e:
        print(f"Error getting data histogram: {e}")

if __name__ == '__main__':
    np.random.seed(42)  # Set seed for reproducibility
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    data_csv = os.path.join(data_dir, 'face_recognition', 'Dataset.csv')
    output_dir_csv = os.path.join(data_dir, 'face_recognition', 'dataset_faces.csv')
    images_dir = os.path.join(data_dir, 'face_recognition', 'Faces')
    output_dir_img = os.path.join(this_dir, 'resources', 'dataset_samples.png')

    get_data(data_csv, output_dir_csv)

    n_samples = 16
    images_path_lt = []
    labels_lt = []
    
    try:
        for img in os.listdir(images_dir):
            if img.endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
                images_path_lt.append(os.path.join(images_dir, img))
                labels_lt.append(img.split('_')[0])
        
        random_indices = np.random.choice(len(images_path_lt), n_samples, replace=False)
        images_path_lt = [images_path_lt[i] for i in random_indices]
        labels_lt = [labels_lt[i] for i in random_indices]

        plot_dataset_samples(n_samples, images_path_lt, labels_lt, output_dir_img)
    except Exception as e:
        print(f"Error processing images: {e}")
        
    output_dir_hist = os.path.join(this_dir, 'resources', 'dataset_distribution.png')
    get_dataset_distribution(output_dir_csv, output_dir_hist)

import os
import numpy as np
import pandas as pd

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

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    data_csv = os.path.join(data_dir, 'face_recognition', 'Dataset.csv')
    output_dir = os.path.join(data_dir, 'face_recognition', 'dataset_faces.csv')
    
    get_data(data_csv, output_dir)

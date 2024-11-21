import os 
import cv2 
import numpy 
from tqdm import tqdm 
import pandas as pd

def load_dataset(path):
    images = []
    i = 0 
    for file in tqdm(os.listdir(path)):
        img = cv2.imread(os.path.join(path, file))
        name = file.split('.')[0]
        images.append((name, img))
        i += 1
        if i == 1000:
            break
    return images 

def show_image(img, title):
    cv2.imshow(f'img id: {title}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def join(path1, path2, path_out,ho ='inner', index='FlickrId'):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df = pd.merge(df1, df2, on=index, how=ho)
    print()
    df.to_csv(path_out, index=False)

def delete_columns(path, columns):
    df = pd.read_csv(path)
    df = df.drop(columns, axis=1)
    df.to_csv(path, index=False)

def merge_data():
     join("../data/raw/downloaded/headers.csv", "../data/raw/downloaded/img_info.csv", "../data/preprocessed/external_features.csv")
     join("../data/preprocessed/external_features.csv", "../data/raw/downloaded/users.csv", "../data/preprocessed/external_features.csv", ho='left',index='UserId') 
     join("../data/preprocessed/external_features.csv", "../data/raw/downloaded/popularity.csv", "../data/preprocessed/external_features.csv")

def sort_data(external_data_path):
    #aca deberiamos splitear los datos HAY QUE TENER MUCHO CUIDADO EN ORDENAR BIEN LOS DATOS CON EL INDICE DE LA FOTO
    df = pd.read_csv(external_data_path)
    df = df.sort_values(by='FlickrId')
    df.to_csv(external_data_path, index=False)

if __name__ == "__main__":
    # merge_data()
    # sort_data("../data/raw/external_features.csv")
    pass


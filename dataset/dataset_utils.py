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





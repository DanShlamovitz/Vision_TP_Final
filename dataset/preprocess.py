import os 
import cv2 
import numpy 
from tqdm import tqdm 
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import shutil
import pickle as pkl


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


def split_data(path, output_folder, test_size=0.2, random_state=42, clip=True):
    # Cargar los datos

    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="FlickrId").dropna(subset=["FlickrId"])

    ids = df["FlickrId"].values

    df["Tags"] = df["Tags"].apply(lambda x: len(x.split(',')))

    columns_to_delete = ["UserId","Username", "Size", "Camera", "Country",
                        "Title", "Description", "URL", "Ispro","Day01","Day02","Day03",
                        "Day04","Day05","Day06","Day07","Day08","Day09","Day10","Day11",
                        "Day12","Day13","Day14","Day15","Day16","Day17","Day18","Day19",
                        "Day20","Day21","Day22","Day23","Day24","Day25","Day26","Day27",
                        "Day28","Day29"]

    X = df.drop(columns_to_delete + ["Day30"], axis=1)  # Excluir también Day30
    Y = df["Day30"]
    X = df.drop(columns_to_delete + ["Day30"], axis=1)  # Excluir también Day30
    Y = df["Day30"]

    filtered_ids = X["FlickrId"].values
    train_ids, test_ids = train_test_split(filtered_ids, test_size=test_size, random_state=random_state)

    print("El largo de train_ids es:", len(train_ids))
    print("El largo de test_ids es:", len(test_ids))

    X_train = X[X["FlickrId"].isin(train_ids)].copy()
    X_test = X[X["FlickrId"].isin(test_ids)].copy()
    Y_train = Y[X["FlickrId"].isin(train_ids)].copy()
    Y_test = Y[X["FlickrId"].isin(test_ids)].copy()

    print("El largo de X_train es:", len(X_train))
    print("El largo de X_test es:", len(X_test))
    print("El largo de Y_train es:", len(Y_train))
    print("El largo de Y_test es:", len(Y_test))

    X_train.drop("FlickrId", axis=1, inplace=True)
    X_test.drop("FlickrId", axis=1, inplace=True)

    if clip:
        train_clip_vectors = []
        test_clip_vectors = []
        
        # Cargar los vectores CLIP
        with open("../data/preprocessed/image_features.pkl", "rb") as f:
            clip_vectors = pkl.load(f)
            
            # Agregar los vectores de entrenamiento
            for train_id in train_ids:
                if str(train_id) in clip_vectors:
                    train_clip_vectors.append(clip_vectors[str(train_id)])
            
            # Agregar los vectores de prueba
            for test_id in test_ids:
                if str(test_id) in clip_vectors:
                    test_clip_vectors.append(clip_vectors[str(test_id)])

        print("El largo de clip vectors train es:", len(train_clip_vectors))
        print("El largo de clip vectors test es:", len(test_clip_vectors))

        # Guardar los vectores CLIP
        with open("../data/preprocessed/train/train_clip_vectors.pkl", "wb") as f:
            pkl.dump(train_clip_vectors, f)
        
        with open("../data/preprocessed/test/test_clip_vectors.pkl", "wb") as f:
            pkl.dump(test_clip_vectors, f)
    
    X_train.to_csv(os.path.join(output_folder, "train/X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_folder, "test/X_test.csv"), index=False)
    Y_train.to_csv(os.path.join(output_folder, "train/Y_train.csv"), index=False)
    Y_test.to_csv(os.path.join(output_folder, "test/Y_test.csv"), index=False)

if __name__ == "__main__":
    #esto si no tens el csv de external featueres hecho no anda
    # merge_data()
    # sort_data("../data/raw/external_features.csv")

    path = "../data/raw/external_features.csv"
    output_folder = "../data/preprocessed/"
    split_data(path, output_folder, test_size=0.1, random_state=42, clip=True)


    pass


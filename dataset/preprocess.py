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
import warnings

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

def one_hot_encoding(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df.drop(column, axis=1, inplace=True)
    return df

def multiply_all_columns(df):
    #add new columns multiplying columns with eachother
    # avoid doing this with columns that are already multiplied with eachothera
    print("Haciendo la cosa más ineficiente de la historia")
    #Print the types of all columns
    print(df.dtypes)
    columns = df.columns
    for i in tqdm(range(len(columns))):
        for j in range(i+1, len(columns)):
            df[f"{columns[i]}_{columns[j]}"] = df[columns[i]]* df[columns[j]]
    return df

def split_data(path, output_folder, test_size=0.2, random_state=42, clip=True, add_tabular=False, split_images=False):
    # Cargar los datos

    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="FlickrId").dropna(subset=["FlickrId"])

    ids = df["FlickrId"].values

    df["Tags"] = df["Tags"].apply(lambda x: len(x.split(',')))

    columns_to_delete = ["UserId","Username", "Size", "Camera",
                        "Title", "Description", "URL", "Ispro","Day01","Day02","Day03",
                        "Day04","Day05","Day06","Day07","Day08","Day09","Day10","Day11",
                        "Day12","Day13","Day14","Day15","Day16","Day17","Day18","Day19",
                        "Day20","Day21","Day22","Day23","Day24","Day25","Day26","Day27",
                        "Day28","Day29"]

    df = df.drop(columns_to_delete, axis=1)
    if add_tabular: df = one_hot_encoding(df, "Country")
    else: df = df.drop("Country", axis=1)
    X = df.drop(["Day30"], axis=1)  
    Y = df["Day30"]

    filtered_ids = X["FlickrId"].values
    train_ids, test_ids = train_test_split(filtered_ids, test_size=test_size, random_state=random_state)

    if split_images:
        image_folder = "../data/raw/imgs"
        for i in tqdm(range(len(ids))):
            if ids[i] in train_ids:
                shutil.move(os.path.join(image_folder, f"{ids[i]}.jpg"), os.path.join(output_folder, "train/train_imgs", f"{ids[i]}.jpg"))
            else:
                shutil.move(os.path.join(image_folder, f"{ids[i]}.jpg"), os.path.join(output_folder, "test/test_imgs", f"{ids[i]}.jpg"))

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
    if add_tabular:
        X_train = multiply_all_columns(X_train)
        X_test = multiply_all_columns(X_test)
    

    if clip:
        train_clip_vectors = []
        test_clip_vectors = []
        
        # Cargar los vectores CLIP
        with open("../data/preprocessed/image_features.pkl", "rb") as f:
            clip_vectors = pkl.load(f)
            
            for train_id in train_ids:
                if str(train_id) in clip_vectors:
                    train_clip_vectors.append(clip_vectors[str(train_id)])
            
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
    
    X_train.to_csv(os.path.join(output_folder, "train/X_train_tabular_add.csv"), index=False)
    X_test.to_csv(os.path.join(output_folder, "test/X_test_tabular_add.csv"), index=False)
    Y_train.to_csv(os.path.join(output_folder, "train/Y_train.csv"), index=False)
    Y_test.to_csv(os.path.join(output_folder, "test/Y_test.csv"), index=False)


def restore_images(output_folder, image_folder="../data/raw/imgs"):
    """
    Restore images from the train/test directories back to the original image folder.

    :param output_folder: Path to the folder where images were previously moved (train/test).
    :param image_folder: Path to the original folder where images should be moved back to.
    """
    train_folder = os.path.join(output_folder, "train/train_imgs")
    test_folder = os.path.join(output_folder, "test/test_imgs")
    
    # Move images from the 'train' folder back to the original location
    for img_file in tqdm(os.listdir(train_folder)):
        img_path = os.path.join(train_folder, img_file)
        if os.path.exists(img_path):
            shutil.move(img_path, os.path.join(image_folder, img_file))
    
    # Move images from the 'test' folder back to the original location
    for img_file in tqdm(os.listdir(test_folder)):
        img_path = os.path.join(test_folder, img_file)
        if os.path.exists(img_path):
            shutil.move(img_path, os.path.join(image_folder, img_file))
    
    print("Imágenes restauradas a su carpeta original.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    #esto si no tens el csv de external featueres hecho no anda
    # merge_data()
    # sort_data("../data/raw/external_features.csv")

    path = "../data/raw/external_features.csv"
    output_folder = "../data/preprocessed/"
    split_data(path, output_folder, test_size=0.1, random_state=42, clip=False, add_tabular=False, split_images=True)


    pass


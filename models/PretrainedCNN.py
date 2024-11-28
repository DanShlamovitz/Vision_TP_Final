import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os
from tqdm import tqdm

class PretrainedCNN:
    def __init__(self, model_name: str = 'resnet50'):
        """
        Inicializa el clasificador de imágenes.

        :param model_name: Nombre del modelo (actualmente soporta 'resnet50').
        """
        if model_name != 'resnet50':
            raise ValueError("Actualmente, solo se admite 'resnet50'.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()  # Modo evaluación
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #estos tienen q ser hiperparámetros
        ])

    def predict(self, folder_path: str) -> torch.Tensor:
        """
        Genera predicciones para todas las imágenes en un directorio.

        :param folder_path: Ruta al directorio con imágenes.
        :return: Tensor con las predicciones.
        """
        image_files = [ os.path.join(folder_path, f) for f in os.listdir(folder_path)]

        predictions = []
        for img_path in tqdm(image_files):
            img = Image.open(img_path).convert('RGB')
            input_tensor = self.transform(img).unsqueeze(0).to(self.device) #ESTO NO ESTA TAN BUNEO DEBERIA ACHICARLO BIEN O UTILIZAR VSCNN
            with torch.no_grad():
                output = self.model(input_tensor)
                predictions.append(output.cpu())


        
        return torch.cat(predictions, dim=0)


if __name__ == "__main__":
    # Crear una instancia de la clase con ResNet50
    resnet50 = PretrainedCNN("resnet50")
    
    # Ruta al directorio con imágenes
    folder_path = "../data/raw/imgs"
    predictions = resnet50.predict(folder_path)
    print(predictions.shape)
    

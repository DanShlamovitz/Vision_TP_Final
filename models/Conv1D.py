import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Conv1D(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_layers: int, 
                 kernel_sizes: list, 
                 num_filters: list, 
                 stride: int = 1, 
                 padding: int = 0, 
                 pooling: bool = True, 
                 pool_kernel: int = 2, 
                 dropout_rate: float = 0.5, 
                 avg_pooling: bool = False, 
                 output_size: int = 1):
        """
        Red neuronal con múltiples capas de convolución 1D configurables.

        Args:
            input_size (int): Tamaño del vector de entrada.
            hidden_layers (int): Número de capas ocultas.
            kernel_sizes (list): Lista de tamaños de kernel para cada capa.
            num_filters (list): Lista de filtros para cada capa.
            stride (int): Stride de las convoluciones.
            padding (int): Padding de las convoluciones.
            pooling (bool): Indica si se aplica pooling.
            pool_kernel (int): Tamaño del kernel de pooling.
            dropout_rate (float): Tasa de dropout.
            avg_pooling (bool): Usa pooling promedio si es True; max pooling si es False.
            output_size (int): Número de salidas de la red.
        """
        super(Conv1D, self).__init__()
        assert len(kernel_sizes) == hidden_layers, "kernel_sizes debe tener el mismo tamaño que hidden_layers."
        assert len(num_filters) == hidden_layers, "num_filters debe tener el mismo tamaño que hidden_layers."

        layers = []
        in_channels = 1
        for i in range(hidden_layers):
            layers.append(nn.Conv1d(in_channels, num_filters[i], kernel_size=kernel_sizes[i], stride=stride, padding=padding))
            layers.append(nn.ReLU())
            if pooling:
                layers.append(nn.AvgPool1d(pool_kernel) if avg_pooling else nn.MaxPool1d(pool_kernel))
            layers.append(nn.Dropout(dropout_rate))
            in_channels = num_filters[i]
        
        self.conv_layers = nn.Sequential(*layers)
        flatten_size = self._calculate_flatten_size(input_size, kernel_sizes, stride, padding, pool_kernel, num_filters)
        self.fc = nn.Linear(flatten_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Añadir dimensión de canal
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Aplanar
        x = self.fc(x)
        return x

    def _calculate_flatten_size(self, input_size, kernel_sizes, stride, padding, pool_kernel, num_filters):
        size = input_size
        for i in range(len(kernel_sizes)):
            size = (size - kernel_sizes[i] + 2 * padding) // stride + 1
            if pool_kernel:
                size = size // pool_kernel
        return size * num_filters[-1]

    def train_model(self, train_loader, num_epochs, learning_rate=0.001):
        """
        Entrena el modelo con un conjunto de datos.

        Args:
            train_loader (DataLoader): Conjunto de entrenamiento.
            num_epochs (int): Número de épocas.
            learning_rate (float): Tasa de aprendizaje.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(num_epochs):
            self.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs).squeeze()
                loss = criterion(outputs, labels.squeeze())
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    def predict(self, test_loader):
        """
        Realiza predicciones en un conjunto de datos.

        Args:
            test_loader (DataLoader): Conjunto de datos de prueba.

        Returns:
            torch.Tensor: Predicciones del modelo.
        """
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = self(inputs).squeeze()
                preds = torch.sigmoid(outputs) > 0.5  # Umbral para clasificación binaria
                predictions.append(preds)
        return torch.cat(predictions)

    def save_weights(self, file_path):
        """
        Guarda los pesos del modelo en un archivo.

        Args:
            file_path (str): Ruta donde guardar los pesos.
        """
        torch.save(self.state_dict(), file_path)

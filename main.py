import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- Definição dos hiperparâmetros ---

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# --- Carregamento e preparação dos dados (dataloaders) ---

# Define uma sequência de transformações para aplicar às imagens
# 1. ToTensor(): Converte a imagem (PIL) para um Tensor PyTorch
# 2. Normalize(): Normaliza os valores dos pixels para ter média 0.5 e desvio padrão 0.5.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5), (0.5, 0.5))
])

# Baixa o dataset de TREINO
# root: onde salvar os dados
# train=True: pegar o conjunto de treino
# download=True: baixar se não existir
# transform: aplicar as transformações que definimos
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Baixa o dataset de TESTE
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Cria os Dataloaders para os conjuntos de treino e teste
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
# --- Definição da arquitetura da rede neural ---

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Camada linear com 28*28=784 entradas e 128 saídas
        self.flatten = nn.Flatten()

        # 'nn.Sequential' é um contêiner que passa os dados por todas as camadas em sequência
        self.fc_layers = nn.Sequential(
            nn.Linear(28*28, 128), # Camada de entrada (784 neurônios) para camada oculta (128)
            nn.ReLU(),             # Função de ativação: Rectified Linear Unit
            nn.Linear(128, 10)     # Camada oculta (128) para camada de saída (10)
                                   # (10 classes, de 0 a 9)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.fc_layers(x)
        return logits

# instancia o modelo
model = SimpleMLP()

# --- Definição da função de perda e otimizador ---

# Função de Perda (Criterion): Mede o quão errado o modelo está.
# 'CrossEntropyLoss' é padrão para classificação com múltiplas classes.
# Ela já aplica a função 'Softmax' internamente.
criterion = nn.CrossEntropyLoss()

# Otimizador: Algoritmo que atualiza os pesos do modelo para reduzir a perda.
# 'Adam' é um otimizador robusto e muito popular.
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- Loop de treinamento ---

for epoch in range(NUM_EPOCHS):
    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # 'images' é um tensor de shape [64, 1, 28, 28] (batch_size, canais, altura, largura)
        # 'labels' é um tensor de shape [64] (os rótulos corretos)

        # 1. Zerar os gradientes
        #    (PyTorch acumula gradientes, então precisamos zerar a cada passo)
        optimizer.zero_grad()

        # 2. Forward pass (passa os dados pelo modelo)
        outputs = model(images)

        # 3. calcular a perda (loss)
        loss = criterion(outputs, labels)

        # 4. backward pass (calcula os gradientes)
        loss.backward()

        # 5. otimizar (atualiza os pesos)
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print(f'--- Fim da Epoch {epoch+1} --- Média de Loss: {running_loss/len(train_loader):.4f} ---')

print('iniciando a avaliação no conjunto de teste...')

# coloca o modelo em modo avaliação
model.eval()

correct = 0
total = 0

# 'with torch.no_grad():' desliga o cálculo de gradientes
# Isso economiza memória e acelera a avaliação, já que não estamos treinando.
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # torch.max retorna (valores, indices) ao longo de uma dimensão
        # queremos os indices com maior probabilidade (dim= 1)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# calcular e imprimir a acurácia final
accuracy = 100 * correct / total
print(f'Acurácia do modelo no conjunto de teste: {accuracy:.2f} %')
# Classificador de Dígitos (MNIST) com PyTorch

Este é um projeto de estudo básico para implementar um classificador de dígitos escritos à mão (dataset MNIST) usando PyTorch.

O objetivo principal é praticar o fluxo de trabalho completo de um projeto de Deep Learning:

1.  Carregamento e transformação de dados (`torchvision.datasets` e `DataLoader`).
2.  Definição de uma arquitetura de rede neural (`nn.Module`).
3.  Definição de uma função de perda (Loss) e um otimizador (`nn.CrossEntropyLoss` e `optim.Adam`).
4.  Implementação do loop de treinamento (training loop).
5.  Avaliação do modelo no conjunto de teste.

## Tecnologias Utilizadas

- Python 3
- PyTorch
- Torchvision

## Como Executar

1.  **Clone o repositório** (ou apenas tenha o arquivo `.py` em uma pasta).

2.  **Crie um ambiente virtual** (Recomendado):

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Linux/macOS
    .\venv\Scripts\activate   # No Windows
    ```

3.  **Instale as dependências**:

    ```bash
    pip install torch torchvision
    ```

4.  **Execute o script**:
    ```bash
    python main.py
    ```
    _Obs: O script irá baixar automaticamente o dataset MNIST para uma pasta local chamada `./data` na primeira execução._

## O que esperar

O script irá treinar o modelo por 5 épocas (padrão) e, ao final, exibirá a acurácia alcançada no conjunto de dados de teste.

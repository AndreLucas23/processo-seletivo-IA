import os

# Oculta alguns logs do Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
import time
import random

# Configura seeds para garantir que os resultados sejam reproduzíveis em diferentes execuções
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("\n--- Iniciando Pipeline de Treinamento (Edge AI) ---")

# Carregamento e pré-processa os dados do MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalização: Converte os pixels de 0 - 255 para 0.0 - 1.0
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape: Adiciona a dimensão do canal (1 = escala de cinza) exigida pela Conv2D
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Arquitetura do Modelo (Foco em Eficiência)
# Utiliza apenas 2 camadas Conv2D com poucos filtros para reduzir parâmetros
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)), 
    
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Configura o processo de aprendizado com o otimizador Adam e a função de perda para classes inteiras
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print()

# Treinamento Limitado (Adequado para CI/CPU)
start_time = time.time()

# Executa o treinamento do modelo
# Restringe a 5 épocas e com batch_size de 64 para acelerar o processo no CI
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64, verbose=2)
end_time = time.time()

# Obtém e imprime as métricas contextualizadas
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print("\n--- Resultados e Métricas ---")
print(f"Tempo de Treinamento: {end_time - start_time:.2f} segundos")
print(f"Acurácia Final no Teste: {test_acc:.4f} (Métrica principal de assertividade)")
print(f"Loss Final no Teste: {test_loss:.4f} (Métrica de penalidade de erro)\n")

# Salva o arquivo 'model.h5' e informa o sucesso na ação seguido do tamanho dele
model_path = 'model.h5'
model.save(model_path)
print(f"\n[SUCESSO] Modelo base salvo em: {model_path}")
print(f"Tamanho inicial do modelo: {os.path.getsize(model_path) / 1024:.2f} KB")

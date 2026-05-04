import os

# Oculta alguns logs do Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

print("--- Iniciando Otimização e Conversão (TFLite) ---\n")

# Guarda os nomes dos arquivos atual e optimizado
model_path = 'model.h5'
tflite_path = 'model.tflite'

# Carrega o modelo baseline
model = tf.keras.models.load_model(model_path)

# Configura o Conversor para TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Aplica a técnica de Dynamic Range Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Executa a Conversão
tflite_quant_model = converter.convert()

# Salva o Artefato Otimizado
with open(tflite_path, 'wb') as f:
    f.write(tflite_quant_model)

# Avaliação de Trade-offs (Análise de Tamanho)
original_size = os.path.getsize(model_path)
tflite_size = os.path.getsize(tflite_path)
reduction = (1 - (tflite_size / original_size)) * 100

# Log final sobre os resultados
print("\n--- Relatório de Otimização ---")
print(f"Tamanho Original (.h5): {original_size / 1024:.2f} KB")
print(f"Tamanho Otimizado (.tflite): {tflite_size / 1024:.2f} KB")
print(f"Redução de Tamanho: {reduction:.2f}%")
print(f"[SUCESSO] Modelo otimizado salvo em: {tflite_path}")

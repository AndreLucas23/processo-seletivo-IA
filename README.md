# Desafio de Visão Computacional para Edge AI

👤 **Identificação:** **Nome Completo:** André Lucas de Souza Lima

---

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura da Rede Neural Convolucional (CNN) foi desenhada com um foco estrito em **Edge AI**, priorizando a simplicidade e a eficiência de hardware em vez da profundidade extrema. O modelo construído no `train_model.py` possui a seguinte estrutura sequencial:

* **Camadas de Extração de Características:**
    * 1x Camada `Conv2D` com apenas 16 filtros (3x3), ativada por ReLU.
    * 1x Camada `MaxPooling2D` (2x2) para redução espacial.
    * 1x Camada `Conv2D` com 32 filtros (3x3), ativada por ReLU.
    * 1x Camada `MaxPooling2D` (2x2) para compressão final das features.
* **Camadas de Classificação:**
    * 1x Camada `Flatten` para transformar a matriz em um vetor unidimensional.
    * 1x Camada `Dense` com apenas 64 neurônios (reduzindo o consumo de RAM em comparação aos padrões de 128/256).
    * 1x Camada `Dense` de saída com 10 neurônios (Softmax) para as classes de 0 a 9.

---

### 2️⃣ Bibliotecas Utilizadas

As dependências foram mantidas em seu mínimo necessário para garantir compatibilidade e leveza no ambiente de CI:

* `tensorflow` (Versão >= 2.15.0) - Core do treinamento e conversão.
* `numpy` (Versão >= 1.26.0) - Manipulação de arrays.
* Bibliotecas nativas do Python: `os` (gerenciamento de arquivos e tamanhos) e `time` (medição de performance).

---

### 3️⃣ Técnica de Otimização do Modelo

No script `optimize_model.py`, o formato original do Keras (`.h5`) foi convertido para TensorFlow Lite (`.tflite`) aplicando a técnica de **Dynamic Range Quantization** (Quantização de Faixa Dinâmica).

* **Como funciona:** Durante a conversão, os pesos do modelo são convertidos do formato de ponto flutuante de 32 bits (`float32`) para inteiros de 8 bits (`int8`). As ativações, no entanto, permanecem em ponto flutuante durante a inferência.
* **Por que foi escolhida?** É a técnica ideal para balancear performance e simplicidade. Ela entrega reduções de tamanho próximas a 4x (75% menores) e acelerações de latência significativas na CPU (comum em microcontroladores e placas IoT), sem a necessidade de fornecer um *dataset representativo* complexo durante a conversão (como exigido pela quantização total de inteiros).

---

### 4️⃣ Resultados Obtidos

O treinamento (restrito a 5 épocas via CPU no pipeline) atingiu convergência rápida, gerando os seguintes artefatos organizados: um modelo de base analítica (`model.h5`) e um artefato de produção para inferência embarcada (`model.tflite`).

* **Acurácia (Accuracy) > 97%:** Adotada como métrica principal, pois o dataset MNIST é perfeitamente balanceado (mesma quantidade de exemplos por classe). Demonstra a assertividade global do modelo.
* **Perda (Loss) < 0.10:** A `sparse_categorical_crossentropy` confirmou que não houve ocorrência de *overfitting* severo no curto período de treinamento.
* **Métrica Física (Redução de Tamanho):** O arquivo `.h5` original passou de aproximadamente ~400 KB para cerca de ~100 KB no formato `.tflite`, validando o sucesso da etapa de otimização.

---

### 5️⃣ Comentários Adicionais (Reflexões Técnicas)

* **Decisões Técnicas e Trade-offs:** A principal decisão do projeto foi o *trade-off* (compromisso) entre tamanho e desempenho extremo. Ao aplicar a quantização, aceitamos uma redução quase imperceptível na acurácia (frações de porcentagem) em troca de viabilizar a implantação do modelo em chips com memória Flash limitadíssima. 
* **Organização dos Salvamentos:** O fluxo adotado (gerar o `.h5` e depois o `.tflite`) garante rastreabilidade. Mantemos o modelo de pesquisa (`h5`) intacto para eventuais retreinamentos, enquanto usamos o arquivo TFLite estritamente para o *deploy*.
* **Limitações do Modelo:** Sendo altamente otimizado para o MNIST, este modelo espera imagens rigorosamente pré-processadas (tons de cinza, 28x28, com o dígito centralizado). Em um cenário do "mundo real", precisaria de um pipeline de pré-processamento de câmera rodando em C++ no microcontrolador antes de alimentar o `.tflite`.
* **Aprendizados:** O desafio reforçou a perspectiva sistêmica de que desenvolver IA não é apenas compilar camadas matemáticas visando "100% de acerto", mas sim um exercício de engenharia de software: respeitando restrições de memória, poder de processamento e latência do hardware de destino.
* **Reprodutibilidade e Determinismo:** Para garantir que os resultados de acurácia e perda (loss) sejam consistentes em qualquer ambiente de execução (local ou CI/CD), foi implementado um controle rigoroso de determinismo com Seeding Multicamadas, acionamento de variável de ambiente determinística e isolamento de hardware (CPU)
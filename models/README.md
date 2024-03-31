# Modelos de Análisis de Sentimiento

Este directorio contiene los modelos entrenados y serializados utilizados en el proyecto de análisis de sentimientos de reseñas de productos deportivos de Amazon. Cada archivo representa un modelo diferente y su estado después del entrenamiento.

## Contenido del Directorio

- `bow_features.pkl`: Características extraídas de Bag of Words, utilizadas para entrenar los modelos de machine learning.
- `gradient_booting.joblib`: Un modelo de Gradient Boosting optimizado con búsqueda de hiperparámetros.
- `logistic_regression.joblib`: Modelo de Regresión Logística que presenta un rendimiento robusto en el análisis de sentimientos.
- `lstm_model.joblib`: Modelo de Deep Learning basado en LSTM, adecuado para capturar relaciones contextuales en las reseñas.
- `tokenizer.pickle`: Tokenizador serializado que prepara el texto para el modelo LSTM.
- `w2v_model.pkl`: Modelo de Word2Vec que puede ser utilizado para convertir palabras en vectores numéricos.

## Uso de los Modelos

Para utilizar cualquiera de estos modelos, deberá cargarlos en su entorno de Python utilizando bibliotecas como `joblib` o `pickle`. Por ejemplo:

```python
from joblib import load

# Para cargar el modelo de Gradient Boosting
gboost_model = load('gradient_booting.joblib')

# Para cargar el tokenizador para LSTM
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
```

## Requerimientos

Asegúrate de tener instaladas las bibliotecas necesarias para la deserialización y el uso de estos modelos, como `scikit-learn`, `keras` y `gensim` para `w2v_model.pkl`.
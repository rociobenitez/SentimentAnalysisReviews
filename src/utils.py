import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk import FreqDist
from sklearn.manifold import TSNE
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import joblib

# Configuración de logging para Gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt='%H:%M:%S', level=logging.INFO)

# Descargar recursos necesarios de NLTK


def download_nltk_spacy_data():
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Para soporte multilingüe en lematización

# Funciones para cargar datos


def load_data(filepath):
    """Carga datos desde un archivo CSV y devuelve un DataFrame."""
    return pd.read_csv(filepath, encoding='utf-8')


def load_data_drive(filepath):
    """
    Carga un archivo CSV desde Google Drive en un DataFrame de pandas.
    Primero monta Google Drive si aún no está montado, y luego carga el archivo.

    Parámetros:
    - filepath (str): La ruta relativa al archivo CSV en Google Drive,
      comenzando desde 'My Drive/'.

    Retorna:
    - Un DataFrame de pandas con los datos cargados desde el archivo CSV.
    """

    from google.colab import drive

    # Montar Google Drive si aún no está montado
    drive_path = '/content/drive'
    if not os.path.isdir(drive_path):
        print("Montando Google Drive...")
        drive.mount(drive_path)
        print("Google Drive montado exitosamente.")
    else:
        print("Google Drive ya está montado.")

    # Definir la ruta completa al archivo
    path = os.path.join(drive_path, 'My Drive', filepath)

    # Cargar y retornar el DataFrame
    try:
        df = pd.read_csv(path)
        print(f"Archivo cargado exitosamente desde: {path}")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

# Funciones para guardar datos y figuras


def save_data_local(df, filename):
    """
    Guarda un DataFrame en un archivo CSV localmente.

    Parámetros:
    - df: DataFrame a guardar.
    - filename: Nombre del archivo para guardar el DataFrame.
    """
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"DataFrame guardado localmente como: {filename}")


def save_data(df, filepath, format='csv'):
    """
    Guarda un DataFrame en un archivo, soportando múltiples formatos.

    Parámetros:
    - df: DataFrame de pandas a guardar.
    - filepath: Ruta completa del archivo donde se guardarán los datos.
    - format: Formato del archivo para guardar los datos ('csv' o 'parquet').
    """
    if format == 'csv':
        df.to_csv(filepath, index=False)
        print(f"DataFrame guardado como archivo CSV en: {filepath}")
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
        print(f"DataFrame guardado como archivo Parquet en: {filepath}")
    else:
        raise ValueError(f"Formato de archivo '{format}' no soportado.")


def save_data_drive(df, filename, filepath_drive='/content/drive/My Drive/Colab Notebooks/Sports_and_Outdoors/'):
    """Guarda un DataFrame en un archivo CSV dentro de Google Drive, en la ruta especificada."""
    # Montar Google Drive si aún no está montado
    from google.colab import drive
    drive_path = '/content/drive'
    if not os.path.isdir(drive_path):
        print("Montando Google Drive...")
        drive.mount(drive_path)
        print("Google Drive montado exitosamente.")

    # Construir la ruta completa donde el archivo será guardado
    path = os.path.join(drive_path, 'My Drive', filepath_drive, filename)

    # Crear directorios si no existen
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Guardar DataFrame en archivo CSV
    df.to_csv(path, index=False)
    print(f"DataFrame guardado exitosamente en: {path}")


def save_figure(figure, filepath, filename, dpi=300):
    """Guarda una figura en la ruta especificada con el nombre de archivo y DPI dados."""
    path = os.path.join(filepath, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=dpi)
    print(f"Figura guardada como: {path}")


# Funciones para visualización
def style_plot(ax=None):
    """Estiliza el gráfico escondiendo ejes innecesarios y cambiando colores."""
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    plt.tight_layout()
    plt.show()


def plot_distribution(data, x_column, title='Distribución', xlabel='X-Axis', ylabel='Y-Axis', color='royalblue'):
    """
    Grafica la distribución de valores de una columna especificada de un DataFrame.

    Parámetros:
    - data (DataFrame): DataFrame de pandas con la columna especificada.
    - x_column (str): Nombre de la columna del DataFrame para graficar.
    - title (str, opcional): Título del gráfico.
    - xlabel (str, opcional): Etiqueta para el eje X.
    - ylabel (str, opcional): Etiqueta para el eje Y.
    - color (str, opcional): Color de las barras del gráfico.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=x_column, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    style_plot()


def plot_ngrams(ngrams_counts, title='N-Grams más frecuentes'):
    """
    Grafica los N-Grams más frecuentes.

    Parámetros:
    - ngrams_counts: Lista de tuplas (ngram, count).
    - title: Título del gráfico.
    """
    ngrams, counts = zip(*ngrams_counts)
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(ngrams)), counts, color='royalblue')
    plt.yticks(range(len(ngrams)), [' '.join(ng) for ng in ngrams])
    plt.title(title)
    style_plot()


def generate_wordcloud(words, width=800, height=400, colormap='viridis'):
    """
    Genera y muestra una nube de palabras.

    Parámetros:
    - words: Una lista de palabras o un string largo de palabras separadas por espacios.
    - width: Ancho de la nube de palabras.
    - height: Altura de la nube de palabras.
    """
    wordcloud = WordCloud(width=width, height=height, background_color='white', colormap=colormap).generate(
        " ".join(words) if isinstance(words, list) else words)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def plot_most_common_words(word_list, num_words=20, title='20 Palabras más comunes', xlabel='Frecuencia'):
    """
    Genera y muestra un gráfico de barras horizontal con las palabras más comunes.

    Parámetros:
    - word_list (list): Lista de palabras (str) de la cual calcular las frecuencias.
    - num_words (int): Número de palabras más comunes a mostrar.
    - title (str): Título del gráfico.
    - xlabel (str): Etiqueta para el eje X del gráfico.
    """
    # Calcular la distribución de frecuencia de las palabras
    freq_dist = FreqDist(word_list)
    most_common_words = freq_dist.most_common(num_words)

    # Separar las palabras y sus frecuencias
    words = [word[0] for word in most_common_words]
    counts = [word[1] for word in most_common_words]

    # Configurar el tamaño y estilo del gráfico
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    plt.barh(words, counts, color='royalblue')
    plt.xlabel(xlabel)
    plt.title(title)

    # Invertir el eje Y para que la palabra más común esté en la parte superior
    plt.gca().invert_yaxis()
    style_plot()


def tsne_plot_similar_words(keys, embeddings_2d, word_clusters, figsize=(16, 9), color_map='plasma', title="TSNE Visualization of Similar Words"):
    """
    Visualiza palabras similares en 2D utilizando t-SNE y agrega una leyenda para identificar los clusters.

    Parámetros:
    - title: Título del gráfico.
    - keys: Palabras clave a partir de las cuales se generaron los clusters de palabras similares.
    - embeddings_2d: Coordenadas 2D de las palabras después de aplicar t-SNE.
    - word_clusters: Clusters de palabras similares para cada palabra clave.
    - figsize: Tamaño de la figura del gráfico.
    - color_map: Mapa de colores a usar para distinguir los diferentes clusters.
    """
    plt.figure(figsize=figsize)
    colors = plt.colormaps[color_map](np.linspace(0, 1, len(keys)))

    for key, color, embeddings, words in zip(keys, colors, embeddings_2d, word_clusters):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=[color], label=key, alpha=0.7)

        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)

    plt.legend(loc='best')
    plt.title(title)
    plt.grid(False)
    style_plot()


def tsne_visualization(model, keys, topn=10, perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32):
    """
    Visualiza en 2D las palabras más similares a una lista de palabras clave utilizando t-SNE.

    Parámetros:
    - model: Modelo Word2Vec de Gensim cargado.
    - keys: Lista de palabras clave para encontrar palabras similares.
    - topn: Número de palabras similares a buscar para cada palabra clave.
    - perplexity, n_components, init, n_iter, random_state: Parámetros para el algoritmo t-SNE.
    """
    similar_words = {search_term: [item[0] for item in model.wv.most_similar(search_term, topn=topn)]
                     for search_term in keys}

    embedding_clusters = []
    word_clusters = []

    for word, similar in similar_words.items():
        embeddings = []
        words = []
        for similar_word in similar:
            words.append(similar_word)
            embeddings.append(model.wv[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    embeddings_2d = np.array(TSNE(perplexity=perplexity, n_components=n_components, init=init, n_iter=n_iter, random_state=random_state)
                             .fit_transform(embedding_clusters.reshape(n * m, k))
                             ).reshape(n, m, 2)

    tsne_plot_similar_words(keys, embeddings_2d, word_clusters)


def plot_review_length_distribution(df, column, bins=50, color='royalblue', title='Distribución de la Longitud de las Reviews', xlabel='Longitud de la Review', ylabel='Frecuencia'):
    """Grafica la distribución de la longitud de las reviews."""
    df['text_length'] = df[column].apply(len)
    sns.histplot(df['text_length'], bins=bins, kde=True, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    style_plot()


def plot_rating_helpfulness(df, x='rating', y='helpful_vote', color='royalblue', title='Relación entre Ratings y Votos de Utilidad', xlabel='Rating', ylabel='Votos de Utilidad'):
    """Grafica la relación entre ratings y votos de utilidad."""
    sns.scatterplot(x=x, y=y, data=df, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    style_plot()


def plot_verified_purchase_distribution(df, column='verified_purchase', color='royalblue', title='Reviews Verificadas vs. No Verificadas', xlabel='Compra Verificada', ylabel='Cantidad'):
    """Grafica la distribución de compras verificadas vs no verificadas."""
    sns.countplot(x=column, data=df, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([0, 1], ['No', 'Sí'])
    style_plot()


def plot_chi2_scores(X, y, vectorizer, top_features=15):
    """
    Plotea los scores de Chi-cuadrado de las palabras en relación a las etiquetas de clase.

    Parámetros:
    - X: matriz de términos-documento (normalmente salida de TfidfVectorizer o CountVectorizer).
    - y: etiquetas de clase para el conjunto de datos.
    - vectorizer: vectorizador utilizado para transformar el texto en la matriz término-documento.
    - top_features: número de palabras principales para mostrar.
    """

    # Calculamos el score de Chi-cuadrado para cada palabra
    chi2score = chi2(X, y)[0]
    scores = list(zip(vectorizer.get_feature_names_out(), chi2score))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    topchi2 = list(zip(*sorted_scores[:top_features]))

    x = range(len(topchi2[1]))
    labels = topchi2[0]

    # Crear figura
    plt.figure(figsize=(12, 8))

    # Crear gráfico de barras horizontal
    plt.barh(x, topchi2[1], align='center', alpha=0.5, color="royalblue")
    plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8, color="royalblue")

    plt.yticks(x, labels, fontsize=10)
    plt.xlabel('$\chi^2$', fontsize=12)
    plt.ylabel('Word', fontsize=14)
    plt.title(f'Top {top_features} $\chi^2$ Scores for Each Word', fontsize=20)
    style_plot()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    style_plot()


def plot_zipf_law(features_train):
    """
    Esta función grafica la distribución de frecuencia de las palabras y la compara con la Ley de Zipf.

    Args:
    - features_train: una matriz de características BoW o TF-IDF de entrenamiento (numpy array).

    Returns:
    - Una gráfica log-log de las frecuencias ordenadas y la ley de Zipf.
    """
    # Sumamos todas las columnas para obtener las frecuencias totales de cada palabra
    word_counts = np.sum(features_train, axis=0)

    # Ordenamos las frecuencias de mayor a menor
    sorted_word_counts = np.sort(word_counts)[::-1]

    # Generamos los rangos para la ley de Zipf
    ranks = np.arange(1, len(sorted_word_counts)+1)

    # Creamos la gráfica
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, sorted_word_counts, label='Frecuencias observadas')
    plt.loglog(ranks, sorted_word_counts[0] /
               ranks, label='Ley de Zipf', linestyle='--')
    plt.xlabel('Rango')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Frecuencias de Palabras y Ley de Zipf')
    plt.legend()
    style_plot()


def plot_word_frequency_distribution(features_train):
    """
    Esta función grafica la distribución de frecuencias de palabras usando un histograma.

    Args:
    - features_train: matriz de características BoW o TF-IDF de entrenamiento (numpy array).
    """
    word_counts = np.sum(features_train, axis=0).ravel()
    word_freq_distribution = np.sort(word_counts)

    # Visualización con Histograma
    plt.figure(figsize=(10, 6))
    plt.hist(word_freq_distribution, bins=50, log=True)
    plt.title('Histograma de la Distribución de Frecuencias de Palabras')
    plt.xlabel('Frecuencia de la Palabra')
    plt.ylabel('Número de Palabras')
    style_plot()
    plt.show()


def plot_word_frequency_boxplot(features_train):
    """
    Esta función grafica un boxplot de la distribución de frecuencias de palabras.

    Args:
    - features_train: matriz de características BoW o TF-IDF de entrenamiento (numpy array).
    """
    word_counts = np.sum(features_train, axis=0).ravel()

    # Visualización con Boxplot
    plt.figure(figsize=(6, 6))
    sns.boxplot(word_counts)
    plt.title('Boxplot de la Distribución de Frecuencias de Palabras')
    plt.xlabel('Frecuencia de la Palabra')
    style_plot()


def plot_accuracy_evolution(c_params, train_acc, test_acc):
    """
    Esta función grafica la evolución de la precisión de los conjuntos de entrenamiento y prueba para diferentes valores de C.

    Args:
    - c_params: Lista de valores de C.
    - train_acc: Lista de precisión del entrenamiento.
    - test_acc: Lista de precisión de la prueba.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(train_acc, label='train')
    plt.plot(test_acc, label='test')
    plt.axvline(np.argmax(test_acc), c='g', ls='--', alpha=0.8)
    plt.title('Evolución de la Precisión para Diferentes Valores de C')
    plt.xlabel('Índice de C')
    plt.ylabel('Precisión')
    plt.legend()
    # rotación agregada para mejorar la legibilidad
    plt.xticks(list(range(len(c_params))), c_params, rotation=45)
    style_plot()
    plt.tight_layout()
    style_plot()

def plot_training_history(history):
    """
    Función para graficar la precisión de entrenamiento y validación a lo largo de las épocas.

    Args:
    - history: El historial de entrenamiento retornado por el método fit() de Keras.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.title('Evolución de la Precisión a lo largo de las Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    style_plot()



# Función de extracción de características
def extract_BoW_features(words_train, words_test, max_features=50000,
                         cache_dir=None, cache_file=None):
    """Extract Bag-of-Words features using TfidfVectorizer."""

    # Intentar cargar las características precalculadas desde la caché
    if cache_file:
        try:
            with open(cache_file, "rb") as f:
                cache_data = joblib.load(f)
            print("Leer características desde archivo de caché:", cache_file)
            return cache_data['features_train'], cache_data['features_test'], cache_data['vocabulary']
        except Exception as e:
            print("Archivo de caché no encontrado. Extrayendo características.", e)
            pass

    # Si no se encuentra la caché, calcular las características
    vectorizer = TfidfVectorizer(max_features=max_features)
    features_train = vectorizer.fit_transform(words_train).toarray()
    features_test = vectorizer.transform(words_test).toarray()

    # Guardar las características en la caché para uso futuro
    if cache_file:
        vocabulary = vectorizer.vocabulary_
        cache_data = {'features_train': features_train,
                      'features_test': features_test,
                      'vocabulary': vocabulary}
        with open(cache_file, "wb") as f:
            joblib.dump(cache_data, f)
        print("Escribir características en el archivo de caché:", cache_file)

    return features_train, features_test, vectorizer.vocabulary_

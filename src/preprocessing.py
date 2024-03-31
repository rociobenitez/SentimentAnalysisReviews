"""
preprocessing.py

Este script define varias funciones de preprocesamiento de texto para análisis de NLP, 
utilizando bibliotecas como NLTK y spaCy. Proporciona diferentes enfoques para limpiar 
y preparar textos, adecuados para modelos basados en bolsa de palabras y modelos 
contextuales.

Requisitos previos:
- Asegúrate de tener instaladas todas las dependencias listadas en requirements.txt.
- Para el uso de spaCy, es necesario descargar el modelo 'en_core_web_sm' ejecutando:
  `python -m spacy download en_core_web_sm`

Cómo usar:
- Importa las funciones definidas en este script a tu proyecto de análisis de NLP.
- Aplica las funciones de limpieza según las necesidades específicas de tu análisis.
"""

# Importaciones necesarias
import re
import unicodedata
import spacy
from negspacy.negation import Negex
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from num2words import num2words
import contractions

# Descargar recursos necesarios de NLTK
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Para soporte multilingüe en lematización
nltk.download('punkt')
nltk.download('vader_lexicon')

# Pipeline 1
# Preprocesamiento con spaCy para Análisis de Sentimientos

# Inicializar nlp fuera de cualquier función para evitar múltiples cargas
# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")

# Agregar el componente de negación de spaCy con configuración para tipos de entidad específicos
nlp.add_pipe("negex", config={"ent_types": [
             "PRODUCT", "ORG", "GPE", "DATE", "MONEY", "QUANTIFY"]})


def clean_text_spacy(text):
    """
    Limpia y procesa el texto utilizando spaCy para análisis de sentimientos.
    Esta función es óptima para análisis que requieren comprensión de negaciones y entidades nombradas.

    Parámetros:
    - text (str): El texto a limpiar.

    Retorna:
    - str: Texto limpio y lematizado, con contracciones expandidas y negaciones manejadas.
    """

    # Procesar el texto con spaCy
    doc = nlp(text)
    clean_tokens = []

    for token in doc:
        # Filtrar stopwords, puntuaciones, espacios y dígitos
        if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num:
            # Aplicar lematización
            lemma = token.lemma_.lower().strip()
            clean_tokens.append(lemma)

    return ' '.join(clean_tokens)


# Pipeline 2
# Función de limpieza de Texto Básica con NLTK
def clean_text(text, low_value_words=None):
    """
    Realiza una limpieza básica del texto utilizando herramientas de NLTK, adecuado para modelos basados en bolsa de palabras.
    Incluye expansión de contracciones, eliminación de HTML/URLs, y conversión de números a palabras.

    Parámetros:
    - text (str): El texto a limpiar.
    - low_value_words: Palabras comunes de poco valor semántico.

    Retorna:
    - str: Texto limpio con lematización, stemming y conversión de números.
    """

    if low_value_words is None:
        low_value_words = {'product', 'really', 'thing'}

    # Definiciones predeterminadas y limpieza inicial
    tokenizer = RegexpTokenizer(r'\w+')
    sw_list = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    clean_text = []

    # Proceso de limpieza detallado
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\bbr\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')

    for word in tokenizer.tokenize(text):
        word = lemmatizer.lemmatize(word)
        word = stemmer.stem(word)
        if word not in sw_list and word not in low_value_words:
            if word.isdigit():
                word = num2words(word, lang='en')
            clean_text.append(word)

    return ' '.join(clean_text)

# Pipeline 3
# Función de limpieza de Texto para Modelos Contextuales


def clean_text_for_contextual_models(text):
    """
    Ajusta la limpieza de texto para ser compatible con modelos contextuales, como BERT.
    Mantiene más la estructura original del texto y preserva el contexto, optimizando para análisis detallados.

    Parámetros:
    - text (str): El texto a limpiar.

    Retorna:
    - str: Texto limpio que preserva el contexto y la estructura original para análisis con modelos contextuales.
    """

    # Limpieza inicial y preparación del texto
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\bbr\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')

    # Minimizar y expandir contracciones
    text = text.lower()
    text = contractions.fix(text)

    # Tokenización
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # Filtrado de stopwords y conversión de números
    sw_list = stopwords.words('english')
    clean_tokens = [
        num2words(token, lang='en') 
        if token.isdigit()
        else token for token in tokens if token not in sw_list
    ]

    return ' '.join(clean_tokens)

# Pipeline 4
# Función de limpieza que excluye los números
def clean_text_exclude_numbers(text):
    """
    Realiza una limpieza del texto excluyendo los números.
    
    Parámetros:
    - text (str): El texto a limpiar.

    Retorna:
    - str: Texto limpio sin números y sin información no deseada.
    """

    # Eliminar URLs y elementos HTML
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\bbr\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()

    # Normalizar y eliminar caracteres especiales
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Minimizar texto y expandir contracciones
    text = text.lower()
    text = contractions.fix(text)

    # Inicializar tokenizador y listas de stopwords
    tokenizer = RegexpTokenizer(r'\w+')
    sw_list = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    # Tokenizar y limpiar
    clean_tokens = [
        lemmatizer.lemmatize(token.lower())  # Lematizar y llevar a minúsculas
        for token in tokenizer.tokenize(text)
        if token.lower() not in sw_list and not token.isdigit()  # Excluir stopwords y dígitos
    ]

    # Unir tokens en una cadena de texto limpio
    return ' '.join(clean_tokens)

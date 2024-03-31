## Descripción de los Conjuntos de Datos

Este directorio contiene los datos utilizados y generados durante el proyecto. Los datos originales se obtuvieron de un archivo `.jsonl.gz` de reseñas de productos de deporte y aire libre de Amazon.

## Contenido del Directorio

- `cleaned_reviews.csv`: Reseñas procesadas y limpias listas para análisis.
- `processed_data.csv.gz`: Datos preprocesados en formato comprimido para ahorro de espacio.
- `processed_data.parquet`: Versión Parquet de los datos procesados, optimizada para carga rápida en pandas.
- `README.md`: Documentación que describe el contenido y la estructura de este directorio.
- `sample_data_balanced_complete.csv.gz`: Muestra balanceada y completa de datos comprimidos para un manejo eficiente.
- `sample_data_balanced.csv`: Una versión más pequeña y balanceada de la muestra de datos para análisis exploratorio.
- `sample_data.csv`: Muestra inicial de datos antes del procesamiento.
- `sample_data.parquet`: Muestra de datos en formato Parquet.

## Uso de los Datos

Los archivos `.csv` pueden ser cargados directamente en pandas para análisis y modelado. Los archivos `.parquet` ofrecen una carga más rápida y un uso eficiente de la memoria. Los archivos comprimidos `.gz` pueden requerir descompresión antes del uso dependiendo de su entorno de trabajo.

## Nota Sobre Grandes Archivos

Debido a restricciones de tamaño en GitHub, decidimos no subir archivos que excedan los 100 MB. Por lo tanto, el archivo `Sports_and_Outdoors.jsonl.gz.zip` no se incluye en el repositorio remoto. Puede descargarse en [este enlace](https://amazon-reviews-2023.github.io/). Para los archivos grandes, considera utilizar `git-lfs` (Git Large File Storage) o proveer enlaces para su descarga desde un almacenamiento en la nube.


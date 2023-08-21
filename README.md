![Encabezado de Overleaf](https://drive.google.com/uc?id=1nqQ0SQ8hMZcjIFJlHjLza8Toza1J8YtI)

# Tesis de Titulación - Predicción de la Contaminación del Aire en la Región del Maule, Chile 🌍

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Resumen 📑

La contaminación del aire es una creciente preocupación para la salud pública en todo el mundo. En esta tesis, presentamos un enfoque innovador para abordar este problema en la Región del Maule, Chile. Utilizamos redes neuronales recurrentes profundas para predecir la contaminación del aire en múltiples ubicaciones. Nuestros resultados indican que el modelo puede predecir con precisión hasta 6 horas en el futuro en ubicaciones no monitoreadas previamente.

## Aspectos Destacados 🚀

- Propuesta de modelo de redes neuronales recurrentes profundas.
- Predicción de la contaminación del aire en múltiples ubicaciones.
- Utilización de datos de calidad del aire y de la API de OpenWeather.
- Implementación en Python.
- Uso de QGIS para la creación de una cuadrícula de puntos equidistantes.

## Fuentes de Datos 📊

En este enfoque, los datos provienen de dos fuentes principales:

1. **Datos de Calidad del Aire**: Obtuvimos datos de calidad del aire del Sistema de Información Nacional de Calidad del Aire de Chile (SINCA). Seleccionamos estaciones en diferentes áreas de la Región del Maule, incluyendo La Florida (LF), Universidad Católica del Maule (UCM) y Universidad de Talca (UTAL).

2. **Datos de OpenWeather API**: Además, recolectamos datos adicionales de la API de OpenWeather. OpenWeather es una organización sin fines de lucro que proporciona una amplia cantidad de datos globales sobre el clima y la calidad del aire, con opciones gratuitas y de pago. Utilizamos la API de Contaminación del Aire, que ofrece un millón de consultas gratuitas al mes.

## Creación de la Cuadrícula de Puntos 🗺️

Utilizamos QGIS, un Sistema de Información Geográfica (GIS), para crear una cuadrícula de puntos equidistantes que cubre toda la Región del Maule. Esta cuadrícula nos permitió evaluar la contaminación del aire en puntos específicos.

## Conclusión y Futuro 📝

Nuestro enfoque puede ser útil para predecir niveles de PM$_{2.5}$ en lugares que carecen de estaciones de monitoreo. Ideas para futuros trabajos incluyen:

1. Considerar otros contaminantes del aire y variables bioclimáticas.
2. Generar polígonos de predicción para lugares con características geológicas similares.
3. Probar el modelo en una cuadrícula de múltiples ubicaciones utilizando datos obtenidos de una API.

¡Gracias por visitar mi repositorio! 😊

2. Generar polígonos de predicción para lugares con características geológicas similares.
3. Probar el modelo en una cuadrícula de múltiples ubicaciones utilizando datos obtenidos de una API.

¡Gracias por visitar nuestro repositorio! 😊

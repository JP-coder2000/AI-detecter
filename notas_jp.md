## Cosas importantes a mencionar en nuestro paper.

1. Faltan datasets relevantes en el tema
2. Casi siempre los clasifican en 4 los tipos de code similarity:

• Type I: The code snippets are entirely the same except for changes that may exist in the white spaces and comments.
• Type II: The structure of the code snippets is the same. However, the identifiers' names, types, white spaces, and
comments differ.
• Type III: In this type, in addition to changes in identifiers, variable names, data types, and comments, some parts of
the code can be deleted or updated, or some new parts can be added.
• Type IV: Two pieces of code have different texts but the same functionality.



## Metodos de analizar los codigos o textos:

* Basados en texto
* Basados en tokens
* Basados en árboles
* Basados en grafos
* Basados en métricas
* Basados en imágenes
* Basados en aprendizaje automático
* Basados en pruebas


Como pienso yo que es una buena idea avanzar:

#### Cinco etapas:

Creación de corpus: Recopilar ejemplos de textos generados por IA y textos escritos por humanos

* Pre-procesamiento: Aplicar técnicas de PLN para extraer características relevantes
* Entrenamiento: Usar aprendizaje automático supervisado (SVM, redes neuronales)
* Producción: Implementar el modelo para evaluar nuevos textos
* Prueba: Evaluar la efectividad mediante métricas como precisión, recall y F1


Tipos de features que tenemos que extraer:

* Características léxicas (patrones de tokens y vocabulario)
* Características sintácticas (estructuras gramaticales)
* Características semánticas (coherencia y relaciones entre conceptos)


Técnicas de aprendizaje automático que podemos usae:

Redes Neuronales Artificiales (RNA) (yo creo que esta es la mejor)
Máquinas de Vectores de Soporte (SVM)
Modelos híbridos que combinen diferentes técnicas


Tipos de similitudes a detectar (adaptando los tipos de clones de código):

Tipo-1: Textos casi idénticos (pequeñas variaciones)
Tipo-2: Textos con misma estructura pero diferentes palabras
Tipo-3: Textos con estructura similar pero con adiciones/eliminaciones
Tipo-4: Textos semánticamente similares pero con sintaxis diferente
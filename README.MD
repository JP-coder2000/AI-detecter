# AI-Detecter

**AI-Detecter** es un sistema diseñado para detectar plagio en textos utilizando modelos de aprendizaje automático, como SVM y CNN. Este proyecto incluye herramientas para preprocesar datos, entrenar modelos y evaluar su rendimiento.

## Estructura del Proyecto

```
.gitignore  
README.MD  
requirements.txt  
models/  
    obfuscation_results.csv  
    results.png  
src/  
    create_dataset.py  
    test_english.py  
    test.py  
    train_and_evaluate.py  
    data/  
        processed/  
            dataset_info.txt  
            metadata_test.csv  
            metadata_train.csv  
    features/  
        feature_extractor.py  
        __pycache__/  
            feature_extractor.cpython-312.pyc  
    integration/  
        hybrid_model.py  
        __pycache__/  
    models/  
        cnn_model.py  
        svm_model.py  
        __pycache__/  
    preprocessing/  
        corpus_loader.py  
        text_processor.py  
        __pycache__/  
```

## Requisitos

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes dependencias:

- Python 3.8 o superior  
- Paquetes listados en `requirements.txt`

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Uso

### 1. Preprocesamiento de Datos

Utiliza el script `corpus_loader.py` para cargar y procesar los datos del corpus PAN2011:

```python
from preprocessing.corpus_loader import PanCorpusLoader

loader = PanCorpusLoader(corpus_path="ruta/a/corpus")
source_docs, suspicious_docs = loader.load_documents(task_type="external")
```

### 2. Procesamiento de Texto

El archivo `text_processor.py` contiene la clase `TextProcessor` que permite preprocesar texto. Por ejemplo:

```python
from preprocessing.text_processor import TextProcessor

processor = TextProcessor()
processed_text = processor.preprocess_text(
    text="Este es un ejemplo de texto.",
    remove_stopwords=True,
    stem_words=False,
    lemmatize=True
)
print(processed_text)
```

### 3. Entrenamiento de Modelos

Entrena los modelos SVM y CNN utilizando `train_and_evaluate.py`:

```bash
python src/train_and_evaluate.py
```

### 4. Pruebas

Ejecuta pruebas para evaluar el rendimiento del sistema:

```bash
python src/test.py
```

### 5. Optimización de Hiperparámetros

Optimiza los hiperparámetros del modelo SVM con:

```python
from models.svm_model import PlagiarismSVM

svm = PlagiarismSVM()
best_params = svm.optimize_hyperparameters(X, y)
print(best_params)
```

## Resultados

Es necesario ejecutar el código para ver los resultados.

## Contribuciones

Si deseas contribuir a este proyecto, por favor abre un issue o envía un pull request.


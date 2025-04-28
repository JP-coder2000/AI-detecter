import os
import sys
from preprocessing.corpus_loader import PanCorpusLoader
from preprocessing.text_processor import TextProcessor

def check_directory_structure(corpus_path, task_type="external"):
    """Verifica la estructura del directorio del corpus según el tipo de tarea"""
    print(f"=== Verificando estructura del corpus para tarea '{task_type}' ===")
    
    if not os.path.exists(corpus_path):
        print(f"Error: El directorio del corpus no existe: {corpus_path}")
        return False
    
    # Seleccionar la ruta base según el tipo de tarea
    base_path = os.path.join(corpus_path, f"{task_type}-detection-corpus")
    if not os.path.exists(base_path):
        print(f"Error: No se encuentra {task_type}-detection-corpus")
        return False
    
    # Verificar suspicious-document (presente en ambos tipos)
    suspicious_path = os.path.join(base_path, "suspicious-document")
    if not os.path.exists(suspicious_path):
        print(f"Error: No se encuentra suspicious-document")
        return False
    
    # Verificar source-document (solo para detección externa)
    if task_type == "external":
        source_path = os.path.join(base_path, "source-document")
        if not os.path.exists(source_path):
            print(f"Error: No se encuentra source-document")
            return False
        
        # Verificar carpetas part* en source-document
        source_parts = [d for d in os.listdir(source_path) if d.startswith("part") and os.path.isdir(os.path.join(source_path, d))]
        print(f"Carpetas part* en source-document: {len(source_parts)}")
        
        if not source_parts:
            print("Error: No se encontraron carpetas part* en source-document")
            return False
    
    # Verificar carpetas part* en suspicious-document
    suspicious_parts = [d for d in os.listdir(suspicious_path) if d.startswith("part") and os.path.isdir(os.path.join(suspicious_path, d))]
    print(f"Carpetas part* en suspicious-document: {len(suspicious_parts)}")
    
    if not suspicious_parts:
        print("Error: No se encontraron carpetas part* en suspicious-document")
        return False
    
    # Verificar que la cantidad de carpetas sea la esperada
    max_parts = 23 if task_type == "external" else 10
    if max(int(p.replace("part", "")) for p in suspicious_parts) > max_parts:
        print(f"Advertencia: Se encontraron carpetas part* con números mayores a {max_parts}")
    
    print("Estructura del corpus verificada correctamente")
    return True


def main():
    # Definir la ruta al corpus PAN
    corpus_path = os.path.join('data', 'pan-plagiarism-corpus-2011')
    
    # Verificar la estructura del corpus
    if not check_directory_structure(corpus_path):
        print("Error en la estructura del corpus. Verifique las rutas y la estructura de directorios.")
        sys.exit(1)
    
    # Inicializar el cargador del corpus
    loader = PanCorpusLoader(corpus_path)
    
    # Cargar un subconjunto pequeño para pruebas (5 documentos)
    print("\n=== Cargando documentos ===")
    source_docs, suspicious_docs = loader.load_documents(task_type="external", limit=5)
    
    print(f"\nDocumentos fuente cargados: {len(source_docs)}")
    if source_docs:
        print("Primeros documentos fuente:")
        for i, (doc_id, _) in enumerate(list(source_docs.items())[:3]):
            print(f"  {i+1}. {doc_id}")
    
    print(f"\nDocumentos sospechosos cargados: {len(suspicious_docs)}")
    if suspicious_docs:
        print("Primeros documentos sospechosos:")
        for i, (doc_id, _) in enumerate(list(suspicious_docs.items())[:3]):
            print(f"  {i+1}. {doc_id}")
    
    # Obtener los casos de plagio para estos documentos
    print("\n=== Cargando casos de plagio ===")
    plagiarism_cases = loader.load_plagiarism_cases(task_type="external")
    print(f"\nCasos de plagio cargados: {len(plagiarism_cases)}")
    
    # Mostrar información del primer caso de plagio
    if plagiarism_cases:
        print("\nPrimer caso de plagio:")
        for key, value in plagiarism_cases[0].items():
            print(f"  {key}: {value}")
    
    # Inicializar el procesador de texto
    processor = TextProcessor(language='english')
    
    # Procesar un documento sospechoso
    if suspicious_docs:
        first_doc_id = list(suspicious_docs.keys())[0]
        first_doc_text = suspicious_docs[first_doc_id]
        
        print(f"\n=== Procesando documento: {first_doc_id} ===")
        
        # Aplicar preprocesamiento
        processed = processor.preprocess_text(
            first_doc_text[:1000],  # Procesar solo los primeros 1000 caracteres para demostración
            remove_stopwords=True,
            stem_words=False,
            lemmatize=True
        )
        
        print(f"Número de oraciones: {processed['n_sentences']}")
        
        print("\nTexto original (primeros 200 caracteres):")
        print(processed['original'][:200])
        
        print("\nTexto procesado (primeros 200 caracteres):")
        print(processed['processed'][:200])
        
        # Generar n-gramas
        if processed['sentences']:
            print("\nN-gramas de caracteres (n=5) para la primera oración:")
            char_ngrams = processor.get_character_ngrams(processed['sentences'][0], 5)
            print(char_ngrams[:10])  # Mostrar solo los primeros 10
            
            print("\nN-gramas de tokens (n=3) para la primera oración:")
            token_ngrams = processor.get_token_ngrams(processed['sentences'][0], 3)
            print(token_ngrams)

if __name__ == "__main__":
    main()
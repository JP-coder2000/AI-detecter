import os
import numpy as np
from preprocessing.corpus_loader import PanCorpusLoader
from preprocessing.text_processor import TextProcessor
from features.feature_extractor import FeatureExtractor

def main():
    # Definir la ruta al corpus PAN
    corpus_path = os.path.join('data', 'pan-plagiarism-corpus-2011')
    
    # Inicializar cargador, procesador y extractor
    loader = PanCorpusLoader(corpus_path)
    processor = TextProcessor(language='english')
    extractor = FeatureExtractor()
    
    # Cargar una muestra pequeña de documentos
    print("Cargando documentos...")
    source_docs, suspicious_docs = loader.load_documents(task_type="external", limit=5)
    
    if not source_docs or not suspicious_docs:
        print("Error: No se pudieron cargar los documentos")
        return
    
    print(f"Documentos fuente cargados: {len(source_docs)}")
    print(f"Documentos sospechosos cargados: {len(suspicious_docs)}")
    
    # Cargar casos de plagio
    print("\nCargando casos de plagio...")
    plagiarism_cases = loader.load_plagiarism_cases(task_type="external")
    
    # Filtrar casos que involucren nuestros documentos cargados
    filtered_cases = []
    for case in plagiarism_cases:
        if (case['source_document'] in source_docs and 
            case['suspicious_document'] in suspicious_docs):
            filtered_cases.append(case)
    
    print(f"Casos de plagio relevantes: {len(filtered_cases)}")
    
    if not filtered_cases:
        print("No se encontraron casos de plagio entre los documentos cargados")
        # En este caso, vamos a crear un caso de prueba
        source_doc_id = list(source_docs.keys())[0]
        suspicious_doc_id = list(suspicious_docs.keys())[0]
        
        print(f"\nCreando un caso de prueba entre {source_doc_id} y {suspicious_doc_id}")
        
        source_text = source_docs[source_doc_id][:1000]  # Tomar los primeros 1000 caracteres
        suspicious_text = suspicious_docs[suspicious_doc_id][:1000]
        
        print("\nExtrayendo características...")
        features = extractor.extract_features_for_fragment_pair(source_text, suspicious_text)
        
        print("\nCaracterísticas SVM:")
        print(f"Dimensiones: {features['svm_features'].shape}")
        print(f"Valores: {features['svm_features']}")
        
        print("\nEntrada CNN:")
        print(f"Tamaño de vocabulario: {features['cnn_input']['vocab_size']}")
        print(f"Longitud de secuencia fuente: {len(features['cnn_input']['source_sequence'])}")
        print(f"Longitud de secuencia sospechosa: {len(features['cnn_input']['suspicious_sequence'])}")
        
        return
    
    # Si hay casos, tomamos el primero para analizar
    case = filtered_cases[0]
    print("\nAnalizando caso de plagio:")
    print(f"Documento fuente: {case['source_document']}")
    print(f"Documento sospechoso: {case['suspicious_document']}")
    print(f"Offset fuente: {case['source_offset']}, Longitud: {case['source_length']}")
    print(f"Offset sospechoso: {case['suspicious_offset']}, Longitud: {case['suspicious_length']}")
    
    # Extraer los fragmentos de texto
    source_text = source_docs[case['source_document']]
    suspicious_text = suspicious_docs[case['suspicious_document']]
    
    source_fragment = source_text[case['source_offset']:case['source_offset'] + case['source_length']]
    suspicious_fragment = suspicious_text[case['suspicious_offset']:case['suspicious_offset'] + case['suspicious_length']]
    
    # Preprocesar los fragmentos
    source_processed = processor.preprocess_text(source_fragment)['processed']
    suspicious_processed = processor.preprocess_text(suspicious_fragment)['processed']
    
    print("\nFragmento fuente (procesado):")
    print(source_processed[:200] + "...")
    
    print("\nFragmento sospechoso (procesado):")
    print(suspicious_processed[:200] + "...")
    
    # Extraer características
    print("\nExtrayendo características...")
    features = extractor.extract_features_for_fragment_pair(source_fragment, suspicious_fragment)
    
    print("\nCaracterísticas SVM:")
    print(f"Dimensiones: {features['svm_features'].shape}")
    print(f"Valores: {features['svm_features']}")
    
    print("\nEntrada CNN:")
    print(f"Tamaño de vocabulario: {features['cnn_input']['vocab_size']}")
    print(f"Longitud de secuencia fuente: {len(features['cnn_input']['source_sequence'])}")
    print(f"Longitud de secuencia sospechosa: {len(features['cnn_input']['suspicious_sequence'])}")

if __name__ == "__main__":
    main()
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

from preprocessing.corpus_loader import PanCorpusLoader
from preprocessing.text_processor import TextProcessor
from features.feature_extractor import FeatureExtractor

def create_dataset(corpus_path, max_examples=500, test_size=0.2, random_state=42):
    """
    Crea un dataset balanceado para entrenamiento y prueba.
    """
    print("Inicializando componentes...")
    processor = TextProcessor(language='english')
    extractor = FeatureExtractor()
    
    # PASO 1: Encontrar casos de plagio directamente de los XML
    print("Buscando casos de plagio en el corpus...")
    suspicious_path = os.path.join(corpus_path, "external-detection-corpus", "suspicious-document")
    
    plagiarism_cases = []
    source_docs_needed = set()
    suspicious_docs_needed = set()
    
    # Buscar en todas las carpetas part*
    for part_dir in sorted(os.listdir(suspicious_path)):
        if not part_dir.startswith("part"):
            continue
            
        part_path = os.path.join(suspicious_path, part_dir)
        print(f"Buscando en {part_dir}...")
        
        for filename in os.listdir(part_path):
            if not filename.endswith(".xml"):
                continue
                
            suspicious_id = filename[:-4]  # Quitar extensión .xml
            file_path = os.path.join(part_path, filename)
            
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                for feature in root.findall('.//feature[@name="plagiarism"]'):
                    source_ref = feature.get('source_reference')
                    
                    if source_ref:
                        # Quitar extensión .txt si existe
                        source_id = source_ref
                        if source_id.endswith('.txt'):
                            source_id = source_id[:-4]
                        
                        # Guardar caso de plagio
                        case = {
                            'suspicious_id': suspicious_id,
                            'source_id': source_id,
                            'suspicious_offset': int(feature.get('this_offset')),
                            'suspicious_length': int(feature.get('this_length')),
                            'source_offset': int(feature.get('source_offset')),
                            'source_length': int(feature.get('source_length')),
                            'type': feature.get('type'),
                            'obfuscation': feature.get('obfuscation')
                        }
                        
                        plagiarism_cases.append(case)
                        
                        # Añadir a documentos necesarios
                        source_docs_needed.add(source_id)
                        suspicious_docs_needed.add(suspicious_id)
                        
                        # Si ya tenemos suficientes casos, podemos parar
                        if len(plagiarism_cases) >= max_examples:
                            break
                
                if len(plagiarism_cases) >= max_examples:
                    break
                    
            except Exception as e:
                print(f"Error al procesar {file_path}: {str(e)}")
                
        if len(plagiarism_cases) >= max_examples:
            break
    
    print(f"Encontrados {len(plagiarism_cases)} casos de plagio")
    print(f"Documentos fuente necesarios: {len(source_docs_needed)}")
    print(f"Documentos sospechosos necesarios: {len(suspicious_docs_needed)}")
    
    if not plagiarism_cases:
        print("Error: No se encontraron casos de plagio. No se puede continuar.")
        return None
    
    # PASO 2: Cargar los documentos necesarios
    print("Cargando documentos fuente...")
    source_docs = {}
    source_path = os.path.join(corpus_path, "external-detection-corpus", "source-document")
    
    for part_dir in sorted(os.listdir(source_path)):
        if not part_dir.startswith("part"):
            continue
            
        part_path = os.path.join(source_path, part_dir)
        
        for filename in os.listdir(part_path):
            if not filename.endswith(".txt"):
                continue
                
            doc_id = filename[:-4]  # Quitar extensión .txt
            
            if doc_id in source_docs_needed:
                file_path = os.path.join(part_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    source_docs[doc_id] = content
                except Exception as e:
                    print(f"Error leyendo {file_path}: {e}")
    
    print("Cargando documentos sospechosos...")
    suspicious_docs = {}
    
    for part_dir in sorted(os.listdir(suspicious_path)):
        if not part_dir.startswith("part"):
            continue
            
        part_path = os.path.join(suspicious_path, part_dir)
        
        for filename in os.listdir(part_path):
            if not filename.endswith(".txt"):
                continue
                
            doc_id = filename[:-4]  # Quitar extensión .txt
            
            if doc_id in suspicious_docs_needed:
                file_path = os.path.join(part_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    suspicious_docs[doc_id] = content
                except Exception as e:
                    print(f"Error leyendo {file_path}: {e}")
    
    print(f"Documentos fuente cargados: {len(source_docs)}/{len(source_docs_needed)}")
    print(f"Documentos sospechosos cargados: {len(suspicious_docs)}/{len(suspicious_docs_needed)}")
    
    # PASO 3: Filtrar casos donde tenemos ambos documentos
    valid_cases = []
    for case in plagiarism_cases:
        if case['source_id'] in source_docs and case['suspicious_id'] in suspicious_docs:
            valid_cases.append(case)
    
    print(f"Casos válidos con ambos documentos disponibles: {len(valid_cases)}")
    
    if not valid_cases:
        print("Error: No hay casos válidos con ambos documentos disponibles.")
        return None
    
    # PASO 4: Generar ejemplos positivos y negativos
    examples = []
    
    # Generar ejemplos positivos
    print("Generando ejemplos positivos...")
    for case in tqdm(valid_cases[:max_examples//2]):
        source_text = source_docs[case['source_id']]
        suspicious_text = suspicious_docs[case['suspicious_id']]
        
        # Extraer fragmentos
        try:
            source_fragment = source_text[case['source_offset']:case['source_offset'] + case['source_length']]
            suspicious_fragment = suspicious_text[case['suspicious_offset']:case['suspicious_offset'] + case['suspicious_length']]
            
            if not source_fragment or not suspicious_fragment:
                continue
            
            # Extraer características
            features = extractor.extract_features_for_fragment_pair(source_fragment, suspicious_fragment)
            
            example = {
                'source_doc': case['source_id'],
                'suspicious_doc': case['suspicious_id'],
                'source_offset': case['source_offset'],
                'suspicious_offset': case['suspicious_offset'],
                'source_fragment': source_fragment[:500],
                'suspicious_fragment': suspicious_fragment[:500],
                'is_plagiarism': 1,
                'obfuscation': case.get('obfuscation', 'unknown'),
                'svm_features': features['svm_features'],
                'source_sequence': features['cnn_input']['source_sequence'],
                'suspicious_sequence': features['cnn_input']['suspicious_sequence']
            }
            
            examples.append(example)
        except Exception as e:
            print(f"Error procesando ejemplo positivo: {str(e)}")
    
    num_positives = len(examples)
    print(f"Ejemplos positivos generados: {num_positives}")
    
    # Generar ejemplos negativos
    print("Generando ejemplos negativos...")
    
    # Usar los mismos documentos pero fragmentos diferentes
    source_ids = list(source_docs.keys())
    suspicious_ids = list(suspicious_docs.keys())
    
    negative_count = 0
    max_attempts = num_positives * 5
    
    for _ in tqdm(range(max_attempts)):
        if negative_count >= num_positives:
            break
            
        # Seleccionar documentos al azar
        source_id = random.choice(source_ids)
        suspicious_id = random.choice(suspicious_ids)
        
        source_text = source_docs[source_id]
        suspicious_text = suspicious_docs[suspicious_id]
        
        if len(source_text) < 500 or len(suspicious_text) < 500:
            continue
        
        # Seleccionar fragmentos aleatorios
        source_start = random.randint(0, max(0, len(source_text) - 500))
        suspicious_start = random.randint(0, max(0, len(suspicious_text) - 500))
        
        source_length = min(500, len(source_text) - source_start)
        suspicious_length = min(500, len(suspicious_text) - suspicious_start)
        
        source_fragment = source_text[source_start:source_start + source_length]
        suspicious_fragment = suspicious_text[suspicious_start:suspicious_start + suspicious_length]
        
        # Verificar similitud
        similarity = extractor._jaccard_similarity(source_fragment, suspicious_fragment)
        if similarity > 0.3:  # Si son muy similares, podría ser plagio no detectado
            continue
        
        # Extraer características
        try:
            features = extractor.extract_features_for_fragment_pair(source_fragment, suspicious_fragment)
            
            example = {
                'source_doc': source_id,
                'suspicious_doc': suspicious_id,
                'source_offset': source_start,
                'suspicious_offset': suspicious_start,
                'source_fragment': source_fragment[:500],
                'suspicious_fragment': suspicious_fragment[:500],
                'is_plagiarism': 0,
                'obfuscation': 'none',
                'svm_features': features['svm_features'],
                'source_sequence': features['cnn_input']['source_sequence'],
                'suspicious_sequence': features['cnn_input']['suspicious_sequence']
            }
            
            examples.append(example)
            negative_count += 1
        except Exception as e:
            print(f"Error procesando ejemplo negativo: {str(e)}")
    
    print(f"Dataset generado: {len(examples)} ejemplos ({num_positives} positivos, {negative_count} negativos)")
    
    if len(examples) == 0:
        print("Error: No se generaron ejemplos. No se puede continuar.")
        return None
    
    # PASO 5: Preparar y guardar dataset
    # Convertir a DataFrame para facilitar manejo
    examples_df = pd.DataFrame([
        {
            'source_doc': ex['source_doc'],
            'suspicious_doc': ex['suspicious_doc'],
            'source_offset': ex['source_offset'],
            'suspicious_offset': ex['suspicious_offset'],
            'source_fragment': ex['source_fragment'],
            'suspicious_fragment': ex['suspicious_fragment'],
            'is_plagiarism': ex['is_plagiarism'],
            'obfuscation': ex['obfuscation']
        } for ex in examples
    ])
    
    # Extraer matrices de características
    X_svm = np.vstack([ex['svm_features'] for ex in examples])
    X_cnn_source = np.vstack([ex['source_sequence'].reshape(1, -1) for ex in examples])
    X_cnn_suspicious = np.vstack([ex['suspicious_sequence'].reshape(1, -1) for ex in examples])
    y = np.array([ex['is_plagiarism'] for ex in examples])
    
    # Dividir en entrenamiento y prueba
    indices = np.arange(len(examples))
    indices_train, indices_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y)
    
    # Preparar datos
    dataset = {
        'train': {
            'X_svm': X_svm[indices_train],
            'X_cnn_source': X_cnn_source[indices_train],
            'X_cnn_suspicious': X_cnn_suspicious[indices_train],
            'y': y[indices_train],
            'metadata': examples_df.iloc[indices_train]
        },
        'test': {
            'X_svm': X_svm[indices_test],
            'X_cnn_source': X_cnn_source[indices_test],
            'X_cnn_suspicious': X_cnn_suspicious[indices_test],
            'y': y[indices_test],
            'metadata': examples_df.iloc[indices_test]
        },
        'vocab_size': examples[0]['svm_features'].shape[0] if examples else 0,
        'max_sequence_length': examples[0]['source_sequence'].shape[0] if examples else 0
    }
    
    # Guardar dataset
    os.makedirs('data/processed', exist_ok=True)
    
    np.save('data/processed/X_svm_train.npy', dataset['train']['X_svm'])
    np.save('data/processed/X_cnn_source_train.npy', dataset['train']['X_cnn_source'])
    np.save('data/processed/X_cnn_suspicious_train.npy', dataset['train']['X_cnn_suspicious'])
    np.save('data/processed/y_train.npy', dataset['train']['y'])
    
    np.save('data/processed/X_svm_test.npy', dataset['test']['X_svm'])
    np.save('data/processed/X_cnn_source_test.npy', dataset['test']['X_cnn_source'])
    np.save('data/processed/X_cnn_suspicious_test.npy', dataset['test']['X_cnn_suspicious'])
    np.save('data/processed/y_test.npy', dataset['test']['y'])
    
    dataset['train']['metadata'].to_csv('data/processed/metadata_train.csv', index=False)
    dataset['test']['metadata'].to_csv('data/processed/metadata_test.csv', index=False)
    
    # Guardar información
    with open('data/processed/dataset_info.txt', 'w') as f:
        f.write(f"Total examples: {len(examples)}\n")
        f.write(f"Positive examples: {num_positives}\n")
        f.write(f"Negative examples: {negative_count}\n")
        f.write(f"SVM features dimension: {dataset['vocab_size']}\n")
        f.write(f"CNN sequence length: {dataset['max_sequence_length']}\n")
    
    return dataset

if __name__ == "__main__":
    corpus_path = os.path.join('data', 'pan-plagiarism-corpus-2011')
    dataset = create_dataset(
        corpus_path=corpus_path,
        max_examples=500,
        test_size=0.2
    )
    
    if dataset:
        print("Dataset creado y guardado en data/processed/")
    else:
        print("No se pudo crear el dataset")
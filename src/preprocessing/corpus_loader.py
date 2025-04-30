import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

class PanCorpusLoader:
    def __init__(self, corpus_path: str):
        """
        Inicializa el cargador del corpus PAN.
        
        Args:
            corpus_path: Ruta al directorio raíz del corpus PAN 2011
        """
        self.corpus_path = corpus_path
        self.external_corpus_path = os.path.join(corpus_path, "external-detection-corpus")
        self.intrinsic_corpus_path = os.path.join(corpus_path, "intrinsic-detection-corpus")
    
    def _list_parts(self, base_dir: str) -> List[str]: ## Estos son los directorios de part1 hasta la 23
        """Lista todos los directorios part* en la ruta especificada"""
        if not os.path.exists(base_dir):
            print(f"Advertencia: Directorio no encontrado: {base_dir}")
            return []
            
        return [d for d in os.listdir(base_dir) if d.startswith("part") and os.path.isdir(os.path.join(base_dir, d))]
        
    def load_documents(self, task_type: str = "external", limit: int = None) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Carga los documentos fuente y sospechosos.
        
        Args:
            task_type: Tipo de tarea ('external' o 'intrinsic')
            limit: Número máximo de documentos a cargar (útil para pruebas)
            
        Returns:
            Tuple de diccionarios con documentos fuente y sospechosos
        """
        if task_type not in ["external", "intrinsic"]:
            raise ValueError("task_type debe ser 'external' o 'intrinsic'")
        
        base_path = self.external_corpus_path if task_type == "external" else self.intrinsic_corpus_path
        
        # Cargar documentos sospechosos
        suspicious_docs = {}
        suspicious_path = os.path.join(base_path, "suspicious-document")
        
        if not os.path.exists(suspicious_path):
            print(f"Error: La ruta {suspicious_path} no existe")
            return {}, {}
        
        counter = 0
        
        # Listar todas las carpetas part*
        part_dirs = self._list_parts(suspicious_path)
        print(f"Procesando {len(part_dirs)} carpetas en {suspicious_path}")
        
        for part_dir in sorted(part_dirs):
            part_path = os.path.join(suspicious_path, part_dir)
            print(f"Cargando documentos sospechosos de {part_dir}...")
            
            # Filtrar solo archivos de texto
            text_files = [f for f in os.listdir(part_path) if f.endswith(".txt")]
            
            for filename in text_files:
                doc_id = filename[:-4]  # Quitar extensión .txt
                file_path = os.path.join(part_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    suspicious_docs[doc_id] = content
                    
                    counter += 1
                    if limit and counter >= limit:
                        break
                except Exception as e:
                    print(f"Error al leer {file_path}: {str(e)}")
            
            if limit and counter >= limit:
                break
        
        # Cargar documentos fuente (solo para detección externa)
        source_docs = {}
        if task_type == "external":
            source_path = os.path.join(base_path, "source-document")
            
            if not os.path.exists(source_path):
                print(f"Error: La ruta {source_path} no existe")
                return source_docs, suspicious_docs
            
            counter = 0
            
            # Listar todas las carpetas part*
            part_dirs = self._list_parts(source_path)
            print(f"Procesando {len(part_dirs)} carpetas en {source_path}")
            
            for part_dir in sorted(part_dirs):
                part_path = os.path.join(source_path, part_dir)
                print(f"Cargando documentos fuente de {part_dir}...")
                
                # Filtrar solo archivos de texto
                text_files = [f for f in os.listdir(part_path) if f.endswith(".txt")]
                
                for filename in text_files:
                    doc_id = filename[:-4]  # Quitar extensión .txt
                    file_path = os.path.join(part_path, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        source_docs[doc_id] = content
                        
                        counter += 1
                        if limit and counter >= limit:
                            break
                    except Exception as e:
                        print(f"Error al leer {file_path}: {str(e)}")
                
                if limit and counter >= limit:
                    break
        
        return source_docs, suspicious_docs
    
    def load_plagiarism_cases(self, task_type: str = "external") -> List[Dict]:
        """
        Carga las anotaciones de plagio de los documentos sospechosos.
        
        Args:
            task_type: Tipo de tarea ('external' o 'intrinsic')
            
        Returns:
            Lista de diccionarios con información de casos de plagio
        """
        if task_type not in ["external", "intrinsic"]:
            raise ValueError("task_type debe ser 'external' o 'intrinsic'")
        
        base_path = self.external_corpus_path if task_type == "external" else self.intrinsic_corpus_path
        suspicious_path = os.path.join(base_path, "suspicious-document")
        
        if not os.path.exists(suspicious_path):
            print(f"Error: La ruta {suspicious_path} no existe")
            return []
        
        plagiarism_cases = []
        
        # Listar todas las carpetas part*
        part_dirs = self._list_parts(suspicious_path)
        print(f"Procesando {len(part_dirs)} carpetas para casos de plagio")
        
        for part_dir in sorted(part_dirs):
            part_path = os.path.join(suspicious_path, part_dir)
            print(f"Cargando anotaciones de plagio de {part_dir}...")
            
            # Filtrar solo archivos XML
            xml_files = [f for f in os.listdir(part_path) if f.endswith(".xml")]
            
            for filename in xml_files:
                doc_id = filename[:-4]  # Quitar extensión .xml
                file_path = os.path.join(part_path, filename)
                
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    
                    # CORRECCIÓN: Buscar específicamente elementos con name="plagiarism"
                    for feature in root.findall('.//feature[@name="plagiarism"]'):
                        try:
                            # Obtener atributos con manejo seguro
                            this_offset = feature.get('this_offset')
                            this_length = feature.get('this_length')
                            
                            if this_offset is None or this_length is None:
                                continue
                            
                            case = {
                                'suspicious_document': doc_id,
                                'suspicious_offset': int(this_offset),
                                'suspicious_length': int(this_length)
                            }
                            
                            # Solo para detección externa
                            if task_type == "external":
                                source_reference = feature.get('source_reference')
                                source_offset = feature.get('source_offset')
                                source_length = feature.get('source_length')
                                
                                if all([source_reference, source_offset, source_length]):
                                    case['source_document'] = source_reference
                                    case['source_offset'] = int(source_offset)
                                    case['source_length'] = int(source_length)
                                
                            # Información adicional
                            for attr in ['type', 'obfuscation']:
                                if feature.get(attr):
                                    case[attr] = feature.get(attr)
                                
                            plagiarism_cases.append(case)
                        except (TypeError, ValueError) as e:
                            print(f"Error en feature de {file_path}: {str(e)}")
                except Exception as e:
                    print(f"Error al procesar {file_path}: {str(e)}")
        
        return plagiarism_cases
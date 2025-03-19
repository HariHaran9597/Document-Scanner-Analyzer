from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import spacy
from spacy.tokens import Span
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import List, Dict, Tuple
import torch

class AdvancedNLPProcessor:
    def __init__(self):
        # Load spaCy model with entity linker
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
            self.nlp = spacy.load("en_core_web_lg")
        
        # Initialize Hugging Face transformers
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        
        # Initialize handwriting recognition model
        self.htr_model = hub.load("https://tfhub.dev/sayakpaul/handwriting-recognition/1")
        
        # Custom entity patterns for domain-specific entities
        self.custom_patterns = [
            {"label": "MEDICAL_TERM", "pattern": [{"LOWER": {"REGEX": r"[a-z]+itis$|[a-z]+oma$"}}]},
            {"label": "LEGAL_CLAUSE", "pattern": [{"LOWER": "section"}, {"SHAPE": "dd"}]},
            {"label": "INVOICE_REF", "pattern": [{"LIKE_NUM": True}, {"ORTH": "/"}, {"LIKE_NUM": True}]}
        ]
        self.add_custom_patterns()

    def add_custom_patterns(self):
        """Add custom patterns to the spaCy pipeline"""
        ruler = self.nlp.get_pipe("entity_ruler") if "entity_ruler" in self.nlp.pipe_names else self.nlp.add_pipe("entity_ruler")
        ruler.add_patterns(self.custom_patterns)

    def process_text(self, text: str) -> Dict:
        """Process text with advanced NLP features"""
        doc = self.nlp(text)
        
        # Entity linking
        linked_entities = self._link_entities(doc)
        
        # Relation extraction
        relations = self._extract_relations(doc)
        
        # Custom NER with transformers
        custom_entities = self._custom_ner(text)
        
        return {
            'linked_entities': linked_entities,
            'relations': relations,
            'custom_entities': custom_entities,
            'doc': doc
        }

    def _link_entities(self, doc) -> List[Dict]:
        """Link entities to knowledge bases"""
        linked_entities = []
        
        for ent in doc.ents:
            # Get most similar entities from the vocabulary
            similar_concepts = []
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT']:
                vector = ent.vector
                similar_concepts = [
                    (token.text, token.similarity(ent))
                    for token in self.nlp.vocab
                    if token.has_vector and token.is_alpha and len(token.text) > 1
                ]
                similar_concepts = sorted(similar_concepts, key=lambda x: x[1], reverse=True)[:3]
            
            linked_entities.append({
                'text': ent.text,
                'label': ent.label_,
                'similar_concepts': similar_concepts,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return linked_entities

    def _extract_relations(self, doc) -> List[Dict]:
        """Extract relationships between entities"""
        relations = []
        
        for sent in doc.sents:
            ents = list(sent.ents)
            
            for i, ent1 in enumerate(ents):
                for ent2 in ents[i+1:]:
                    # Find the path between entities
                    path = []
                    for token in sent:
                        if token.is_ancestor(ent1.root) and token.is_ancestor(ent2.root):
                            path = self._get_path(token, ent1.root) + [token] + self._get_path(token, ent2.root)
                            break
                    
                    if path:
                        relation = ' '.join([t.text for t in path if t.dep_ not in ['punct', 'det']])
                        relations.append({
                            'entity1': {'text': ent1.text, 'label': ent1.label_},
                            'entity2': {'text': ent2.text, 'label': ent2.label_},
                            'relation': relation
                        })
        
        return relations

    def _get_path(self, start, end):
        """Get dependency path between tokens"""
        path = []
        current = end
        while current != start and current != None:
            path.append(current)
            current = current.head
        return path[::-1]

    def _custom_ner(self, text: str) -> List[Dict]:
        """Perform custom NER using transformers"""
        entities = self.ner_pipeline(text)
        
        # Group consecutive tokens of the same entity
        grouped_entities = []
        current_entity = None
        
        for ent in entities:
            if current_entity and current_entity['entity'] == ent['entity'] and \
               current_entity['end'] == ent['start']:
                current_entity['word'] += ' ' + ent['word']
                current_entity['end'] = ent['end']
                current_entity['score'] = (current_entity['score'] + ent['score']) / 2
            else:
                if current_entity:
                    grouped_entities.append(current_entity)
                current_entity = ent
        
        if current_entity:
            grouped_entities.append(current_entity)
        
        return grouped_entities

    def process_handwritten_text(self, image_array: np.ndarray) -> Dict:
        """Process handwritten text in an image"""
        # Ensure image is in the correct format
        if len(image_array.shape) == 3:
            # Convert to grayscale if needed
            if image_array.shape[2] == 3:
                image_array = np.mean(image_array, axis=2).astype(np.float32)
        
        # Normalize image
        image_array = image_array / 255.0
        
        # Resize image to model's expected input size
        image_array = tf.image.resize(
            tf.expand_dims(image_array, axis=0),
            [32, 128],
            method='bilinear'
        )
        
        # Run inference
        predictions = self.htr_model(image_array)
        
        # Process predictions
        decoded_text = self._decode_predictions(predictions)
        
        return {
            'text': decoded_text,
            'confidence': float(tf.reduce_mean(predictions['confidence']))
        }

    def _decode_predictions(self, predictions) -> str:
        """Decode model predictions into text"""
        # This is a simplified decoder - actual implementation would depend on the specific model
        return predictions['text'].numpy()[0].decode('utf-8')

    def train_custom_ner(self, training_data: List[Tuple[str, Dict[str, List[Tuple[int, int, str]]]]], epochs: int = 10):
        """Train a custom NER model using spaCy"""
        # Convert training data to spaCy format
        train_data = []
        for text, annotations in training_data:
            entities = []
            for start, end, label in annotations['entities']:
                entities.append(Span(self.nlp.make_doc(text), start, end, label=label))
            train_data.append((text, {'entities': entities}))
        
        # Configure pipeline
        pipeline = self.nlp.create_pipe("ner")
        for _, annotations in train_data:
            for ent in annotations['entities']:
                pipeline.add_label(ent.label_)
        
        # Train the model
        optimizer = self.nlp.begin_training()
        for _ in range(epochs):
            losses = {}
            for text, annotations in train_data:
                self.nlp.update([text], [annotations], sgd=optimizer, losses=losses)
        
        return losses
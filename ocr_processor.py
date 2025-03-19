import pytesseract
import re
from PIL import Image
import spacy
import pandas as pd
import os
import json
from datetime import datetime
from fpdf import FPDF
import cv2
import numpy as np
import sys
from advanced_nlp import AdvancedNLPProcessor

class OCRProcessor:
    def __init__(self):
        # Initialize advanced NLP processor
        self.advanced_nlp = AdvancedNLPProcessor()
        
        # Basic OCR configuration
        self.tesseract_config = '--oem 3 --psm 3 -c textord_heavy_nr=1 -c textord_min_linesize=2.5'
        
        # Enhanced regex patterns
        self.patterns = {
            'date': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
            'amount': r'\$?\s*\d+(?:,\d{3})*\.?\d*\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'invoice_number': r'\b(?:INV|INVOICE|inv)[-\s]?\d+\b',
            'product_code': r'\b[A-Z]{2,3}[-]\d{3,4}[-][A-Z0-9]{2,4}\b',
            'postal_code': r'\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b|\b\d{5}(?:[-]\d{4})?\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){4}\b',
            'tracking_number': r'\b\d{12,14}\b|\b[A-Z]{2}\d{9}[A-Z]{2}\b',
            'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*',
            'social_security': r'\b\d{3}[-]\d{2}[-]\d{4}\b',
            'bank_account': r'\b\d{8,17}\b'
        }

    def extract_text_with_regions(self, image):
        """Extract text with bounding box information"""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Enhance image size for better OCR if needed
        width, height = image.size
        scale_factor = 1
        if width < 1000 or height < 1000:
            scale_factor = 2000 / min(width, height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Get detailed OCR data including bounding boxes with enhanced config
        data = pytesseract.image_to_data(
            image, 
            output_type=pytesseract.Output.DICT,
            config=self.tesseract_config
        )
        
        text = []
        regions = []
        
        # Process OCR results with improved confidence handling
        n_boxes = len(data['text'])
        conf_threshold = 40  # Lower threshold for initial capture
        
        for i in range(n_boxes):
            conf = int(data['conf'][i])
            if conf > conf_threshold:
                text_block = data['text'][i].strip()
                if text_block:  # Only process non-empty text
                    # Adjust coordinates back to original scale
                    x = int(data['left'][i] / scale_factor)
                    y = int(data['top'][i] / scale_factor)
                    w = int(data['width'][i] / scale_factor)
                    h = int(data['height'][i] / scale_factor)
                    
                    # Apply confidence boost for numeric and common patterns
                    if re.match(r'^\d+$', text_block):  # Numbers
                        conf = min(conf * 1.2, 100)
                    elif re.match(r'^[A-Za-z]+$', text_block):  # Words
                        conf = min(conf * 1.1, 100)
                    
                    text.append(text_block)
                    regions.append({
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'text': text_block,
                        'conf': conf
                    })
        
        # Sort regions by vertical position for natural reading order
        regions.sort(key=lambda r: (r['y'], r['x']))
        text = ' '.join(text)
        
        return text, regions

    def categorize_text(self, text):
        """Enhanced text categorization using spaCy NER and regex patterns"""
        doc = self.nlp(text)
        
        # Initialize categories
        categories = {
            'dates': [],
            'amounts': [],
            'emails': [],
            'phones': [],
            'names': [],
            'organizations': [],
            'locations': [],
            'job_titles': [],
            'invoice_numbers': [],
            'product_codes': [],
            'postal_codes': [],
            'credit_cards': [],
            'tracking_numbers': [],
            'urls': [],
            'bank_accounts': [],
            'social_security': [],
            'uncategorized': []
        }
        
        # Extract entities using spaCy NER
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                categories['names'].append(ent.text)
            elif ent.label_ == 'ORG':
                categories['organizations'].append(ent.text)
            elif ent.label_ in ['GPE', 'LOC']:
                categories['locations'].append(ent.text)
            elif ent.label_ == 'WORK_OF_ART':
                categories['job_titles'].append(ent.text)

        # Apply regex patterns
        for category, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if category in categories:
                categories[category].extend(matches)

        # Remove duplicates and clean up
        for category in categories:
            categories[category] = list(set(categories[category]))
        
        return categories

    def create_pdf_report(self, results, image_path=None, output_path=None):
        """Create an enhanced PDF report with NLP insights"""
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Document Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Add image if provided
        if image_path:
            try:
                pdf.image(image_path, x=10, y=30, w=190)
                pdf.ln(100)
            except:
                pass
        
        # Add document type
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"Document Type: {'Handwritten' if results.get('is_handwritten') else 'Printed'}", 0, 1, 'L')
        
        # Add raw text
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Extracted Text:', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, results['raw_text'])
        pdf.ln(10)
        
        # Add NLP insights
        if 'nlp_analysis' in results:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'NLP Analysis:', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            
            # Linked entities
            if results['nlp_analysis']['linked_entities']:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 5, 'Linked Entities:', 0, 1, 'L')
                pdf.set_font('Arial', '', 10)
                for entity in results['nlp_analysis']['linked_entities']:
                    pdf.cell(0, 5, f"- {entity['text']} ({entity['label']})", 0, 1, 'L')
                    if entity['similar_concepts']:
                        pdf.cell(5)  # indent
                        concepts = ', '.join([f"{c[0]} ({c[1]:.2f})" for c in entity['similar_concepts']])
                        pdf.multi_cell(0, 5, f"Similar concepts: {concepts}")
                pdf.ln(5)
            
            # Relations
            if results['nlp_analysis']['relations']:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 5, 'Entity Relations:', 0, 1, 'L')
                pdf.set_font('Arial', '', 10)
                for relation in results['nlp_analysis']['relations']:
                    pdf.multi_cell(0, 5, 
                        f"- {relation['entity1']['text']} → {relation['relation']} → {relation['entity2']['text']}")
                pdf.ln(5)
        
        # Add categorized information
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Categorized Information:', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        for category, items in results['categorized'].items():
            if items:
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 5, category.title() + ':', 0, 1, 'L')
                pdf.set_font('Arial', '', 10)
                for item in items:
                    pdf.cell(0, 5, f'- {item}', 0, 1, 'L')
                pdf.ln(2)
        
        # Save PDF
        if output_path is None:
            output_path = f'exports/report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pdf.output(output_path)
        return output_path

    def export_results(self, results, output_dir='exports', format='all'):
        """Export results in various formats"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exported_files = {}
        
        formats = format if isinstance(format, list) else [format]
        
        if 'txt' in formats or format == 'all':
            txt_path = os.path.join(output_dir, f'extracted_text_{timestamp}.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(results['raw_text'])
                f.write('\n\nCategorized Information:\n')
                for category, items in results['categorized'].items():
                    if items:
                        f.write(f'\n{category.upper()}:\n')
                        for item in items:
                            f.write(f'- {item}\n')
            exported_files['txt'] = txt_path

        if 'csv' in formats or format == 'all':
            csv_path = os.path.join(output_dir, f'extracted_data_{timestamp}.csv')
            df = pd.DataFrame(results['categorized'])
            df.to_csv(csv_path, index=False)
            exported_files['csv'] = csv_path

        if 'json' in formats or format == 'all':
            json_path = os.path.join(output_dir, f'extracted_data_{timestamp}.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            exported_files['json'] = json_path

        if 'pdf' in formats or format == 'all':
            pdf_path = os.path.join(output_dir, f'report_{timestamp}.pdf')
            self.create_pdf_report(results, output_path=pdf_path)
            exported_files['pdf'] = pdf_path

        return exported_files

    def process_document(self, image, lang='eng'):
        """Process document image and return extracted and categorized text with regions"""
        # Check if the image might contain handwritten text
        is_handwritten = self._check_if_handwritten(image)
        
        if is_handwritten:
            # Process as handwritten text
            if isinstance(image, str):
                image = cv2.imread(image)
            results = self.advanced_nlp.process_handwritten_text(image)
            text = results['text']
            confidence = results['confidence']
            regions = [{'x': 0, 'y': 0, 'w': image.shape[1], 'h': image.shape[0], 
                       'text': text, 'conf': confidence * 100}]
        else:
            # Process as printed text
            text, regions = self.extract_text_with_regions(image)
        
        # Apply advanced NLP processing
        nlp_results = self.advanced_nlp.process_text(text)
        
        # Combine results
        return {
            'raw_text': text,
            'categorized': self.categorize_text(text),
            'regions': regions,
            'nlp_analysis': {
                'linked_entities': nlp_results['linked_entities'],
                'relations': nlp_results['relations'],
                'custom_entities': nlp_results['custom_entities']
            },
            'is_handwritten': is_handwritten
        }

    def _check_if_handwritten(self, image) -> bool:
        """
        Determine if the image contains handwritten text using image analysis
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Calculate line properties
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # Analyze line characteristics
            straight_lines = 0
            curved_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Check line straightness
                if abs(x2-x1) > abs(y2-y1):
                    deviation = abs(y2-y1) / length
                else:
                    deviation = abs(x2-x1) / length
                
                if deviation < 0.1:  # Threshold for straight lines
                    straight_lines += 1
                else:
                    curved_lines += 1
            
            # If there are more curved lines than straight lines, likely handwritten
            return curved_lines > straight_lines
        
        return False

    def extract_numbers(self, text):
        """
        Extract all numbers from the text.
        Returns a list of numbers (both integers and decimals).
        """
        # Pattern matches both integers and decimal numbers
        number_pattern = r'-?\d*\.?\d+'
        numbers = re.findall(number_pattern, text)
        # Convert strings to float/int where possible
        return [float(num) if '.' in num else int(num) for num in numbers]
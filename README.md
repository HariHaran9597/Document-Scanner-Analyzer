# Document Scanner & Analyzer

An advanced document processing application that combines OCR, image processing, and advanced NLP capabilities to extract and analyze text from both printed and handwritten documents.

## Features

- **Document Detection & Processing**
  - Automatic document boundary detection
  - Perspective correction and image enhancement
  - Support for both printed and handwritten text
  - Adaptive image preprocessing with configurable parameters

- **Advanced Text Recognition**
  - OCR for printed documents using Tesseract
  - Handwriting recognition using TensorFlow Hub models
  - Confidence scoring for extracted text regions
  - Multi-language support

- **Natural Language Processing**
  - Named Entity Recognition (NER) using spaCy and Transformers
  - Entity linking with similarity analysis
  - Relationship extraction between entities
  - Custom pattern matching for domain-specific entities
  - Support for medical terms, legal clauses, and invoice references

- **Information Extraction**
  - Automatic categorization of extracted information
  - Detection of dates, amounts, emails, phone numbers
  - Recognition of invoice numbers, product codes, tracking numbers
  - Identification of personal and business information

- **Export Capabilities**
  - Multiple export formats (TXT, CSV, JSON, PDF)
  - Detailed PDF reports with NLP insights
  - Organized data categorization
  - Visual annotations of detected regions

## Requirements

- Python 3.11 or higher
- Dependencies listed in requirements.txt:
  - opencv-python
  - pytesseract
  - Pillow
  - streamlit
  - numpy
  - spacy==3.5.3
  - python-dotenv==1.0.0
  - pandas
  - fpdf==1.7.2
  - scikit-image==0.21.0
  - transformers==4.30.2
  - torch
  - tensorflow==2.12.0

## Installation

1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Install spaCy language models:
```bash
python -m spacy download en_core_web_lg
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Use the web interface to:
   - Upload documents for processing
   - Adjust processing parameters
   - View extracted text and analysis
   - Export results in desired formats

## Project Structure

- `app.py` - Main Streamlit application interface
- `image_processor.py` - Image preprocessing and document detection
- `ocr_processor.py` - Text extraction and OCR processing
- `advanced_nlp.py` - NLP analysis and entity extraction
- `config.py` - Configuration management
- `install_dependencies.py` - Dependency installation helper

## Configuration

The application can be configured through:
- Web interface controls
- `config.py` default settings
- External configuration file (scanner_config.json)

## Advanced Features

### Custom Entity Recognition
The system supports custom pattern matching for domain-specific entities:
- Medical terms (e.g., conditions ending in "itis" or "oma")
- Legal clauses
- Invoice references

### Document Analysis
- Automatic detection of handwritten vs printed text
- Confidence scoring for extracted text
- Entity relationship mapping
- Similar concept detection

## Export Formats

1. **Text (TXT)**
   - Raw extracted text
   - Categorized information

2. **CSV**
   - Structured data export
   - Categorized entities

3. **JSON**
   - Complete analysis results
   - NLP insights
   - Entity relationships

4. **PDF Report**
   - Document image
   - Extracted text
   - NLP analysis
   - Entity relationships
   - Categorized information

## Screenshots

![image](https://github.com/user-attachments/assets/850bae48-27cd-4732-b498-9b6101b5b682)

![image](https://github.com/user-attachments/assets/26eb2c82-43dc-4948-a9a7-2197f954ca7d)

![image](https://github.com/user-attachments/assets/2e37a521-1c4a-4c97-82f6-3ffbdb7e3688)

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

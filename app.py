import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from image_processor import ImageProcessor
from ocr_processor import OCRProcessor
from config import Config

def initialize_session_state():
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    if 'processor_params' not in st.session_state:
        st.session_state.processor_params = st.session_state.config.settings['image_processing']
    if 'ocr_params' not in st.session_state:
        st.session_state.ocr_params = st.session_state.config.settings['ocr']
    if 'export_params' not in st.session_state:
        st.session_state.export_params = st.session_state.config.settings['export']

def save_settings():
    config = st.session_state.config
    config.update_settings('image_processing', **st.session_state.processor_params)
    config.update_settings('ocr', **st.session_state.ocr_params)
    config.update_settings('export', **st.session_state.export_params)
    st.success("Settings saved successfully!")

def settings_sidebar():
    st.sidebar.header("Settings")
    
    # Image Processing Settings
    st.sidebar.subheader("Image Processing")
    processor_params = st.session_state.processor_params
    
    # Updated blur kernel controls
    col1, col2 = st.sidebar.columns(2)
    with col1:
        processor_params['blur_kernel_width'] = st.number_input(
            "Blur Width",
            1, 31, processor_params['blur_kernel_width'], step=2,
            help="Must be an odd number"
        )
    with col2:
        processor_params['blur_kernel_height'] = st.number_input(
            "Blur Height",
            1, 31, processor_params['blur_kernel_height'], step=2,
            help="Must be an odd number"
        )
    
    processor_params['threshold_block_size'] = st.sidebar.slider(
        "Threshold Block Size",
        3, 21, processor_params['threshold_block_size'], step=2
    )
    processor_params['threshold_constant'] = st.sidebar.slider(
        "Threshold Constant",
        0, 10, processor_params['threshold_constant']
    )

    # OCR Settings
    st.sidebar.subheader("OCR Settings")
    ocr_params = st.session_state.ocr_params
    
    ocr_params['language'] = st.sidebar.selectbox(
        "OCR Language",
        ['eng', 'fra', 'deu', 'spa'],
        index=['eng', 'fra', 'deu', 'spa'].index(ocr_params['language']),
        format_func=lambda x: {
            'eng': 'English',
            'fra': 'French',
            'deu': 'German',
            'spa': 'Spanish'
        }[x]
    )

    ocr_params['min_confidence'] = st.sidebar.slider(
        "OCR Minimum Confidence",
        0, 100, ocr_params['min_confidence']
    )

    # Export Settings
    st.sidebar.subheader("Export Settings")
    export_params = st.session_state.export_params
    
    export_params['default_formats'] = st.sidebar.multiselect(
        "Default Export Formats",
        ['txt', 'csv', 'json', 'pdf'],
        default=export_params['default_formats']
    )

    if st.sidebar.button("Save Settings"):
        save_settings()

def save_uploaded_file(uploadedfile, directory='temp'):
    """Save uploaded file temporarily"""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def display_document_analysis(original_image, processed_results, ocr_results, filename, ocr_processor):
    """Display document analysis with text regions highlighted"""
    col1, col2 = st.columns(2)
    
    # Original image with document bounds
    with col1:
        st.subheader("Original Document")
        if 'annotated' in processed_results:
            st.image(processed_results['annotated'], channels="BGR")
        else:
            st.image(original_image, channels="BGR")

    # Processed image with text regions
    with col2:
        st.subheader("Processed Document")
        processed_image = processed_results['processed']
        
        # Draw text regions if available
        if 'regions' in ocr_results:
            highlighted = ImageProcessor().draw_text_regions(
                processed_image,
                ocr_results['regions']
            )
            st.image(highlighted, channels="BGR")
        else:
            st.image(processed_image, channels="BGR")

    # OCR Results
    st.subheader("Extracted Text")
    
    # Show confidence scores for each text region
    if 'regions' in ocr_results:
        regions_df = []
        for region in ocr_results['regions']:
            if region['text'].strip():  # Only show non-empty text
                regions_df.append({
                    'Text': region['text'],
                    'Confidence': f"{region['conf']:.1f}%",
                    'Position': f"({region['x']}, {region['y']})"
                })
        if regions_df:
            st.dataframe(regions_df)
    
    st.text_area("Raw Text", ocr_results['raw_text'], height=150)

    # Categorized Information
    st.subheader("Categorized Information")
    categorized = ocr_results['categorized']
    
    # Create three columns for better organization
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    
    # Group categories by type
    category_groups = {
        'Personal Information': ['names', 'emails', 'phones', 'social_security'],
        'Business Information': ['organizations', 'job_titles', 'locations'],
        'Financial Information': ['amounts', 'credit_cards', 'bank_accounts'],
        'Document Information': ['dates', 'invoice_numbers', 'product_codes', 'tracking_numbers'],
        'Other': ['urls', 'postal_codes']
    }
    
    # Display categories by group
    for i, (group_name, categories) in enumerate(category_groups.items()):
        col = cols[i % 3]
        with col:
            st.markdown(f"**{group_name}**")
            for category in categories:
                if category in categorized and categorized[category]:
                    st.write(f"📌 {category.title()}:")
                    for item in categorized[category]:
                        st.write(f"- {item}")
            st.markdown("---")

    # Export options
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        if st.button(f"Export {filename}"):
            export_formats = st.session_state.export_params['default_formats']
            exported = ocr_processor.export_results(
                ocr_results,
                format=export_formats
            )
            for fmt, path in exported.items():
                if path:
                    st.success(f"Exported to {path}")

def main():
    st.title("Document Scanner & Analyzer")
    
    # Initialize processors
    config = Config()
    image_processor = ImageProcessor()
    ocr_processor = OCRProcessor()
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Document type settings
    doc_type = st.sidebar.radio(
        "Document Type",
        ["Auto Detect", "Printed", "Handwritten"]
    )
    
    # Advanced NLP settings
    st.sidebar.subheader("Advanced NLP Settings")
    enable_entity_linking = st.sidebar.checkbox("Enable Entity Linking", value=True)
    enable_relation_extraction = st.sidebar.checkbox("Enable Relation Extraction", value=True)
    
    # Image processing settings
    st.sidebar.subheader("Image Processing")
    blur_kernel = st.sidebar.slider("Blur Kernel Size", 1, 11, 3, step=2)
    threshold_block_size = st.sidebar.slider("Threshold Block Size", 3, 99, 15, step=2)
    threshold_constant = st.sidebar.slider("Threshold Constant", 0, 50, 8)
    
    # Export settings
    st.sidebar.subheader("Export Settings")
    export_formats = st.sidebar.multiselect(
        "Export Formats",
        ["txt", "csv", "json", "pdf"],
        default=["txt", "pdf"]
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Document", use_column_width=True)
        
        # Process image
        enhanced_image = image_processor.process_image(
            np.array(image),
            blur_kernel_size=(blur_kernel, blur_kernel),
            threshold_block_size=threshold_block_size,
            threshold_constant=threshold_constant
        )
        
        # Display enhanced image
        st.image(enhanced_image, caption="Enhanced Document", use_column_width=True)
        
        # Process button
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Convert enhanced image back to PIL
                enhanced_pil = Image.fromarray(enhanced_image)
                
                # Force handwritten processing if selected
                if doc_type == "Handwritten":
                    results = ocr_processor.process_document(enhanced_pil)
                    results['is_handwritten'] = True
                else:
                    results = ocr_processor.process_document(enhanced_pil)
                    
                    # Override auto-detection if printed is selected
                    if doc_type == "Printed":
                        results['is_handwritten'] = False
                
                # Display document type
                st.subheader("Document Analysis")
                st.write(f"Detected Document Type: {'Handwritten' if results['is_handwritten'] else 'Printed'}")
                
                # Display extracted text
                st.subheader("Extracted Text")
                st.text_area("", results['raw_text'], height=150)
                
                # Display NLP analysis if enabled
                if enable_entity_linking or enable_relation_extraction:
                    st.subheader("NLP Analysis")
                    
                    if enable_entity_linking and results['nlp_analysis']['linked_entities']:
                        st.write("Linked Entities:")
                        for entity in results['nlp_analysis']['linked_entities']:
                            with st.expander(f"{entity['text']} ({entity['label']})"):
                                if entity['similar_concepts']:
                                    st.write("Similar concepts:")
                                    for concept, score in entity['similar_concepts']:
                                        st.write(f"- {concept} (similarity: {score:.2f})")
                    
                    if enable_relation_extraction and results['nlp_analysis']['relations']:
                        st.write("Entity Relations:")
                        for relation in results['nlp_analysis']['relations']:
                            st.write(f"- {relation['entity1']['text']} → {relation['relation']} → {relation['entity2']['text']}")
                
                # Display categorized information
                st.subheader("Categorized Information")
                for category, items in results['categorized'].items():
                    if items:
                        with st.expander(category.title()):
                            for item in items:
                                st.write(f"- {item}")
                
                # Export results
                if export_formats:
                    st.subheader("Export Results")
                    exported_files = ocr_processor.export_results(
                        results,
                        format=export_formats
                    )
                    
                    for format, filepath in exported_files.items():
                        with open(filepath, 'rb') as f:
                            st.download_button(
                                f"Download {format.upper()} Report",
                                f,
                                file_name=os.path.basename(filepath),
                                mime=f"application/{format}"
                            )

if __name__ == "__main__":
    main()
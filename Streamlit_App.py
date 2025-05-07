import streamlit as st
import os
import json
import pandas as pd
import tempfile
import anthropic  # Import Anthropic instead of OpenAI
import docx
from PyPDF2 import PdfReader
import io
import zipfile

# Set page config and title
st.set_page_config(
    page_title="Translation Assistant",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# App title and description
st.title("Translation Assistant")
st.write("Upload documents to generate translation resources")

# Function to load Anthropic API key
def get_anthropic_client():
    try:
        # Try to get API key from secrets
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        
        # If not found in secrets, try environment variable
        if not api_key:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        # If not found in environment, try config file
        if not api_key:
            config_path = 'config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    api_key = config.get('ANTHROPIC_API_KEY')
        
        # If API key is found, create client
        if api_key:
            # Create Anthropic client
            return anthropic.Anthropic(api_key=api_key)
        else:
            st.error('API key not configured. Please set the ANTHROPIC_API_KEY in config.json, environment, or Streamlit secrets.')
            return None
    except Exception as e:
        st.error(f'Error initializing Anthropic client: {e}')
        return None

# Function to extract text from files
def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file based on file type"""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'txt':
            # For text files
            return uploaded_file.getvalue().decode('utf-8')
        elif file_ext == 'docx':
            # For Word documents
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            return '\n'.join([para.text for para in doc.paragraphs])
        elif file_ext == 'pdf':
            # For PDF files
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            st.error(f"Unsupported file type: {file_ext}")
            return None
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {e}")
        return None

# Function to save analysis as Word document
def save_analysis_as_word(content):
    """Save analysis content as Word document"""
    try:
        # Create Word document
        doc = docx.Document()
        
        # Add title
        doc.add_heading('Translation Analysis and Persona', 0)
        
        # Split content by sections (headers starting with #)
        if '## ' in content:
            sections = content.split('\n## ')
        else:
            # Fallback if no ## headers
            sections = [content]
        
        # Process first section (if not starting with ##)
        if not content.strip().startswith('## '):
            first_part = sections[0]
            doc.add_paragraph(first_part)
            sections = sections[1:]
            
        # Process remaining sections
        for section in sections:
            if not section.strip():
                continue
                
            # Split into title and content
            lines = section.split('\n', 1)
            if len(lines) > 0:
                title = lines[0]
                doc.add_heading(title, level=2)
                
                if len(lines) > 1:
                    content = lines[1]
                    # Process content, handling potential markdown formatting
                    paragraphs = content.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            doc.add_paragraph(para)
        
        # Save to BytesIO object
        docx_io = io.BytesIO()
        doc.save(docx_io)
        docx_io.seek(0)
        return docx_io
    except Exception as e:
        st.error(f"Error creating Word document: {e}")
        return None

# Function to extract glossary terms
def extract_glossary_terms(text, source_language, target_language):
    """Extract glossary terms from the API response"""
    try:
        # Look for the glossary table
        lines = text.split('\n')
        table_lines = []
        
        # Find any lines that look like a table row
        for line in lines:
            if line.strip().startswith('|') and line.strip().endswith('|'):
                table_lines.append(line)
        
        if not table_lines:
            st.warning("No glossary table found in the response. Please check the analysis document.")
            return []
        
        # Process the table
        terms = []
        header_found = False
        
        for line in table_lines:
            # Skip separator rows
            if '-+-' in line.replace(' ', '') or '-|-' in line.replace(' ', ''):
                continue
                
            # Skip empty lines
            if line.strip() == '' or line.strip() == '|':
                continue
                
            # Split by pipe and remove leading/trailing whitespace
            columns = [col.strip() for col in line.split('|')]
            # Remove empty entries from start and end
            columns = [col for col in columns if col]
            
            # Skip if this looks like a header row
            if not header_found and (
                'term' in ' '.join(columns).lower() or 
                source_language.lower() in ' '.join(columns).lower() or
                'translation' in ' '.join(columns).lower()
            ):
                header_found = True
                continue
            
            if len(columns) >= 2:
                term = {
                    'source_term': columns[0],
                    'target_term': columns[1],
                    'english_reference': columns[2] if len(columns) > 2 else '',
                    'example': columns[3] if len(columns) > 3 else ''
                }
                terms.append(term)
        
        if terms:
            st.success(f"Successfully extracted {len(terms)} glossary terms!")
        else:
            st.warning("Glossary table found but no terms could be extracted.")
            
        return terms
    except Exception as e:
        st.error(f"Error extracting glossary terms: {e}")
        return []

# Function to create Excel file from glossary terms
def create_glossary_excel(terms):
    """Create Excel file from extracted glossary terms"""
    try:
        if not terms:
            # Create a minimal structure if no terms were found
            terms = [{'source_term': 'No terms extracted', 
                     'target_term': 'Please check the analysis document', 
                     'english_reference': '', 
                     'example': ''}]
        
        # Create DataFrame with explicit column names
        df = pd.DataFrame(terms)
        
        # Rename columns to match expected format
        column_mapping = {
            'source_term': 'Source Term',
            'target_term': 'Target Translation', 
            'english_reference': 'English Reference',
            'example': 'Example Sentence'
        }
        
        # If columns exist in the DataFrame, rename them
        existing_cols = [col for col in df.columns if col in column_mapping]
        df = df.rename(columns={col: column_mapping[col] for col in existing_cols})
        
        # Save to BytesIO object
        excel_io = io.BytesIO()
        df.to_excel(excel_io, index=False, engine='openpyxl')
        excel_io.seek(0)
        return excel_io
    except Exception as e:
        st.error(f"Error creating Excel file: {e}")
        return None

# Create form for user input
with st.form("translation_form"):
    # Language Selection
    col1, col2 = st.columns(2)
    
    with col1:
        source_language = st.selectbox(
            "Source Language:",
            options=[
                "English", "Spanish", "French", "German", "Italian", 
                "Bulgarian", "Romanian", "Turkish", "Chinese", "Japanese", 
                "Russian", "Arabic", "Portuguese", "Dutch", "Greek", 
                "Polish", "Swedish", "Czech", "Hungarian"
            ],
            key="source_language"
        )
    
    with col2:
        target_language = st.selectbox(
            "Target Language:",
            options=[
                "English", "Spanish", "French", "German", "Italian", 
                "Bulgarian", "Romanian", "Turkish", "Chinese", "Japanese", 
                "Russian", "Arabic", "Portuguese", "Dutch", "Greek", 
                "Polish", "Swedish", "Czech", "Hungarian"
            ],
            key="target_language"
        )
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Documents (TXT, DOCX, PDF):",
        type=["txt", "docx", "pdf"],
        accept_multiple_files=True
    )
    
    # Model selection
    model = st.selectbox(
        "AI Model:",
        options=["claude-3-5-sonnet", "claude-3-opus", "claude-3-haiku"],
        index=0,  # Default to claude-3-5-sonnet
        help="Select the Claude model to use. Opus is more powerful but slower, Haiku is faster but less capable."
    )
    
    # Submit button
    submit_button = st.form_submit_button("Generate Translation Resources")

# Process files when form is submitted
if submit_button and uploaded_files:
    # Check if API key is configured
    client = get_anthropic_client()
    if not client:
        st.error("Anthropic API key is not configured. Please check your configuration.")
        st.stop()
    
    if not source_language or not target_language:
        st.error("Please select both source and target languages.")
        st.stop()
    
    if source_language == target_language:
        st.error("Source and target languages must be different.")
        st.stop()
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Processing files...")
    
    # Extract text from uploaded files
    file_contents = []
    for i, file in enumerate(uploaded_files):
        progress = (i / len(uploaded_files)) * 0.3  # 30% of progress for file extraction
        progress_bar.progress(progress)
        status_text.text(f"Extracting text from {file.name}...")
        
        content = extract_text_from_file(file)
        if content:
            file_contents.append(content)
    
    if not file_contents:
        st.error("No valid content could be extracted from the uploaded files.")
        st.stop()
    
    # Update progress
    progress_bar.progress(0.3)
    status_text.text("Analyzing content with Claude...")
    
    # Combine file contents
    combined_content = "\n\n".join(file_contents)
    
    # Create analysis prompt for Claude
    analysis_prompt = f"""
    I need you to analyze these {source_language} documents for translation into {target_language}. Please:
    
    1. Analyze the content and subject matter of these documents 
    2. Create a detailed translator persona specializing in {source_language} to {target_language} translation for this specific content domain 
    3. Create a comprehensive glossary of terms with example sentences from the documents 
    4. Format the glossary with {source_language} terms, {target_language} translations, English reference translations, and example sentences from the original documents
    
    Please don't translate the documents themselves - I just need the analysis and glossary to assist a human translator.
    
    Here are the document contents:
    
    {combined_content}
    """
    
    # Call Anthropic API
    try:
        status_text.text("Sending request to Claude...")
        progress_bar.progress(0.4)
        
        # Make API request to Claude
        message = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0,
            system="You are an expert translator and linguistics specialist.",
            messages=[
                {"role": "user", "content": analysis_prompt}
            ]
        )
        
        # Get response content
        analysis_result = message.content[0].text
        
        # Update progress
        progress_bar.progress(0.7)
        status_text.text("Processing results...")
        
        # Extract glossary terms
        glossary_terms = extract_glossary_terms(analysis_result, source_language, target_language)
        
        # Create Excel file
        progress_bar.progress(0.8)
        excel_data = create_glossary_excel(glossary_terms)
        
        # Create Word document with full analysis
        progress_bar.progress(0.9)
        docx_data = save_analysis_as_word(analysis_result)
        
        # Update progress
        progress_bar.progress(1.0)
        status_text.text("Complete!")
        
        # Show success message
        st.success(f"Successfully generated translation resources for {source_language} to {target_language}!")
        
        # Create a combined ZIP file with both documents
        if excel_data is not None and docx_data is not None:
            # Create a ZIP file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                # Add Excel file to ZIP
                zip_file.writestr(f"glossary_{source_language}_to_{target_language}.xlsx", excel_data.getvalue())
                
                # Add Word file to ZIP
                zip_file.writestr(f"analysis_{source_language}_to_{target_language}.docx", docx_data.getvalue())
            
            # Reset buffer position
            zip_buffer.seek(0)
            
            # Create download button for ZIP file
            st.download_button(
                label="📦 Download All Translation Resources (ZIP)",
                data=zip_buffer,
                file_name=f"translation_resources_{source_language}_to_{target_language}.zip",
                mime="application/zip"
            )

        # Also show individual download buttons
        st.markdown("### Individual Files")
        st.markdown("If you prefer to download files separately, use these buttons:")
        col1, col2 = st.columns(2)
        
        with col1:
            if excel_data is not None:
                st.download_button(
                    label="📊 Download Glossary Only",
                    data=excel_data,
                    file_name=f"glossary_{source_language}_to_{target_language}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_glossary"
                )
                
        with col2:
            if docx_data is not None:
                st.download_button(
                    label="📝 Download Analysis Only",
                    data=docx_data,
                    file_name=f"analysis_{source_language}_to_{target_language}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="download_analysis"
                )
            
        # Show preview of analysis
        with st.expander("Preview Analysis"):
            st.markdown(analysis_result)
        
    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        progress_bar.empty()
        status_text.empty()

# Show information when no files are uploaded
elif submit_button and not uploaded_files:
    st.error("Please upload at least one file.")

# Add footer
st.markdown("---")
st.caption("This application uses Claude AI to analyze documents and create translation resources.")
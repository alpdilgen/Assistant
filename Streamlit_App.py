import streamlit as st
import os
import json
import pandas as pd
import tempfile
import anthropic
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
            # Create Anthropic client without any extra parameters
            return anthropic.Anthropic(api_key=api_key)
        else:
            st.error('API key not configured. Please set the ANTHROPIC_API_KEY in config.json, environment, or Streamlit secrets.')
            return None
    except Exception as e:
        st.error(f'Error initializing Anthropic client: {e}')
        return None

# Function to extract text from files with improved encoding handling
def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file based on file type with improved encoding handling"""
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_ext == 'txt':
            # For text files - try various encodings
            content = None
            encodings = ['utf-8', 'cp1251', 'iso-8859-5', 'windows-1251']  # Common encodings for Cyrillic
            
            for encoding in encodings:
                try:
                    content = uploaded_file.getvalue().decode(encoding)
                    # If we get here, decoding worked
                    st.success(f"Successfully decoded text with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                st.error(f"Could not decode text file with any of the attempted encodings")
                return None
                
            return content
            
        elif file_ext == 'docx':
            # For Word documents - extract text with more validation
            doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
            
            # Extract text from paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():  # Only include non-empty paragraphs
                    paragraphs.append(para.text)
            
            # Also check for tables which might contain text
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            paragraphs.append(cell.text)
            
            content = '\n'.join(paragraphs)
            
            # Validation and debugging
            if not content:
                st.warning(f"The DOCX file appears to be empty or contains no extractable text")
            else:
                st.success(f"Successfully extracted {len(content)} characters from DOCX file")
                
            return content
            
        elif file_ext == 'pdf':
            # For PDF files - with enhanced extraction
            reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            
            if len(reader.pages) == 0:
                st.warning("The PDF has no pages")
                return None
                
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"--- Page {i+1} ---\n{page_text}\n\n"
            
            if not text:
                st.warning(f"The PDF file appears to be empty or the text couldn't be extracted")
            else:
                st.success(f"Successfully extracted {len(text)} characters from {len(reader.pages)} PDF pages")
                
            return text
            
        else:
            st.error(f"Unsupported file type: {file_ext}")
            return None
            
    except Exception as e:
        st.error(f"Error extracting text from {uploaded_file.name}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
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

# First extraction method - direct from Claude's response
def extract_glossary_terms(text, source_language, target_language):
    """Extract glossary terms from the API response with improved table detection"""
    try:
        # First approach: Use the full text from Claude's response to create the glossary
        # This is the most direct approach since Claude already formats glossaries well
        
        # Get the model's response directly
        analysis_lines = text.split('\n')
        glossary_section = []
        in_glossary = False
        
        # Find the glossary section
        for i, line in enumerate(analysis_lines):
            if ('glossary' in line.lower() or 'key terms' in line.lower() or 
                'terms' in line.lower() and ('table' in line.lower() or '|' in line)):
                in_glossary = True
                
            # Once we've found the glossary section, collect all lines
            if in_glossary:
                glossary_section.append(line)
        
        # If we found a glossary section, extract terms
        if glossary_section:
            # Process all lines after the glossary header
            terms = []
            table_started = False
            headers_found = False
            
            for line in glossary_section:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # If we find a line with pipes, we've started the table
                if '|' in line:
                    table_started = True
                    
                    # Skip header and separator rows
                    if (not headers_found and 
                        (source_language in line or target_language in line or 
                        'english' in line.lower() or 'term' in line.lower() or
                        'source' in line.lower() or 'target' in line.lower())):
                        headers_found = True
                        continue
                    
                    # Skip separator rows (contain only dashes and pipes)
                    if set(line.replace('|', '').replace('-', '').replace(' ', '')) == set():
                        continue
                    
                    # Extract columns from the line
                    columns = [col.strip() for col in line.split('|')]
                    columns = [col for col in columns if col]  # Remove empty entries
                    
                    if len(columns) >= 2:
                        term = {
                            'source_term': columns[0],
                            'target_term': columns[1],
                            'english_reference': columns[2] if len(columns) > 2 else '',
                            'example': columns[3] if len(columns) > 3 else ''
                        }
                        
                        # Only add if we have non-empty source and target terms
                        if term['source_term'] and term['target_term']:
                            terms.append(term)
        
            # If we couldn't extract terms from the table, try a different approach
            if not terms:
                st.warning("Could not extract terms from table format. Trying direct extraction...")
                # Create a new prompt specifically for glossary extraction
                return extract_terms_with_second_prompt(text, source_language, target_language)
                
            return terms
        else:
            # If no glossary section found, try the secondary approach
            st.warning("No glossary section found. Using secondary extraction method...")
            return extract_terms_with_second_prompt(text, source_language, target_language)
            
    except Exception as e:
        st.error(f"Error in primary glossary extraction: {e}")
        st.warning("Using secondary extraction method due to error...")
        return extract_terms_with_second_prompt(text, source_language, target_language)

# Second extraction method - dedicated glossary request
def extract_terms_with_second_prompt(text, source_language, target_language):
    """Extract terms using a secondary prompt to Claude"""
    try:
        client = get_anthropic_client()
        if not client:
            st.error("Could not initialize Anthropic client for secondary extraction")
            return []
            
        # Create a targeted prompt to extract the glossary
        extraction_prompt = f"""
        I need you to extract a comprehensive glossary from this translation analysis. 
        
        The source language is {source_language} and the target language is {target_language}.
        
        Please format your response ONLY as a table with these exact columns:
        | Source Term | Target Translation | English Reference | Example |
        
        Include as many terms as possible from the document (at least a minimum of 50 terms if available).
        Make sure to preserve all terminology from the original analysis.
        DO NOT summarize or reduce the number of terms.
        
        Here is the content:
        
        {text}
        """
        
        # Make API request to Claude specifically for glossary extraction
        st.info("Making secondary API call to extract comprehensive glossary...")
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",  # Use a consistent model for extraction
            max_tokens=4000,
            temperature=0,
            system="You are a specialized translation terminology extractor. Extract ALL terms from the document without summarizing.",
            messages=[
                {"role": "user", "content": extraction_prompt}
            ]
        )
        
        # Parse the table from the response
        extraction_result = message.content[0].text
        
        # Process the response to extract terms
        terms = []
        lines = extraction_result.split('\n')
        table_started = False
        
        for line in lines:
            # Skip empty lines and lines without pipes
            if not line.strip() or '|' not in line:
                continue
                
            # Skip header and separator rows
            if 'Source Term' in line or 'Target Translation' in line:
                table_started = True
                continue
                
            if table_started and set(line.replace('|', '').replace('-', '').replace(' ', '')) == set():
                continue
                
            if table_started:
                # Extract columns from the line
                columns = [col.strip() for col in line.split('|')]
                columns = [col for col in columns if col]  # Remove empty entries
                
                if len(columns) >= 2:
                    term = {
                        'source_term': columns[0],
                        'target_term': columns[1],
                        'english_reference': columns[2] if len(columns) > 2 else '',
                        'example': columns[3] if len(columns) > 3 else ''
                    }
                    
                    # Only add if we have non-empty source and target terms
                    if term['source_term'] and term['target_term']:
                        terms.append(term)
        
        if terms:
            st.success(f"Successfully extracted {len(terms)} glossary terms with secondary extraction!")
            return terms
        else:
            st.warning("Secondary extraction also failed. Trying last resort method...")
            return extract_terms_last_resort(text, source_language, target_language)
            
    except Exception as e:
        st.error(f"Error in secondary glossary extraction: {e}")
        st.warning("Trying last resort extraction method...")
        return extract_terms_last_resort(text, source_language, target_language)

# Last resort extraction method for when all else fails
def extract_terms_last_resort(text, source_language, target_language):
    """Last resort method for term extraction when other methods fail"""
    try:
        client = get_anthropic_client()
        if not client:
            st.error("Could not initialize Anthropic client for last resort extraction")
            return [{"source_term": "Extraction Failed", 
                     "target_term": "Please check the analysis document for terminology", 
                     "english_reference": "", 
                     "example": ""}]
        
        # Create a completely different prompt focusing only on term extraction
        content_extract = text[:10000]  # Limit to first 10000 chars to avoid token limits
        
        last_resort_prompt = f"""
        Create a comprehensive glossary for translation from {source_language} to {target_language}.
        
        The glossary should contain AT LEAST 50 terms (or more if possible) and be formatted as follows:
        
        | {source_language} Term | {target_language} Translation | English Reference | Example |
        |---|---|---|---|
        | term1 | translation1 | english1 | example1 |
        
        Include domain-specific terminology for the subject matter described.
        DO NOT respond with anything except the glossary table.
        
        Content description: {content_extract}
        """
        
        st.info("Making final attempt to extract glossary...")
        message = client.messages.create(
            model="claude-3-opus-20240229",  # Use the most capable model for this extraction
            max_tokens=4000,
            temperature=0.2,  # Slight increase in temperature to encourage creativity
            system="You are a terminology expert. Your task is to create a comprehensive glossary with at least 50 terms.",
            messages=[
                {"role": "user", "content": last_resort_prompt}
            ]
        )
        
        extraction_result = message.content[0].text
        
        # Process the response
        terms = []
        lines = extraction_result.split('\n')
        table_started = False
        
        for line in lines:
            # Skip empty lines and lines without pipes
            if not line.strip() or '|' not in line:
                continue
                
            # Skip header and separator rows
            if line.lower().startswith('| source') or line.lower().startswith('|source') or '|---' in line:
                table_started = True
                continue
                
            if table_started:
                # Extract columns from the line
                columns = [col.strip() for col in line.split('|')]
                columns = [col for col in columns if col]  # Remove empty entries
                
                if len(columns) >= 2:
                    term = {
                        'source_term': columns[0],
                        'target_term': columns[1],
                        'english_reference': columns[2] if len(columns) > 2 else '',
                        'example': columns[3] if len(columns) > 3 else ''
                    }
                    
                    # Only add if we have non-empty source and target terms
                    if term['source_term'] and term['target_term']:
                        terms.append(term)
        
        if terms:
            st.success(f"Final attempt extracted {len(terms)} glossary terms!")
            return terms
        else:
            # As an absolute fallback, return some basic terms
            st.warning("All extraction methods failed. Creating minimal glossary.")
            return [{"source_term": "Extraction Failed", 
                     "target_term": "Please check the analysis document for terminology", 
                     "english_reference": "", 
                     "example": "The full analysis contains the glossary that could not be automatically extracted."}]
    
    except Exception as e:
        st.error(f"Error in last resort glossary extraction: {e}")
        return [{"source_term": "Extraction Error", 
                 "target_term": f"Error: {str(e)}", 
                 "english_reference": "", 
                 "example": ""}]

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
        options=["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        index=0,  # Default to claude-3-5-sonnet
        help="Select the Claude model to use. Opus is more powerful but slower, Haiku is faster but less capable."
    )
    
    # Glossary extraction intensity
    extraction_method = st.radio(
        "Glossary Extraction Mode:",
        options=["Standard", "Comprehensive"],
        index=1,  # Default to Comprehensive
        help="Standard uses a single API call. Comprehensive makes multiple API calls to extract more terms."
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
    
    # Debug file contents
    if file_contents:
        with st.expander("Debug: View Extracted File Contents"):
            for i, content in enumerate(file_contents):
                st.text(f"File {i+1} content preview (first 500 chars):")
                preview = content[:500] + "..." if len(content) > 500 else content
                st.text(preview)
                st.text(f"Character count: {len(content)}")
    else:
        st.error("No content was extracted from the uploaded files.")
        st.stop()
    
    # Update progress
    progress_bar.progress(0.3)
    status_text.text("Analyzing content with Claude...")
    
    # Ensure there is actual content to process
    combined_content = "\n\n".join(file_contents)
    if not combined_content.strip():
        st.error("The extracted file content appears to be empty. Please check that your files contain text that can be properly extracted.")
        st.stop()

    st.info(f"Total character count of extracted content: {len(combined_content)}")
    
    # Add a small example if the content is too short (for testing)
    if len(combined_content) < 10:
        st.warning("The extracted content is very short. Adding example text for testing.")
        if source_language == "Bulgarian":
            combined_content += """
            Пример текст на български:
            Здравейте! Това е пример за текст на български. Тук може да добавите вашия собствен текст.
            Използвайте реален текст, за да получите по-добри резултати от анализа.
            """
    
    # Create analysis prompt for Claude - improved for better glossary extraction
    analysis_prompt = f"""
    I need you to analyze these {source_language} documents for translation into {target_language}. Please:
    
    1. Analyze the content and subject matter of these documents 
    2. Create a detailed translator persona specializing in {source_language} to {target_language} translation for this specific content domain 
    3. Create a COMPREHENSIVE glossary of terms with example sentences from the documents - this is the most important part
    4. Format the glossary with {source_language} terms, {target_language} translations, English reference translations, and example sentences from the original documents
    
    For the glossary:
    - Include ALL domain-specific terms (at least 100+ terms if available)
    - Format as a table with | {source_language} Term | {target_language} Translation | English Reference | Example |
    - Do not summarize or reduce the number of terms
    - Include ALL terminology from the documents
    
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
            system="You are an expert translator and linguistics specialist with deep expertise in terminology extraction. When creating glossaries, you are thorough and comprehensive, extracting ALL domain-specific terms without summarizing.",
            messages=[
                {"role": "user", "content": analysis_prompt}
            ]
        )
        
        # Get response content
        analysis_result = message.content[0].text
        
        # Update progress
        progress_bar.progress(0.7)
        status_text.text("Processing results...")
        
        # Extract glossary terms based on selected method
        if extraction_method == "Standard":
            glossary_terms = extract_glossary_terms(analysis_result, source_language, target_language)
        else:  # Comprehensive
            # Try primary extraction first
            primary_terms = extract_glossary_terms(analysis_result, source_language, target_language)
            
            # If primary extraction found few terms, always try secondary
            if len(primary_terms) < 20:
                st.warning(f"Primary extraction only found {len(primary_terms)} terms. Trying secondary extraction...")
                secondary_terms = extract_terms_with_second_prompt(analysis_result, source_language, target_language)
                
                # Use the method that found more terms
                if len(secondary_terms) > len(primary_terms):
                    st.info(f"Using secondary extraction results with {len(secondary_terms)} terms instead of primary with {len(primary_terms)} terms")
                    glossary_terms = secondary_terms
                else:
                    glossary_terms = primary_terms
            else:
                glossary_terms = primary_terms
        
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
        st.success(f"Successfully generated translation resources for {source_language} to {target_language} with {len(glossary_terms)} glossary terms!")
        
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
        import traceback
        st.error(traceback.format_exc())
        progress_bar.empty()
        status_text.empty()

# Show information when no files are uploaded
elif submit_button and not uploaded_files:
    st.error("Please upload at least one file.")

# Add footer
st.markdown("---")
st.caption("This application uses Claude AI to analyze documents and create translation resources.")
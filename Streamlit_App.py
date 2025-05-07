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
import time

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
        elif '\n2.' in content:
            # Alternative way to split sections if ## not used 
            sections = content.split('\n2.')
            if len(sections) > 1:
                sections[1] = '2.' + sections[1]  # Add back the "2." that was removed in the split
        else:
            # Fallback if no ## headers
            sections = [content]
        
        # Process first section (if not starting with ##)
        if not content.strip().startswith('## '):
            first_part = sections[0]
            # Find any level 1 header 
            if '\n1.' in first_part:
                parts = first_part.split('\n1.', 1)
                if len(parts) > 1:
                    # Add any intro text before the header
                    if parts[0].strip():
                        doc.add_paragraph(parts[0].strip())
                    # Add the header and its content
                    doc.add_heading('1. ' + parts[1].strip().split('\n', 1)[0], level=1)
                    rest_of_content = parts[1].strip().split('\n', 1)[1] if len(parts[1].strip().split('\n', 1)) > 1 else ''
                    if rest_of_content:
                        doc.add_paragraph(rest_of_content)
                else:
                    doc.add_paragraph(first_part)
            else:
                doc.add_paragraph(first_part)
            
            # Adjust sections to avoid duplication
            if len(sections) > 1:
                sections = sections[1:]
            
        # Process remaining sections
        for section in sections:
            if not section.strip():
                continue
                
            # Handle different header formats
            if section.startswith('2.'):
                # Add level 1 header
                header_line = section.split('\n', 1)[0]
                doc.add_heading(header_line, level=1)
                
                # Add content after the header
                content = section.split('\n', 1)[1] if len(section.split('\n', 1)) > 1 else ''
                
                # Process content by paragraphs
                if content:
                    paragraphs = content.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            # Check if this is a subheader
                            if para.strip().startswith('Background:') or para.strip().startswith('Translation Approach:'):
                                doc.add_heading(para.strip().split('\n', 1)[0], level=2)
                                # Add content after the subheader
                                subcontent = para.strip().split('\n', 1)[1] if len(para.strip().split('\n', 1)) > 1 else ''
                                if subcontent:
                                    doc.add_paragraph(subcontent)
                            else:
                                doc.add_paragraph(para)
            else:
                # Regular section processing for ## format
                lines = section.split('\n', 1)
                if len(lines) > 0:
                    title = lines[0]
                    doc.add_heading(title, level=1)
                    
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
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to extract glossary terms from the API response
def extract_glossary_terms(text, source_language, target_language):
    """Extract glossary terms from the API response"""
    try:
        # Process the response to extract terms
        terms = []
        lines = text.split('\n')
        table_started = False
        
        for line in lines:
            # Skip empty lines and lines without pipes
            if not line.strip() or '|' not in line:
                continue
                
            # Skip header and separator rows
            if 'Source Term' in line or source_language in line or target_language in line or 'Term' in line:
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
            st.success(f"Successfully extracted {len(terms)} glossary terms!")
            return terms
        else:
            st.warning("Could not extract glossary terms from table format. Trying direct query...")
            return request_glossary_direct(source_language, target_language, text)
            
    except Exception as e:
        st.error(f"Error extracting glossary terms: {e}")
        st.warning("Attempting direct glossary extraction...")
        return request_glossary_direct(source_language, target_language, text)

# Function to directly request a glossary
def request_glossary_direct(source_language, target_language, analysis_text=None):
    """Request a glossary directly from Claude"""
    try:
        client = get_anthropic_client()
        if not client:
            st.error("Could not initialize Anthropic client for glossary extraction")
            return []
        
        extraction_prompt = f"""
        Create a comprehensive glossary for translation from {source_language} to {target_language} based on the document analysis below.
        
        Format the glossary as a table with these exact columns:
        | {source_language} Term | {target_language} Translation | English Reference | Example |
        
        Include AT LEAST [number] terms and domain-specific terminology.
        Do not translate all words - focus on specialized terms and important phrases.
        
        Analysis context:
        {analysis_text[:4000] if analysis_text else "Related to hospitality, tourism, and spa services."}
        """
        
        # Replace [number] with an appropriate target based on language complexity
        if source_language in ["Bulgarian", "Romanian", "Hungarian", "Czech", "Turkish", "Arabic", "Russian", "Japanese", "Chinese"]:
            extraction_prompt = extraction_prompt.replace("[number]", "80")
        else:
            extraction_prompt = extraction_prompt.replace("[number]", "50")
        
        st.info("Making direct glossary request...")
        message = client.messages.create(
            model="claude-3-opus-20240229",  # Use the most capable model for terminology
            max_tokens=4000,
            temperature=0.1,
            system="You are a terminology expert specialized in creating comprehensive multilingual glossaries with accurate translations. Generate a detailed professional glossary with as many terms as possible.",
            messages=[
                {"role": "user", "content": extraction_prompt}
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
            if any(header in line.lower() for header in ['term', 'source', 'target', 'translation', 'english', 'example']):
                table_started = True
                continue
                
            if '---' in line or '===' in line:
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
            st.success(f"Successfully extracted {len(terms)} glossary terms via direct query!")
            return terms
        else:
            st.warning("All extraction methods failed. Creating minimal glossary.")
            return [{"source_term": "Extraction Failed", 
                     "target_term": "Please check the analysis document for terminology", 
                     "english_reference": "", 
                     "example": ""}]
    
    except Exception as e:
        st.error(f"Error in direct glossary request: {e}")
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

# Function to make a combined analysis and glossary API call (to avoid rate limits)
def generate_combined_analysis_and_glossary(source_language, target_language, combined_content, model, analysis_style):
    client = get_anthropic_client()
    if not client:
        st.error("Could not initialize Anthropic client")
        return None, None
    
    # Create a combined prompt that requests both analysis and glossary
    if analysis_style == "Detailed (with translator persona)":
        combined_prompt = f"""
        I need you to provide two separate and complete analyses of these {source_language} documents for translation into {target_language}:

        PART 1: CONTENT ANALYSIS
        
        1. Analysis of the Content and Subject Matter:
           - Identify the specific organization, company, or entity these documents belong to
           - Name the specific location mentioned in the documents
           - Categorize each document individually (Document 1, Document 2, etc.) with its purpose
           - List the key topics and services covered in each document
           - Identify the content domain and any specialized fields involved
        
        2. Translator Persona: {source_language} to {target_language} Specialist
           - Give the persona a name that would be appropriate for a {target_language} native speaker
           - Specialization: Clearly state their expertise relevant to these documents
           - Background:
             * Education details (university, degree)
             * Years of experience
             * Specialized training relevant to the document domains
             * Language proficiency levels
             * Previous relevant work experience
           - Translation Approach:
             * How they handle register (formal/informal)
             * How they handle specialized terminology
             * Cultural adaptation approaches
             * Consistency strategies
             * How they handle legal or technical content
             * Tone preservation techniques
        
        PART 2: GLOSSARY OF TERMS
        
        After the analysis, please provide a comprehensive glossary formatted as a markdown table with the following columns:
        | {source_language} Term | {target_language} Translation | English Reference | Example |
        
        For the glossary:
        - Include AT LEAST 80 domain-specific terms from the documents
        - Focus on specialized terminology relevant to the document domain
        - Provide accurate {target_language} translations
        - Include an English reference term for each entry
        - Where possible, include an example phrase from the original documents
        
        Here are the document contents:
        
        {combined_content}
        """
    else:
        # Basic analysis prompt
        combined_prompt = f"""
        I need you to provide two separate analyses of these {source_language} documents for translation into {target_language}:
        
        PART 1: CONTENT ANALYSIS
        
        Provide a brief analysis:
        - Main subject matter (what type of documents are these?)
        - Primary content areas and domain
        - Key challenges for translation
        
        PART 2: GLOSSARY OF TERMS
        
        After the analysis, please provide a comprehensive glossary formatted as a markdown table with the following columns:
        | {source_language} Term | {target_language} Translation | English Reference | Example |
        
        For the glossary:
        - Include AT LEAST 80 domain-specific terms from the documents
        - Focus on specialized terminology relevant to the document domain
        - Provide accurate {target_language} translations
        - Include an English reference term for each entry
        - Where possible, include an example phrase from the original documents
        
        Here are the document contents:
        
        {combined_content}
        """
    
    # Make a single API call for both parts
    try:
        message = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.1,
            system="You are an expert translator analyzing documents for translation. First provide detailed document analysis, then create a comprehensive terminology glossary with at least 80 domain-specific terms.",
            messages=[
                {"role": "user", "content": combined_prompt}
            ]
        )
        
        combined_result = message.content[0].text
        
        # Split the result into analysis and glossary parts
        if "PART 2: GLOSSARY OF TERMS" in combined_result:
            parts = combined_result.split("PART 2: GLOSSARY OF TERMS", 1)
            analysis_result = parts[0].strip()
            glossary_result = parts[1].strip() if len(parts) > 1 else ""
        elif "## Glossary of Terms" in combined_result:
            parts = combined_result.split("## Glossary of Terms", 1)
            analysis_result = parts[0].strip()
            glossary_result = "## Glossary of Terms" + parts[1].strip() if len(parts) > 1 else ""
        elif "|" in combined_result:
            # If there's a table, use that as a separator
            lines = combined_result.split("\n")
            table_start = -1
            
            for i, line in enumerate(lines):
                if "|" in line and (i+1 < len(lines) and "---" in lines[i+1]):
                    table_start = i
                    break
            
            if table_start > 0:
                analysis_result = "\n".join(lines[:table_start-1]).strip()
                glossary_result = "\n".join(lines[table_start-1:]).strip()
            else:
                analysis_result = combined_result
                glossary_result = ""
        else:
            # Fallback
            analysis_result = combined_result
            glossary_result = ""
        
        return analysis_result, glossary_result
        
    except Exception as e:
        st.error(f"Error in combined API call: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

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
        index=1,  # Default to claude-3-opus for better analysis
        help="Select the Claude model to use. Opus is more powerful but slower, Haiku is faster but less capable."
    )
    
    # Analysis style selection
    analysis_style = st.radio(
        "Analysis Style:",
        options=["Detailed (with translator persona)", "Basic"],
        index=0,  # Default to detailed
        help="Detailed creates a full analysis with persona. Basic provides minimal analysis."
    )

    # API mode to avoid rate limits
    api_mode = st.radio(
        "API Mode:",
        options=["Single Call (recommended)", "Separate Calls"],
        index=0,  # Default to single call
        help="Single Call avoids rate limits but may have less detailed results. Separate Calls gives better quality but may hit rate limits."
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

    # Show total character count
    character_count = len(combined_content)
    token_estimate = character_count // 4  # Rough estimate
    st.info(f"Total character count: {character_count} (est. {token_estimate} tokens)")
    
    # Check if content is too long and truncate if necessary
    max_tokens = 90000  # Maximum safe input size
    if token_estimate > max_tokens:
        st.warning(f"Content may be too large for analysis. Truncating to approximately {max_tokens} tokens.")
        combined_content = combined_content[:max_tokens*4]  # Truncate to max token limit
    
    # Add a small example if the content is too short (for testing)
    if len(combined_content) < 10:
        st.warning("The extracted content is very short. Adding example text for testing.")
        if source_language == "Bulgarian":
            combined_content += """
            Пример текст на български:
            Здравейте! Това е пример за текст на български. Тук може да добавите вашия собствен текст.
            Използвайте реален текст, за да получите по-добри резултати от анализа.
            """
    
    # Process the content based on selected API mode
    try:
        if api_mode == "Single Call (recommended)":
            # Use combined API call approach to avoid rate limits
            status_text.text("Sending request to Claude (combined analysis and glossary)...")
            progress_bar.progress(0.4)
            
            analysis_result, glossary_result = generate_combined_analysis_and_glossary(
                source_language, target_language, combined_content, model, analysis_style
            )
            
            if not analysis_result:
                st.error("Failed to generate analysis. Please try again with smaller files or use a different model.")
                st.stop()
                
            progress_bar.progress(0.8)
            status_text.text("Processing results...")
            
        else:  # Separate Calls mode
            # Use two separate API calls (may hit rate limits)
            # Create analysis prompt for Claude
            if analysis_style == "Detailed (with translator persona)":
                analysis_prompt = f"""
                I need you to analyze these {source_language} documents for translation into {target_language}. Please:
                
                1. Analysis of the Content and Subject Matter:
                   - Identify the specific organization, company, or entity these documents belong to
                   - Name the specific location mentioned in the documents
                   - Categorize each document individually (Document 1, Document 2, etc.) with its purpose
                   - List the key topics and services covered in each document
                   - Identify the content domain and any specialized fields involved
                
                2. Translator Persona: {source_language} to {target_language} Specialist
                   - Give the persona a name that would be appropriate for a {target_language} native speaker
                   - Specialization: Clearly state their expertise relevant to these documents
                   - Background:
                     * Education details (university, degree)
                     * Years of experience
                     * Specialized training relevant to the document domains
                     * Language proficiency levels
                     * Previous relevant work experience
                   - Translation Approach:
                     * How they handle register (formal/informal)
                     * How they handle specialized terminology
                     * Cultural adaptation approaches
                     * Consistency strategies
                     * How they handle legal or technical content
                     * Tone preservation techniques
                
                DO NOT include any glossary/terminology list in your response.
                
                Here are the document contents:
                
                {combined_content}
                """
            else:
                # Basic analysis prompt
                analysis_prompt = f"""
                Provide a brief analysis of these {source_language} documents for translation into {target_language}:
                
                1. Main subject matter (what type of documents are these?)
                2. Primary content areas and domain
                3. Key challenges for translation
                
                Keep this analysis brief and focused. DO NOT include any glossary or terminology list.
                
                Here are the document contents:
                
                {combined_content}
                """
                
            # Create glossary prompt
            glossary_prompt = f"""
            I need you to create a comprehensive glossary of terms from these {source_language} documents for translation into {target_language}.
            
            Format the glossary as a table with these exact columns:
            | {source_language} Term | {target_language} Translation | English Reference | Example |
            
            Guidelines:
            - Include AT LEAST 80 domain-specific terms from the documents
            - Focus on specialized terminology relevant to the document domain
            - Provide accurate {target_language} translations
            - Include an English reference term for each entry
            - Where possible, include an example phrase from the original documents
            - DO NOT include common words unless they have a specialized meaning in this context
            - DO NOT include explanations or notes outside the table
            
            Here are the document contents:
            
            {combined_content}
            """
            
            # Make API request to Claude for analysis
            status_text.text("Generating content analysis and translator persona...")
            progress_bar.progress(0.4)
            
            analysis_message = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.1,
                system="You are an expert document analyst specializing in translation preparation. You identify document types, subject matter, and create detailed translator personas with backgrounds and specialized approaches.",
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            # Get analysis response content
            analysis_result = analysis_message.content[0].text
            
            # Update progress
            progress_bar.progress(0.6)
            status_text.text("Generating comprehensive terminology glossary...")
            
            # Wait a bit to avoid rate limits
            time.sleep(3)
            
            # Make API request to Claude for glossary
            glossary_message = client.messages.create(
                model="claude-3-opus-20240229",  # Use Opus for best terminology results
                max_tokens=4000,
                temperature=0.1,
                system="You are a terminology specialist expert in creating comprehensive multilingual glossaries. You extract ALL domain-specific terms from documents and provide accurate translations without summarizing or reducing the number of terms.",
                messages=[
                    {"role": "user", "content": glossary_prompt}
                ]
            )
            
            # Get glossary response content
            glossary_result = glossary_message.content[0].text
            
            # Update progress
            progress_bar.progress(0.8)
            status_text.text("Processing glossary and preparing documents...")
        
        # Extract glossary terms
        glossary_terms = extract_glossary_terms(glossary_result, source_language, target_language)
        
        # Create Excel file
        excel_data = create_glossary_excel(glossary_terms)
        
        # Create Word document with analysis only
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
        
        # Show preview of analysis and glossary
        with st.expander("Preview Analysis"):
            st.markdown(analysis_result)
            
        with st.expander("Preview Glossary (First 10 Terms)"):
            if glossary_terms and len(glossary_terms) > 0:
                df_preview = pd.DataFrame(glossary_terms[:10])
                st.dataframe(df_preview.rename(columns={
                    'source_term': 'Source Term',
                    'target_term': 'Target Translation',
                    'english_reference': 'English Reference',
                    'example': 'Example'
                }))
                st.info(f"Total terms in glossary: {len(glossary_terms)}")
            else:
                st.warning("No glossary terms to preview")
        
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
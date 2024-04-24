import streamlit as st
from streamlit_pills import pills
from bs4 import BeautifulSoup
from xhtml2pdf import pisa
import fitz
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os, json
import tempfile
import PyPDF2
import shutil

def generate_pdf():
    if 'output' in st.session_state:
        st.session_state.output = ''
    st.session_state.generate = True
    
def combine_responses(section):
    if section in st.session_state.completed:
        return
    p = section.split(' ')[-1][:-2]
    if p:
        parent_sec = 'Section ' + p
        combine_responses(parent_sec)
    
    st.session_state.output += st.session_state.all_titles[section]
    if section in st.session_state.output_dict:
        for subpart in st.session_state.output_dict[section]:
            if st.session_state.output_dict[section][subpart]:
                st.session_state.output += st.session_state.output_dict[section][subpart].prettify() + "\n"
                
    st.session_state.completed[section] = True

def convert_html_to_pdf(code, pdf_file):

    # Open the HTML file and read its content
    # try:
    #     with open(html_file, 'r', encoding='utf-8') as html_file:
    #         html_content = html_file.read()
    # except FileNotFoundError:
    #     print("HTML file not found.")
    #     return

    # Create PDF output file
    try:
        with open(pdf_file, 'wb') as pdf_file:
            # Convert HTML to PDF with custom CSS styles
            pisa_status = pisa.CreatePDF(code, dest=pdf_file)
    except Exception as e:
        print(f"Error occurred during PDF generation: {e}")
        return

    # Check if PDF generation was successful
    if pisa_status and pisa_status.err:
        print(f"PDF generation failed: {pisa_status.err}")
    else:
        print("PDF generated successfully!")

def extract_outline(pdf_path):
    outlines = []
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    if toc:
        for item in toc:
            title = item[1]
            page_num = item[2] + 1  # Page numbers are 0-based
            lvl = item[0]
            
            outlines.append({'level':lvl,'title': title, 'page_num': page_num})
    return outlines

def create_table_of_contents(outlines, pdf_path):
    # Create a new PDF document for the table of contents
    toc_pdf_path = pdf_path.replace('.pdf', '_2TOC.pdf')
    doc = SimpleDocTemplate(toc_pdf_path, pagesize=letter)
    
    # Define styles for the table of contents
    styles = getSampleStyleSheet()
    toc_title_style = styles['Title']
    toc_title_style.alignment = 1  # Center alignment
    # toc_title_style.textColor = colors.blue
    
    toc_entry_style = ParagraphStyle(
        'TOCEntry',
        parent=styles['Normal'],
        leftIndent=20,
    )
    
    # Add the title for the table of contents
    toc_title = Paragraph("Table of Contents", toc_title_style)
    
    # Add the title and a spacer to the document
    elements = [toc_title, Spacer(1, 12)]
    
    # Add entries to the Table of Contents
    for item in outlines:
        indent = item['level'] * 10  # Adjust indentation based on the level of the entry
        title = Paragraph(item['title'], styles['Normal'])
        page_num = Paragraph(str(item['page_num']), styles['Normal'])
        
        # Add indentation and align title and page number
        title_style = ParagraphStyle(
            'TOCTitle',
            parent=styles['Normal'],
            leftIndent=indent,
            rightIndent=indent,
            firstLineIndent=4,
            alignment=0,  # Left alignment
        )
        page_num_style = ParagraphStyle(
            'TOCPageNum',
            parent=styles['Normal'],
            leftIndent=0,
            rightIndent=0,  # Adjusted for right alignment
            firstLineIndent=0,
            alignment=2,  # Right alignment
        )
        
        elements.append(Paragraph(item['title'], title_style))
        elements.append(Paragraph(str(item['page_num']), page_num_style))
    
    doc.build(elements)
    
    print(f'Table of Contents created successfully: {toc_pdf_path}')

def merge_pdfs(output_path, *input_paths):
    # Merge multiple PDFs into a single PDF
    pdf_writer = PyPDF2.PdfWriter()

    for path in input_paths:
        pdf_reader = PyPDF2.PdfReader(path)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    with open(output_path, 'wb') as out:
        pdf_writer.write(out)

if 'all_titles' not in st.session_state:
    file_path = 'prompts.json'
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        st.session_state.all_titles = data['all_sections']
        
if 'completed' not in st.session_state:
    st.session_state.completed = {}

if "output" not in st.session_state:
    st.session_state.output = ""
if "output_dict" not in st.session_state:
    st.session_state.output_dict = {}
if 'generate' not in st.session_state:
    st.session_state.generate = False

st.subheader("Convert generated data to PDF")
if "all_sections" in st.session_state:
    selected_section = pills("Select the section:", list(st.session_state.all_sections.keys()))
    if selected_section:
        if selected_section not in st.session_state.output_dict:
            st.session_state.output_dict[selected_section] = {}
        selected_subpart = pills("Select the subpart:", list(st.session_state.all_sections[selected_section].keys()), index = None)
        if selected_subpart:
            selected_response = pills("Choose any one response among these:", st.session_state.all_sections[selected_section][selected_subpart], format_func=lambda x:x.get_text(), index = None) 
            if selected_response:
                st.session_state.output_dict[selected_section][selected_subpart] = selected_response
                # st.write(selected_response.prettify())


st.button("Create pdf file", on_click=generate_pdf)
if st.session_state.generate:
    output_path = st.text_input("Provide the output path")
    st.session_state.output_path = output_path
    if st.session_state.output_path:
        with st.spinner('Preparing output data...'):           
            for section in st.session_state.all_titles:
                if section in st.session_state.output_dict:
                    combine_responses(section)

            
            code = '''<!DOCTYPE html>
                <html>
                <head>
                    <title>Critical Features of Study Design</title>
                    <style>
                        @page {
                            size: A4; /* Specify page size (e.g., A4, Letter, etc.) */
                            margin-top: 2cm; /* Specify top margin */
                            margin-right: 2cm; /* Specify right margin */
                            margin-bottom: 2cm; /* Specify bottom margin */
                            margin-left: 2cm;
                            
                            @frame footer_frame {           /* Static frame */
                                -pdf-frame-content: footer_content;
                                left: 500pt; 
                                width: 512pt; 
                                top: 800pt; 
                                height: 20pt;
                            }
                        }
                        body {
                            font-family: Arial, sans-serif; /* Specify font family */
                            font-size: 14px; /* Specify font size */
                        }
                        table, th, td {
                            font-family: Arial, sans-serif; /* Specify font family */
                            font-size: 8px; /* Specify font size */
                            border: 1px solid;
                            text-align: left;
                        }
                        h1 {
                            font-size: 26px; /* Specify font size for headings */
                            font-weight: bold; /* Specify font weight */
                        }
                        h2 {
                            font-size: 24px; /* Specify font size for headings */
                            font-weight: bold; /* Specify font weight */
                        }
                        h3 {
                            font-size: 22px; /* Specify font size for headings */
                            font-weight: bold;
                        }
                        h4 {
                            font-size: 20px; /* Specify font size for headings */
                            font-weight: bold; /* Specify font weight */
                        }
                        h5 {
                            font-size: 18px; /* Specify font size for headings */
                            font-weight: bold; /* Specify font weight */
                        }
                        h6 {
                            font-size: 16px; /* Specify font size for headings */
                            font-weight: bold; /* Specify font weight */
                        }
                        p {
                            font-size: 14px; /* Specify font size for paragraphs */
                        }
                    </style>
                </head>
                ''' + f'''
                <body>
                <div id="footer_content">Page <pdf:pagenumber>
                    of <pdf:pagecount>
                </div>

                {st.session_state.output}

                </body>
                </html>'''
        st.success("Data prepeared Successfully")
        
        pdf_output_path = st.session_state.output_path + '.pdf'
        temp_dir = tempfile.mkdtemp()
        content_path = os.path.join(temp_dir, 'content_' + pdf_output_path)
        toc_path = os.path.join(temp_dir, 'toc_' + pdf_output_path)
        with st.spinner('Generating PDF...'):
            # Convert HTML to PDF
            convert_html_to_pdf(code, content_path)

            # HTML(html_file_path).write_pdf(pdf_output_path)
        with st.spinner('Generating Table of Contents...'):
            outlines = extract_outline(pdf_output_path)
            create_table_of_contents(outlines, toc_path)

        with st.spinner('Merging both files...'):
            # Merge PDF files
            merged_pdf_path = os.path.join(temp_dir, 'merged.pdf')
            merge_pdfs(merged_pdf_path, toc_path, content_path)

            # Move merged PDF to cache directory
            cache_dir = tempfile.gettempdir()  # Get system's temporary directory
            final_pdf_path = os.path.join(pdf_output_path)
            shutil.move(merged_pdf_path, final_pdf_path)

            print(f"Merged PDF saved to: {final_pdf_path}")
            shutil.rmtree(temp_dir)
        
        st.success("Pdf file generated successfully")
        st.session_state.output = ''
        st.session_state.output_path = None
        st.session_state.generate = False
    else:
        st.warning('Provide output pdf file name.')

# %% [markdown]
# ### Magical PDF V2.1

# %% [markdown]
# # -- IMPORTS --

# %%
import pandas as pd
from pdfminer.layout import LAParams, LTTextBoxHorizontal, LTTextBoxVertical, LTTextLineHorizontal, LTTextLineVertical, LTChar, LTTextContainer,LTAnno, LTFigure, LTImage, LTRect, LTLine, LTPage,LTTextLine
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfpage import PDFPage
from pdfminer.high_level import extract_pages
import re
import fitz  # PyMuPDF
import os
from rapidfuzz import fuzz
import io
import cv2 
import numpy as np
import pytesseract
from PIL import Image
from tqdm import tqdm
import time
from spellchecker import SpellChecker
import string

# %% [markdown]
# # -- PATTERNS --

# %%
patterns = [
    # Patterns for "Section"
    r"^section\s\d{2}\s\d{2}\s\d{2}",
    r"^section—\d+",
    r"^section\s\d+",
    r"^Section\s\d{2}\s\d{2}\s\d{2}",
    r"^Section—\d+",
    r"^Section\s\d+",
    r"^SECTION\s\d{2}\s\d{2}\s\d{2}",
    r"^SECTION—\d+",
    r"^SECTION\s\d+",

    # Patterns for "Chapter"
    r"^chapter—\d+",
    r"^chapter\s\d+",
    r"^Chapter—\d+",
    r"^Chapter\s\d+",
    r"^CHAPTER—\d+",
    r"^CHAPTER\s\d+",
    # Patters for Exhibit
    r"^Exhibit\s[A-Z]\s",
    r"^Exhibit\s\d+\s",
    # Patterns for "Part"
    r"^part\s\d",
    r"^Part\s\d",
    r"^PART\s\d",

    # Patterns for "Step"
    r"^Step\s\d\:",
    r"^STEP\s\d\:",
    r"^step\s\d\:",

    # Patterns for "Appendix"
    r"^Appendix\s[a-z]\s",
    r"^APPENDIX\s[a-z]\s",
    r"^appendix\s[a-z]\s",

    # Patterns for "Article"
    r"^Article\s\d\.\d\d\s",
    r"^Article\s\d",
    r"^ARTICLE\s\d",
    r"^article\s\d",

    # Patterns for "Schedule"
    r"^SCHEDULE\s[A-Z]\s",

    # Patterns for Roman Enumeral
    r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\.\s*",  # "I., II."
    r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\)\s*",  # "I), II)"
    r"^\s*\((I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\)\s*",# "(I), (II)"
    r"^\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\.\s*",  # "i., ii."
    r"^\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\)\s*",  # "i), ii)"
    r"^\s*\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\)\s*",# "(i), (ii)"

    # Others: i, v, and x are excluded on purpose
    r"^\(\s*([A-RTUWYZ])\s*\)",   # "(H)" & " (H) "
    r"^(?!I\.)[A-RTUWYZ]\.\s",    # "H. "
    r"^([A-RTUWYZ])\)\s",         # "H) "
    r"^[A-RTUWYZ]\.\s",           # "H. "
    r"^[A-RTUWYZ][A-RTUWYZ]\.\s", # "HH. "
    r"^\(\s*([a-rtuwyz])\s*\)",   # "(h)" & " (h) "
    r"^(?!i\.)[a-rtuwyz]\.\s",    # "h. "
    r"^([a-rtuwyz])\)\s",         # "h) "
    r"^[a-rtuwyz]\.\s",           # "h. "
    r"^[a-rtuwyz][a-rtuwyz]\.\s", # "hh. "
    # Patterns with Letters
    r"^[A-Z]\.\d+\s",
    r"^[A-Z]\.\d+\.\d+\s",
    r"^[A-Z]\.\d+\.\d+\.\d+\s",
    r"^[A-Z]\.\d+\.\d+\.\d+\.\d+\s",
    
    # Patterns with Numbers
    r"^\d{2}\s\d{2}\s\d{2}",
    r"^\d{1,5}\.\s",
    r"^\d{1,5}\.\d{1,5}\s",
    r"^\d{1,5}\.\d{1,5}.\s",
    r"^[a-z]\s\d{1,5}\.\d{1,5}\s",
    r"^[A-Z]\s\d{1,5}\.\d{1,5}\s",
    r"^\d{1,5}\.\d{1,5}\*\s",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\s",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}.\s",
    r"^\Δ\s\d{1,5}\.\d{1,5}\.\d{1,5}\s",
    r"^[a-z]\s\d{1,5}\.\d{1,5}\.\d{1,5}\s",
    r"^[A-Z]\s\d{1,5}\.\d{1,5}\.\d{1,5}\s",
    r"^\Δ\s\d{1,5}\.\d{1,5}\.\d{1,5}\*\s",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\*\s",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\s",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}.\s",
    r"^[a-z]\s\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\s",
    r"^[A-Z]\s\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\s",
    r"^\Δ\s\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\s",
    r"^\Δ\s\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\*\s",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\*\s",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\s",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}.\s",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\*\s",
    r"^\d{1,5}\)\s",
    r"^\d{1,5}\s",
    r"^\d{1,5}\*\s",
    r"^\[\d{1,5}\]",
    r"^\d{1,5}\—",
    r"^\d{1,5}\.\—",
    r"^\d{1,5}\.\d{1,5}\—",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\—",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\—",
    r"^\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\.\d{1,5}\—",
    r"^\(\d+\)",

    #### next patterns used to split Section and PT ###
    r"^b\[\d+\]\s",
    r"^B\[\d+\]\s",
    r"^\[Table\/Figure\]\s",
    r"^\[table\/figure\]\s",
    r"^P\[\d+\]\s",
    r"^p\[\d+\]\s",
]

bullet_patterns = [
    r"^\W\s",  # Any non-alphanumeric character followed by a space
    r"^\•\s",
    r"^\-\s",
    r"^\*\s",
    r"^\"\s",
    r"^\'\s",
    r"^\u2022\s",  # Bullet point (•)
    r"^\u25AA\s",  # Black small square (▪)
    r"^\u25AB\s",  # White small square (▫)
    r"^\u25A0\s",  # Black square (■)
    r"^\u25A1\s",  # White square (□)
    r"^\u25B2\s",  # Black up-pointing triangle (▲)
    r"^\u25B3\s",  # White up-pointing triangle (△)
    r"^\u25BC\s",  # Black down-pointing triangle (▼)
    r"^\u25BD\s",  # White down-pointing triangle (▽)
    r"^\u25C6\s",  # Black diamond (◆)
    r"^\u25C7\s",  # White diamond (◇)
    r"^\u25CF\s",  # Black circle (●)
    r"^\u25CB\s",  # White circle (○)
    r"^\u25D8\s",  # Inverse bullet (◘)
    r"^\u25D9\s",  # Inverse white circle (◙)
    r"^\u25E6\s",  # White bullet (◦)
    r"^\u2756\s",  # Diamond with a question mark inside (❖)
    r"^\uF0E0\s",  # Special bullet character ()
    r"^\uf07d\s",
]

table_figure_patterns = [
    r"^table\s\d+",             
    r"^table\s\d+:\s",          
    r"^table\s\d+\.\d+",        
    r"^figure\s\d+",            
    r"^figure\s\d+\.\d+",
    r"^Table\s\d+",             
    r"^Table\s\d+:\s",          
    r"^Table\s\d+\.\d+",        
    r"^Figure\s\d+",            
    r"^Figure\s\d+\.\d+",
    r"^TABLE\s\d+",             
    r"^TABLE\s\d+:\s",          
    r"^TABLE\s\d+\.\d+",        
    r"^FIGURE\s\d+",            
    r"^FIGURE\s\d+\.\d+",      
]

# %% [markdown]
# # -- OCR ENGINE --> PDF to Brut Text

# %% [markdown]
# ## Step 1 - Run PDF Miner on PDF to extract text from PDF

# %%
def recalculate_bounding_box(chars):
    x0, y0, x1, y1 = None, None, None, None
    for char in chars:
        if isinstance(char, LTChar):
            char_x0, char_y0, char_x1, char_y1 = char.bbox
            if x0 is None or char_x0 < x0:
                x0 = char_x0
            if y0 is None or char_y0 < y0:
                y0 = char_y0
            if x1 is None or char_x1 > x1:
                x1 = char_x1
            if y1 is None or char_y1 > y1:
                y1 = char_y1
    return x0, y0, x1, y1

def rebuild_lineboxes(document_path, laparams): # Rebuild PdfMiner Line Boxes Based on Characters Coordinates
    lineboxes_data = []
    previous_x0, previous_y0, previous_x1, previous_y1 = None, None, None, None

    # Extract pages with custom laparams
    pages = list(extract_pages(document_path, laparams=laparams))
    for page_number, page in enumerate(tqdm(pages, desc="Processing Pages for Lineboxes"), start=1):
        # Check and rotate page if necessary
        if page.width > page.height:
            # Page is in landscape mode; rotate it
            page.rotate = 90
        for element in page:
            if isinstance(element, LTTextBoxHorizontal):
                for text_line in element:
                    chars = []
                    for char in text_line:
                        if isinstance(char, LTChar):
                            chars.append(char)
                    if chars:
                        x0, y0, x1, y1 = recalculate_bounding_box(chars)
                        line_text = ''.join([char.get_text() for char in chars])
                        
                        if previous_x0 is not None and previous_y0 is not None:
                            x_gap = abs(round(x0 - previous_x0, 2))
                            y_gap = abs(round(y0 - previous_y0, 2))
                        else:
                            x_gap, y_gap = None, None

                        lineboxes_data.append({
                            'Page': page_number,
                            'Text': line_text.strip(),
                            'x0': round(x0, 2),
                            'y0': round(y0, 2),
                            'x1': round(x1, 2),
                            'y1': round(y1, 2),
                            'x_gap': x_gap,
                            'y_gap': y_gap
                        })

                        previous_x0, previous_y0, previous_x1, previous_y1 = x0, y0, x1, y1
    
    df = pd.DataFrame(lineboxes_data)
    return df

def CreateDataFrameFromPDF(document_path): # Main PDF Miner Funcion & Initiate PDF Miner LAParams 
    laparams = LAParams(boxes_flow=None)
    # Rebuild lineboxes based on characters
    df = rebuild_lineboxes(document_path, laparams)
    
    return df

# %% [markdown]
# ## Step 2 - Run Tesseract OCR on text not detected by PDF Miner, Higlight findinds in PDF and consolidate Dataframe

# %%
def redact_combined_text(pdf_path, combined_df, output_pdf_path, problematic_pages=None, dpi_value=72): # Redact PDF with white boxes over text already detected by PDF Miner
    doc = fitz.open(pdf_path)
    dpi_scale = dpi_value / 72  # Adjust scaling if needed
    padding = 0
    # Default to empty list if no problematic pages specified
    if problematic_pages is None:
        problematic_pages = [page_num + 1 for page_num in range(len(doc)) if doc[page_num].rotation != 0]

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Get page dimensions
        page_width = page.mediabox_size[0]
        page_height = page.mediabox_size[1]

        # Filter DataFrame for the current page
        page_df = combined_df[combined_df['Page'] == page_num + 1]

        for _, row in page_df.iterrows():
            x0 = row['x0'] - padding
            y0 = row['y0'] - padding
            x1 = row['x1'] + padding
            y1 = row['y1'] + padding

            # Adjust y-coordinates for PyMuPDF coordinate system
            y0_pdf = page_height - y1
            y1_pdf = page_height - y0

            if (page_num + 1) in problematic_pages:
                # Use swapped coordinates for problematic pages
                rect = fitz.Rect(y0, x0, y1, x1)
            else:
                # Use standard coordinates for normal pages
                rect = fitz.Rect(x0, y0_pdf, x1, y1_pdf)
            
            # Add the redaction annotation with a white fill
            page.add_redact_annot(rect, fill=(1, 1, 1))

        # Apply the redactions to the page
        page.apply_redactions()

    # Save the redacted PDF
    doc.save(output_pdf_path)
    print(f"Redacted PDF saved as '{output_pdf_path}'.")

def preprocess_image_1(image_bytes, gamma = 2.3): # PDF Preprossessing
    # Gamma Correction
    image = Image.open(io.BytesIO(image_bytes))
    open_cv_image = np.array(image)

    # Convert to float and normalize
    img_float = open_cv_image.astype(np.float32) / 255.0

    # Apply gamma correction
    gamma_corrected = np.power(img_float, gamma)

    # Convert back to uint8
    gamma_corrected = np.uint8(gamma_corrected * 255)

    if len(gamma_corrected.shape) == 3:
        gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2GRAY)
    else:
        gray = gamma_corrected
    filtered_image = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Thresholding
    _, binary = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary
def preprocess_image_2(image_bytes, gamma = 3.5): # PDF Preprossessing
    # Gamma Correction
    image = Image.open(io.BytesIO(image_bytes))
    open_cv_image = np.array(image)

    # Convert to float and normalize
    img_float = open_cv_image.astype(np.float32) / 255.0

    # Apply gamma correction
    gamma_corrected = np.power(img_float, gamma)

    # Convert back to uint8
    gamma_corrected = np.uint8(gamma_corrected * 255)

    if len(gamma_corrected.shape) == 3:
        gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2GRAY)
    else:
        gray = gamma_corrected
    filtered_image = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Thresholding
    _, binary = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary
def preprocess_image_3(image_bytes, gamma = 3.5): # PDF Preprossessing
    # Gamma Correction
    image = Image.open(io.BytesIO(image_bytes))
    open_cv_image = np.array(image)

    # Convert to float and normalize
    img_float = open_cv_image.astype(np.float32) / 255.0

    # Apply gamma correction
    gamma_corrected = np.power(img_float, gamma)

    # Convert back to uint8
    gamma_corrected = np.uint8(gamma_corrected * 255)

    if len(gamma_corrected.shape) == 3:
        gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2GRAY)
    else:
        gray = gamma_corrected

    # Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
def preprocess_image_4(image_bytes, gamma = 2.0): # PDF Preprossessing
    # Gamma Correction
    image = Image.open(io.BytesIO(image_bytes))
    open_cv_image = np.array(image)

    # Convert to float and normalize
    img_float = open_cv_image.astype(np.float32) / 255.0

    # Apply gamma correction
    gamma_corrected = np.power(img_float, gamma)

    # Convert back to uint8
    gamma_corrected = np.uint8(gamma_corrected * 255)

    if len(gamma_corrected.shape) == 3:
        gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2GRAY)
    else:
        gray = gamma_corrected
    filtered_image = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Thresholding
    _, binary = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary

def remove_watermark(image_bytes, padding=3): # Function to remove watermarks
    image = Image.open(io.BytesIO(image_bytes))
    open_cv_image = np.array(image)
    # Convert to grayscale
    if len(open_cv_image.shape) == 3:
        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = open_cv_image
    # Invert the image
    inverted_image = cv2.bitwise_not(gray_image)

    # Apply thresholding to keep only solid black areas
    _, thresholded_image = cv2.threshold(inverted_image, 125, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise and expand black regions
    kernel = np.ones((padding, padding), np.uint8)
    padded_image = cv2.dilate(thresholded_image, kernel, iterations=1)
    
    # Convert back to image bytes
    final_image = cv2.bitwise_not(padded_image)
    pil_image = Image.fromarray(final_image)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    processed_image_bytes = img_byte_arr.getvalue()

    return processed_image_bytes
def make_near_white_pixels_white(img_bytes, threshold=180):
    """
    Convert near-white pixels to pure white.
    threshold: The lower bound for pixel values considered 'nearly white'.
               Any pixel with all channels > threshold will be set to pure white.
    """
    # Open image from bytes
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img)
    
    # Create a mask for near-white pixels
    # For a pixel to be considered near-white, all its channels should be above `threshold`
    near_white_mask = (arr[:,:,0] > threshold) & (arr[:,:,1] > threshold) & (arr[:,:,2] > threshold)
    
    # Set these pixels to pure white
    arr[near_white_mask] = [255, 255, 255]
    
    # Convert back to PIL image if needed
    return arr
def extract_words_with_tesseract(doc_path, dpi_value, preprocessing_functions, oem_value, psm_value, existing_df=None, is_there_a_watermark=False): # Tesseract OCR word Extraction
    data = []
    scale = 72 / dpi_value
    # Open the PDF document
    doc = fitz.open(doc_path)
    num_pages = len(doc)
    if existing_df is not None:
        existing_words = existing_df[['Page', 'Text', 'x0', 'y0', 'x1', 'y1']].copy()
    else:
        existing_words = pd.DataFrame(columns=['Page', 'Text', 'x0', 'y0', 'x1', 'y1'])

    for page_num in tqdm(range(num_pages), desc="Processing Pages", unit="page"):
        page = doc[page_num]
        page_height_points = page.rect.height
        # Render page to an image (Pixmap)
        pix = page.get_pixmap(dpi=dpi_value)
        img_bytes = pix.tobytes()

        # Open the image with PIL
        image = Image.open(io.BytesIO(img_bytes))

        # Remove watermark if necessary
        if is_there_a_watermark:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
            # Remove watermark
            processed_image_bytes = remove_watermark(image_bytes)
            image = Image.open(io.BytesIO(processed_image_bytes))
            
        # Rotate image if necessary
        if image.width > image.height:
            # Rotate the image to portrait orientation
            image = image.rotate(-90, expand=True)

        # Loop through preprocessing functions
        page_has_words = False
        for round_name, preprocess_func in preprocessing_functions:
            # Preprocess the image
            preprocessed_image = preprocess_func(img_bytes)
            pil_image = Image.fromarray(preprocessed_image)

            # Define Tesseract config
            custom_config = f'--oem {oem_value} --psm {psm_value}'

            # Get per-word bounding box information
            word_data = pytesseract.image_to_data(
                pil_image, config=custom_config, output_type=pytesseract.Output.DICT
            )

            # Check if there are any recognized words on this page
            # (Skip if no words or all words are empty strings)
            if all(word.strip() == '' for word in word_data['text']):
                # No words found by Tesseract for this preprocessing round
                continue
            else:
                page_has_words = True
            
            # If words are found, process them
            num_words = len(word_data['text'])
            for i in range(num_words):
                word = word_data['text'][i].strip()
                conf = int(word_data['conf'][i])
                if word and conf > 10:
                    x0 = word_data['left'][i]
                    y0 = word_data['top'][i]
                    width = word_data['width'][i]
                    height = word_data['height'][i]
                    x1 = x0 + width
                    y1 = y0 + height
                    text_height = height * scale
                    word_length = width * scale
                    x0 = x0 * scale
                    y0 = y0 * scale
                    x1 = x1 * scale
                    y1 = y1 * scale
                    y0_inverted = page_height_points - y1
                    y1_inverted = page_height_points - y0
                    page_number = page_num + 1
                    is_duplicate = False
                    threshold = 3 * scale
                    matching_words = existing_words[
                        (existing_words['Page'] == page_number) &
                        (existing_words['Text'] == word)
                    ]
                    for _, existing_word in matching_words.iterrows():
                        dx = abs(existing_word['x0'] - x0)
                        dy = abs(existing_word['y0'] - y0_inverted)
                        if dx <= threshold and dy <= threshold:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        # Append the data to the list
                        data.append({
                            'Page': page_number,
                            'Text': word,
                            'x0': x0,
                            'y0': y0_inverted,
                            'x1': x1,
                            'y1': y1_inverted,
                            'textHeight': text_height,
                            'wordLength': word_length,
                            'Confidence': conf
                        })
                        new_row = pd.DataFrame([{
                            'Page': page_number,
                            'Text': word,
                            'x0': x0,
                            'y0': y0_inverted,
                            'x1': x1,
                            'y1': y1_inverted
                        }])
                        existing_words = pd.concat([existing_words, new_row], ignore_index=True)

        # If after all preprocessing functions, no words were found, skip this page
        if not page_has_words:
            continue

    # Create a pandas DataFrame
    df = pd.DataFrame(data, columns=[
        'Page', 'Text', 'x0', 'y0', 'x1', 'y1', 'textHeight', 'wordLength', 'Confidence'])

    # Remove duplicates
    if existing_df is not None:
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.drop_duplicates(subset=['Page', 'Text', 'x0', 'y0', 'x1', 'y1'], inplace=True)
    else:
        combined_df = df
    return combined_df
def flatten_pdf(input_pdf_path, output_pdf_path, dpi=300):
    """
    Rasterizes each page of input_pdf_path into a PNG image, then rebuilds a PDF from these images.
    This results in a flattened PDF with no vector/text layers, just images.
    """
    temp_images = []
    doc = fitz.open(input_pdf_path)
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=dpi)
        img_data = pix.tobytes("png")
        img_path = f"flatten_temp_page_{page_num}.png"
        with open(img_path, "wb") as f:
            f.write(img_data)
        temp_images.append(img_path)
    doc.close()

    # Create a new PDF from the rendered images
    img_doc = fitz.open()
    for img_path in temp_images:
        img = fitz.open(img_path)
        pdf_bytes = img.convert_to_pdf()  # Convert image to a single-page PDF
        img_pdf = fitz.open("pdf", pdf_bytes)
        img_doc.insert_pdf(img_pdf)
        img.close()

    img_doc.save(output_pdf_path)
    img_doc.close()

    # Clean up temporary image files
    for img_path in temp_images:
        os.remove(img_path)

def tesseract_adjustment(PDFData, output_redacted_pdf_name, document_path, is_there_a_watermark, run_tesseract):
    dpiValue = 400
    oem_value = 3
    psm_value = 6

    # Extract text with PDFMiner
    print("Extracting text with PDFMiner...")
    pdfminer_df = PDFData
    document_dir = os.path.dirname(document_path)

    # If run_tesseract is False, skip Tesseract and just redact PDFMiner results
    if not run_tesseract:
        print("Skipping Tesseract OCR since run_tesseract is False.")

        # Ensure 'Source' column exists
        if 'Source' not in pdfminer_df.columns:
            pdfminer_df['Source'] = 'PDFMiner'

        # This will be our "combined" text since we have no Tesseract results
        combined_text_df = pdfminer_df.sort_values(by=['Page', 'y0', 'x0'], ascending=[True, True, True])
        final_redacted_pdf_path = os.path.join(document_dir, output_redacted_pdf_name)
        redact_combined_text(document_path, combined_text_df, final_redacted_pdf_path, dpi_value=dpiValue)
        print(f"Final redacted PDF saved as '{final_redacted_pdf_path}'.")
        return combined_text_df

    # If run_tesseract is True, continue with original logic:
    if pdfminer_df.empty:
        print("PDFMiner did not capture any text. Running Tesseract on the original PDF.")
        tesseract_input_path = document_path
    else:
        redacted_pdf_path = os.path.join(document_dir, 'pdfminer_redacted.pdf')
        print("Redacting extracted text from PDFMiner results...")
        redact_combined_text(document_path, pdfminer_df, redacted_pdf_path, dpi_value=dpiValue)

        # flattened_pdf_path = os.path.join(document_dir, 'pdfminer_redacted_flattened.pdf')
        # print("Flattening redacted PDF...")
        # flatten_pdf(redacted_pdf_path, flattened_pdf_path, dpi = dpiValue)
        tesseract_input_path = redacted_pdf_path

    print("Extracting words with Tesseract...")
    preprocessing_functions = [
        ("white_cleaning", make_near_white_pixels_white),
        ('preprocess_image_1', preprocess_image_1),
        ('preprocess_image_2', preprocess_image_2),
        ('preprocess_image_3', preprocess_image_3),
        ('preprocess_image_4', preprocess_image_4)
    ]
    start_time = time.time()
    combined_df = extract_words_with_tesseract(
        tesseract_input_path, dpiValue, preprocessing_functions, oem_value, psm_value, 
        existing_df=None, is_there_a_watermark=is_there_a_watermark
    )

    step_time = time.time() - start_time
    print(f"Tesseract OCR completed in {step_time:.2f} seconds")

    # Combine PDFMiner and Tesseract results
    tesseract_df = combined_df[['Page', 'Text', 'x0', 'y0', 'x1', 'y1']]
    tesseract_df['Source'] = 'Tesseract'
    if 'Source' not in pdfminer_df.columns:
        pdfminer_df['Source'] = 'PDFMiner'

    combined_text_df = pd.concat([pdfminer_df, tesseract_df], ignore_index=True)
    combined_text_df = combined_text_df.sort_values(by=['Page', 'y0', 'x0'], ascending=[True, True, True])
    final_redacted_pdf_path = os.path.join(document_dir, output_redacted_pdf_name)
    redact_combined_text(document_path, combined_text_df, final_redacted_pdf_path, dpi_value=dpiValue)

    # Clean up intermediate file if created
    if not pdfminer_df.empty:
        redacted_pdf_path = os.path.join(document_dir, 'pdfminer_redacted.pdf')
        if os.path.exists(redacted_pdf_path):
            os.remove(redacted_pdf_path) 
            print(f"Intermediate files '{redacted_pdf_path}' and has been deleted.")

    print(f"Final redacted PDF saved as '{final_redacted_pdf_path}'.")
    return combined_text_df

# %% [markdown]
# # -- OCR CLEANUP --> Brut Text to Cropped & Clean lines

# %% [markdown]
# ## Reorder based on Y coordinates

# %%
def verticalOrdering(df):
    # Convert 'y0' column to numeric values, errors='coerce' will replace non-numeric values with NaN
    df['y0'] = pd.to_numeric(df['y0'], errors='coerce')
    # Group by 'Page' and then apply sorting to each group
    df_sorted = df.groupby('Page', group_keys=False).apply(lambda x: x.sort_values(by='y0', ascending=False))
    return df_sorted

# %% [markdown]
# ## Remove Text on the Left and Right

# %%
def removeTextonTheLeftAndright(df, x0_Left, x0_Right):
    # Remove rows starting before x0coord
    filtered_df = df[(df['x0'] >= x0_Left) & (df['x0'] <= x0_Right)]
    return pd.DataFrame(filtered_df)

# %% [markdown]
# ## Remove Headers and Footers

# %%
def remove_headers_and_footers_Manual(df, header_y0, footer_y1):
    # Remove rows above the header_y0 value
    df = df[df['y0'] <= header_y0]
    # Remove rows below the footer_y1 value
    df = df[df['y1'] >= footer_y1]
    return pd.DataFrame(df)

# %%
def gatherHeaderAndFooterText(df, page_height, h_area):
    # Calculate header and footer limits
    header_limit = (h_area / 100) * page_height
    # Prepare lists to collect data for the output DataFrame
    data = []
    for page_number in df["Page"].unique():
        # Filter DataFrame for the current page
        page_df = df[df["Page"] == page_number]
        # Separate header and footer areas
        header_df = page_df[page_df["y0"] >= (page_height - header_limit)]
        footer_df = page_df[page_df["y1"] <= header_limit]
    
        # Collect header text and coordinates
        for _, row in header_df.iterrows():
            data.append({
                "Page": int(page_number),
                "Area": "Header",
                "Text": row["Text"],
                "y0": row["y0"],
                "y1": row["y1"]
            })

        # Collect footer text and coordinates
        for _, row in footer_df.iterrows():
            data.append({
                "Page": int(page_number),
                "Area": "Footer",
                "Text": row["Text"],
                "y0": row["y0"],
                "y1": row["y1"]
            })
    # Convert the collected data into a DataFrame
    return pd.DataFrame(data)

# %%
def classifyHeaderFooterText(df, similarity_threshold, pageCoverage): #Works but not if footer/header drastically different for a couple of pages
    # Step 1: Get the total number of pages in the document
    total_pages = df["Page"].nunique()
    # Step 2: Function to group similar texts using fuzzy matching
    def group_similar_texts(area_df, similarity_threshold):
        similarity_groups = []
        group_map = {}

        for idx, text in zip(area_df.index, area_df["Text"]):
            matched = False
            for group_id, group_texts in enumerate(similarity_groups):
                # Compare text with existing group representatives
                if any(fuzz.ratio(text, existing) >= similarity_threshold for existing in group_texts):
                    group_texts.append(text)
                    group_map[idx] = group_id
                    matched = True
                    break
            if not matched:
                # Create a new group
                similarity_groups.append([text])
                group_map[idx] = len(similarity_groups) - 1

        return group_map

    # Step 3: Create header and footer group maps separately
    headers = df[df["Area"] == "Header"].copy()
    footers = df[df["Area"] == "Footer"].copy()

    header_group_map = group_similar_texts(headers, similarity_threshold)
    footer_group_map = group_similar_texts(footers, similarity_threshold)

    # Assign similarity groups back to headers and footers
    headers["Similarity Group"] = headers.index.map(header_group_map)
    footers["Similarity Group"] = footers.index.map(footer_group_map)

    # Combine headers, footers, and the rest of the data back into the main DataFrame
    other_rows = df[~df["Area"].isin(["Header", "Footer"])]
    df = pd.concat([headers, footers, other_rows]).sort_index()

    # Step 4: Count occurrences for headers and footers separately
    header_occurrences = headers["Similarity Group"].value_counts()
    footer_occurrences = footers["Similarity Group"].value_counts()

    # Map occurrences back to the DataFrame
    df["Header Group Occurrences"] = df["Similarity Group"].map(header_occurrences).fillna(0).astype(int)
    df["Footer Group Occurrences"] = df["Similarity Group"].map(footer_occurrences).fillna(0).astype(int)

    # Step 5: Calculate the occurrence ratio
    df["Header Occurrence Ratio"] = (df["Header Group Occurrences"] / total_pages * 100).fillna(0)
    df["Footer Occurrence Ratio"] = (df["Footer Group Occurrences"] / total_pages * 100).fillna(0)

    # Step 6: Default Classification as 'Body'
    df["Classification"] = "Body"

    # Step 7: Classify as Header or Footer based on the occurrence ratio
    if total_pages > 100:
        df.loc[(df["Area"] == "Header") & (df["Header Occurrence Ratio"] >= pageCoverage/20), "Classification"] = "Header"
        df.loc[(df["Area"] == "Footer") & (df["Footer Occurrence Ratio"] >= pageCoverage/20), "Classification"] = "Footer"
    else:
        df.loc[(df["Area"] == "Header") & (df["Header Group Occurrences"] >= pageCoverage), "Classification"] = "Header"
        df.loc[(df["Area"] == "Footer") & (df["Footer Group Occurrences"] >= pageCoverage), "Classification"] = "Footer"
    return df

# %%
def getHeaderFooterLineCoordinates(df):
    df["Header Line"] = None
    df["Footer Line"] = None

    # Group by page to process headers and footers for each page
    for page, page_data in df.groupby("Page"):
        # Filter for headers and footers
        header_rows = page_data[page_data["Classification"] == "Header"]
        footer_rows = page_data[page_data["Classification"] == "Footer"]
        
        # Calculate header and footer line coordinates
        header_line = header_rows["y0"].min()-1 if not header_rows.empty else "No Header"
        footer_line = footer_rows["y1"].max()+1 if not footer_rows.empty else "No Footer"

        # Update the original DataFrame with the calculated values
        df.loc[df["Page"] == page, "Header Line"] = header_line
        df.loc[df["Page"] == page, "Footer Line"] = footer_line
        
    return df

# %%
def automaticallyRemoveHeadersFooter(File2, header_footer_lines):
    # Create a copy of File2 to avoid modifying the original DataFrame
    cleaned_df = File2.copy()

    # Iterate through each page to apply header and footer line boundaries
    for page in header_footer_lines["Page"].unique():
        # Get the header and footer line coordinates for the current page
        page_lines = header_footer_lines[header_footer_lines["Page"] == page]
        header_line = page_lines.iloc[0]["Header Line"]
        footer_line = page_lines.iloc[0]["Footer Line"]

        # Remove rows in the header area
        if header_line != "No Header":
            cleaned_df = cleaned_df[
                ~((cleaned_df["Page"] == page) & (cleaned_df["y1"] >= header_line))
            ]

        # Remove rows in the footer area
        if footer_line != "No Footer":
            cleaned_df = cleaned_df[
                ~((cleaned_df["Page"] == page) & (cleaned_df["y0"] <= footer_line))
            ]

    # Return the cleaned DataFrame
    return cleaned_df

# %%
def remove_headers_and_footers(document_path, File0, File2, header_y0, footer_y1, auto_header_footer, h_area):
    doc = fitz.open(document_path)  # Open the document
    page_height = doc[0].rect.height
    page_count = len(doc)

    if auto_header_footer:
        # Automatic detection workflow
        # Gather Header and Footer Text for all pages in a df "page"/"Area"/"Text"/"y0"/"y1"
        header_footer_text = gatherHeaderAndFooterText(File2, page_height, h_area)
        if header_footer_text.empty:
            df = File2
            header_footer_lines = []
        else:
            # Classify Header/Footer Text to exclude body text
            classified_text = classifyHeaderFooterText(header_footer_text, similarity_threshold=85, pageCoverage=2)
            # Get Header and Footer Line Coordinates
            header_footer_lines = getHeaderFooterLineCoordinates(classified_text)
            # Remove headers and footers automatically based on classification
            df = automaticallyRemoveHeadersFooter(File2, header_footer_lines)
    else:
        # Manual mode
        # Remove headers and footers based on manual parameters
        df = remove_headers_and_footers_Manual(File2, header_y0, footer_y1)

        # Construct a DataFrame for header_footer_lines using the provided manual coordinates.
        # If no header is intended, set it to "No Header", similarly for footer
        header_line_val = header_y0 if header_y0 is not None else "No Header"
        footer_line_val = footer_y1 if footer_y1 is not None else "No Footer"

        # Create a DataFrame that mirrors the structure from the automatic case,
        # applying the same header/footer line to all pages.
        header_footer_lines = pd.DataFrame({
            "Page": range(1, page_count + 1),
            "Header Line": [header_line_val] * page_count,
            "Footer Line": [footer_line_val] * page_count
        })

    return pd.DataFrame(df), pd.DataFrame(header_footer_lines)


# %% [markdown]
# ## merge Items with similar y0 and rebuild lines as needed based on x0 coord

# %%
def merge_items_with_similar_y0(df, tolerance):
    # Group by 'Page' and then apply sorting to each group
    df_sorted = df.groupby('Page', group_keys=False).apply(lambda x: x.sort_values(by='y0', ascending=False))
    
    # Initialize a list to store the merged rows
    merged_rows = []
    
    # Iterate through each page group
    for page, group in df_sorted.groupby('Page'):
        i = 0
        while i < len(group):
            current_row = group.iloc[i].copy()
            temp_rows = [current_row]

            while i + 1 < len(group) and abs(current_row['y0'] - group.iloc[i + 1]['y0']) < tolerance:
                next_row = group.iloc[i + 1]
                temp_rows.append(next_row)
                i += 1

            if len(temp_rows) > 1:
                temp_rows = sorted(temp_rows, key=lambda row: row['x0'])

            merged_row = temp_rows[0].copy()
            for row in temp_rows[1:]:
                merged_row['Text'] += ' ' + row['Text']
                merged_row['x0'] = min(merged_row['x0'], row['x0'])
                merged_row['x1'] = max(merged_row['x1'], row['x1'])
                merged_row['y0'] = min(merged_row['y0'], row['y0'])
                merged_row['y1'] = max(merged_row['y1'], row['y1'])

            merged_rows.append(merged_row)
            i += 1
    
    # Create a new DataFrame from the merged rows
    df_merged = pd.DataFrame(merged_rows)
    
    # Recalculate y_gap
    df_merged['y0'] = pd.to_numeric(df_merged['y0'], errors='coerce')
    df_merged['y_gap'] = df_merged['y0'].diff().fillna(0).abs().round(2)

    return df_merged

# %% [markdown]
# # -- RECREATE PARAGRAPHS --> Clean Lines to Clean Paragraphs with artificial headings

# %% [markdown]
# ## Calculate line gaps

# %%
def calculate_line_gap(df, margin, percentage_linegaps):
    # Drop rows with NaN values in the 'y_gap' column
    df = df.dropna(subset=['y_gap'])
    # Check if there are any valid 'y_gap' values
    if df.empty:
        raise ValueError("No valid 'y_gap' values found in the DataFrame.")
    
    # Count the occurrences of each 'y_gap' value
    counts = df['y_gap'].value_counts()
    
    # Calculate the total count of unique 'y_gap' values
    total_unique_count = counts.size
    
    # Calculate the cumulative percentage
    counts_with_percentage = pd.DataFrame({'Count': counts})
    counts_with_percentage['Percentage'] = (counts_with_percentage['Count'] / counts_with_percentage['Count'].sum()) * 100
    counts_with_percentage['CumulativePercentage'] = counts_with_percentage['Percentage'].cumsum()
    # Get the top % most common values
    top_percent = counts_with_percentage[counts_with_percentage['CumulativePercentage'] < percentage_linegaps]
    # Determine the highest value among the top 35% most common values
    if pd.isna(top_percent.index.max()):
        line_gap = counts.idxmax() + margin # Update with the most common value
    else:
        line_gap = top_percent.index.max() + margin


    return line_gap, counts_with_percentage, total_unique_count

# %% [markdown]
# ## Merge lines together into paragraphs

# %%
def merge_lines(df, line_gap):#base function - Does not include paragrph gap = line gap with pattern detection
    merged_data = []
    previous_row = None
    
    for index, row in df.iterrows():
        if previous_row is not None and row['y_gap'] <= line_gap:
            previous_row['Text'] += ' ' + row['Text']
            # Update the coordinates
            previous_row['x0'] = min(previous_row['x0'], row['x0'])
            previous_row['x1'] = max(previous_row['x1'], row['x1'])
            previous_row['y0'] = min(previous_row['y0'], row['y0'])
            previous_row['y1'] = max(previous_row['y1'], row['y1'])

        else:
            if previous_row is not None:
                merged_data.append(previous_row)
            previous_row = row.copy()
    
    if previous_row is not None:
        merged_data.append(previous_row)   
    return pd.DataFrame(merged_data)

# %%
def merge_linesWithPattern(df, line_gap, patterns, bullet_patterns, table_figure_patterns):#keeps lines separated if pattern is matched while paragrph gap = line gap
    merged_data = []
    previous_row = None 
    for index, row in df.iterrows():
        # Check if the row is within the vertical proximity (y_gap)
        if previous_row is not None and row['y_gap'] <= line_gap:
            # Now check for pattern matches only within the vertical proximity
            matches_patterns = any(re.search(pattern, row['Text']) for pattern in patterns)
            matches_bullet_patterns = any(re.search(pattern, row['Text']) for pattern in bullet_patterns)
            matches_table_figure_patterns = any(re.search(pattern, row['Text']) for pattern in table_figure_patterns)
            if not matches_patterns and not matches_bullet_patterns and not matches_table_figure_patterns:
                # If no pattern matches, merge the lines
                previous_row['Text'] += ' ' + row['Text']
                # Update the coordinates
                previous_row['x0'] = min(previous_row['x0'], row['x0'])
                previous_row['x1'] = max(previous_row['x1'], row['x1'])
                previous_row['y0'] = min(previous_row['y0'], row['y0'])
                previous_row['y1'] = max(previous_row['y1'], row['y1'])
            else:
                # If a pattern matches, don't merge and treat the row as a new entry
                #print(f"Skipping merge for row {index} due to pattern match: {row['Text']}")
                merged_data.append(previous_row)
                previous_row = row.copy()
        else:
            # If not within vertical proximity, consider it a new line
            if previous_row is not None:
                merged_data.append(previous_row)
            previous_row = row.copy()
    if previous_row is not None:
        merged_data.append(previous_row)
    return pd.DataFrame(merged_data)

# %% [markdown]
# ## Merge_paragraphs_with_page_break

# %%
def merge_paragraphs_with_page_break(df):
    # Create a list to store the rows
    rows = []
    # Iterate through the dataframe by index
    i = 0
    while i < len(df) - 1:
        current_row = df.iloc[i].copy()
        next_row = df.iloc[i + 1]
        # Check if the current row is not on the same page as the previous row
        if current_row['Page'] != next_row['Page']:
            # Define your condition for merging here
            if (not current_row['Text'].endswith('.') and 
                not current_row['Text'].endswith(':') and
                not current_row['Text'].endswith(';') and
                #not current_row['Text'].endswith('END OF SECTION') and
                #not next_row['Text'][0].isupper() and
                not any(re.match(pattern, next_row['Text']) for pattern in patterns) and
                not any(re.match(pattern, next_row['Text']) for pattern in bullet_patterns) and
                not any(re.match(pattern, next_row['Text']) for pattern in table_figure_patterns)):
                # Merge the 'Text' from the current and next rows
                merged_text = current_row['Text'] + ' ' + next_row['Text']
                # Update the 'Text' in the current row
                current_row['Text'] = merged_text
                # Update the coordinates
                current_row['x0'] = min(current_row['x0'], next_row['x0'])
                current_row['x1'] = max(current_row['x1'], next_row['x1'])
                current_row['y0'] = min(current_row['y0'], next_row['y0'])
                current_row['y1'] = max(current_row['y1'], next_row['y1'])
                # Add the merged row to the list
                rows.append(current_row)
                # Skip the next row as it's been merged
                i += 2
                continue
        # Add the current row to the list
        rows.append(current_row)
        i += 1
    # Add the last row of the dataframe to the list if it wasn't merged
    if i == len(df) - 1:
        rows.append(df.iloc[-1])
    # Create a new DataFrame from the list of rows
    merged_df = pd.DataFrame(rows)
    return merged_df

# %% [markdown]
# ## Add P[x], B[y], Table/Figure[z]

# %%
def add_Artificial_Headers(df, patterns, bullet_patterns, table_figure_patterns):
    p_counter = 1
    b_counter = 1
    
    def normalize_text(text):
        # Remove all leading whitespace and replace multiple spaces with a single space
        return re.sub(r'\s+', ' ', text.lstrip())

    def check_patterns(text):
        nonlocal p_counter, b_counter
        normalized_text = normalize_text(text)
        
        if any(re.match(pattern, normalized_text) for pattern in bullet_patterns):
            result = f"B[{b_counter}] {text}"
            b_counter += 1
            # p_counter = 1  # Reset P counter
            return result
        elif any(re.match(pattern, normalized_text) for pattern in table_figure_patterns):
            return f"[Table/Figure] {text}"
        elif any(re.match(pattern, normalized_text) for pattern in patterns):
            p_counter = 1  # Reset P counter
            b_counter = 1  # Reset B counter
            return text
        else:
            result = f"P[{p_counter}] {text}"
            p_counter += 1
            return result

    new_df = df.copy()
    new_df['Text'] = new_df['Text'].apply(check_patterns)
    return new_df

# %% [markdown]
# # -- RVTM FORMATTING --> Clean Paragraphs to Section and Primary field of RVTM

# %% [markdown]
# ## Split Section and Primary Text

# %%
def move_headers_to_section(df, patterns):
    # Create an empty 'Section' column
    df['Section'] = ''
    
    # Iterate over each pattern in the patterns list
    for pattern in patterns:
        # Apply the pattern to create the 'Section' column using case-insensitive matching
        df['Section'] = df.apply(lambda row: re.search(pattern, row['Text'], re.IGNORECASE).group() if re.search(pattern, row['Text'], re.IGNORECASE) else row['Section'], axis=1)
        # Remove the matched pattern from 'Text' using case-insensitive matching
        df['Text'] = df.apply(lambda row: re.sub(pattern, '', row['Text'], flags=re.IGNORECASE) if re.search(pattern, row['Text'], re.IGNORECASE) else row['Text'], axis=1)
    
    # Reorder columns to place 'Section' before 'Text'
    cols = df.columns.tolist()
    text_index = cols.index('Text')
    cols.insert(text_index, cols.pop(cols.index('Section')))
    df = df[cols]
    return df

# %% [markdown]
# ## Create Section Type column

# %%
def sectionType(df, patterns):
    df.insert(1, 'Section Type', '')
    for i, row in df.iterrows():
        section_text = row['Section']
        for pattern in patterns:
            if re.match(pattern, section_text):
                df.at[i, 'Section Type'] = pattern
                break
    return df

# %% [markdown]
# ## Create Section Hiearchy Column

# %%
def sectionHierarchy(df): ### V1 Version Safe to use utill V2 is adjusted
    df.insert(2, 'Section Hierarchy', '')  # Insert a new column for Section Hierarchy
    df.insert(1, 'Dict', '')  # Insert a new column for Dictionnary
    hierarchy_dict = {}
    last_type = None
    last_hierarchy = None

    # Section types Roman
    RomanStyle1 = "^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\.\s*"
    RomanStyle2 = "^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\)\s*"
    RomanStyle3 = "^\s*\((I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\)\s*"
    RomanStyle4 = "^\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\.\s*"
    RomanStyle5 = "^\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\)\s*"
    RomanStyle6 = "^\s*\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\)\s*"
    
    # Section Types Non Roman
    NonRomanStyle1 = r"^(?!I\.)[A-RTUWYZ]\.\s"   
    NonRomanStyle2 = r"^([A-RTUWYZ])\)\s"    
    NonRomanStyle3 = r"^\(\s*([A-RTUWYZ])\s*\)"
    NonRomanStyle4 = r"^(?!i\.)[a-rtuwyz]\.\s"   
    NonRomanStyle5 = r"^([a-rtuwyz])\)\s"      
    NonRomanStyle6 = r"^\(\s*([a-rtuwyz])\s*\)"

    margin = 10

    for index, row in df.iterrows():
        section_type = row['Section Type']
        current_x0 = row['x0']
        current_section = row['Section']  # Get the Section value
        
        if index > 0:  # Ensure you're not accessing the previous row for the first row
            previous_section_type = df.at[index - 1, 'Section Type']
            previous_text = df.at[index - 1, 'Text']
            previous_x0 = df.at[index - 1, 'x0']
            previous_section = df.at[index - 1, 'Section']
        
        if section_type in hierarchy_dict: # Use the existing hierarchy if this type was seen before
            
            hierarchy, _ = hierarchy_dict[section_type]

            ### Fix the paragraph and bullet increment issue where we had a higher hierarchy level between.
            if ((str(section_type) == "^P\[\d+\]\s") or #and (str(previous_section_type) != "^P\[\d+\]\s" and str(previous_section_type) != "^B\[\d+\]\s") or
                (str(section_type) == "^B\[\d+\]\s") #and (str(previous_section_type) != "^P\[\d+\]\s" and str(previous_section_type) != "^B\[\d+\]\s")
                ):
                # Gather previous section and extract "N" value
                _ , previous_index = hierarchy_dict[section_type]
                SectionValuePrevindex = df.at[previous_index, 'Section']
                match = re.search(r"[PB]\[(\d+)\]", SectionValuePrevindex)
                n_value_previous = int(match.group(1))

                # Extract the number N from current P[N] or B[N]
                match = re.search(r"[PB]\[(\d+)\]", current_section)
                n_value = int(match.group(1))

                # If current = previous add one and save result in section column
                if n_value <= n_value_previous:
                    new_n_value = n_value_previous + 1
                    if str(section_type) == "^P\[\d+\]\s":
                        NewSection = f"P[{new_n_value}]"
                    if str(section_type) == "^B\[\d+\]\s":
                        NewSection = f"B[{new_n_value}]"
                    df.at[index, 'Section'] = NewSection
                    # Update the dictionary with the current index
                    hierarchy_dict[section_type] = (hierarchy, index)
                    # print(f"Current Index : {index}")
                    # print(f"Previous Index : {previous_index}")
                    # print(f"Previous Section : {SectionValuePrevindex}")
                    # print(f"new section : {NewSection}")
            ### END Fix the paragraph and bullet increment issue where we had a higher hierarchy level between
            else:
                #Update Dictionary Section value with most up to date
                del hierarchy_dict[section_type] # Remove current with old data
                hierarchy_dict[section_type] = (hierarchy, index) #add current with fresh data             

        else: # Set hierarchy for a new type
            if last_type is None:
                hierarchy = 0  # First type gets hierarchy 0
                #hierarchy_dict[section_type] = hierarchy
                hierarchy_dict[section_type] = (hierarchy, index)
                #df.at[index, 'Dict'] = hierarchy_dict # Update "Dict" column for troubleshooting only

            else:# If this section is a different type and we already have a hierarchy for the last one
                ### Base case : We increase Hierarchy by one and add the new type with their new hierarchy to the dictionnary
                hierarchy = last_hierarchy + 1
                #hierarchy_dict[section_type] = hierarchy
                hierarchy_dict[section_type] = (hierarchy, index)
                #df.at[index, 'Dict'] = hierarchy_dict # Update "Dict" column for troubleshooting only
                
                #### Paragraph P[N] issue where the paragraph does not have a proper section to ensure it is not taken in account for
                if (str(previous_section_type) == "^P\[\d+\]\s" and not previous_text.endswith(":") or
                    str(previous_section_type) == "^B\[\d+\]\s" or #and not previous_text.endswith(":") or
                    str(previous_section_type) == "^\[Table\/Figure\]\s"
                ):
                    #Get the last key in the dictionary
                    last_key = list(hierarchy_dict.keys())[-1] 
                    last_item = hierarchy_dict.pop(last_key) # Remove last item that was added in the else statement above
                    last_key = list(hierarchy_dict.keys())[-1]
                    last_item = hierarchy_dict.pop(last_key) # Remove previous P[N] item 
                    hierarchy = last_hierarchy #update Hierarchy to previous
                    #hierarchy_dict[section_type] = hierarchy # add current element to the list so hierarchy and types are updated moving forward
                    hierarchy_dict[section_type] = (hierarchy, index)
                    

                ### Logic to deal with some of Roman Numeral Numbers issues
                section_type = str(section_type)
                previous_section_type = str(previous_section_type)
                
                if (RomanStyle1 == section_type and NonRomanStyle1 == previous_section_type or
                    RomanStyle2 == section_type and NonRomanStyle2 == previous_section_type or
                    RomanStyle3 == section_type and NonRomanStyle3 == previous_section_type or
                    RomanStyle4 == section_type and NonRomanStyle4 == previous_section_type or 
                    RomanStyle5 == section_type and NonRomanStyle5 == previous_section_type or
                    RomanStyle6 == section_type and NonRomanStyle6 == previous_section_type
                ):
                    
                    if abs(current_x0 - previous_x0) <= margin: # This is in case we have H then I but there was in indentation +> we ignore
                        hierarchy = last_hierarchy
                        section_type = previous_section_type
                        
                ### END OF ROMAN ENUMERAL FIX ###
        # Update DataFrame
        df.at[index, 'Section Hierarchy'] = hierarchy
        df.at[index, 'Section Type'] = section_type

        # If current hierarchy is lower than the previous one, clear out the dictionary
        if last_hierarchy is not None and hierarchy < last_hierarchy:
            keys_to_remove = [key for key, value in hierarchy_dict.items() if value[0] > hierarchy]
            for key in keys_to_remove:
                del hierarchy_dict[key]

        # Update last_type and last_hierarchy for the next iteration
        last_type = section_type
        last_hierarchy = hierarchy

    return df

# %%
def sectionHierarchyToAdjust(df): #V2 Version to adjust
    df.insert(2, 'Section Hierarchy', '')  # Insert a new column for Section Hierarchy
    df.insert(1, 'Dict', '')  # Insert a new column for Dictionary
    hierarchy_dict = {}
    last_type = None
    last_hierarchy = None

    # Section types Roman
    RomanStyle1 = r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\.\s*"
    RomanStyle2 = r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\)\s*"
    RomanStyle3 = r"^\s*\((I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)\)\s*"
    RomanStyle4 = r"^\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\.\s*"
    RomanStyle5 = r"^\s*(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\)\s*"
    RomanStyle6 = r"^\s*\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx)\)\s*"
    
    # Section Types Non Roman
    NonRomanStyle1 = r"^(?!I\.)[A-RTUWYZ]\.\s"   
    NonRomanStyle2 = r"^([A-RTUWYZ])\)\s"    
    NonRomanStyle3 = r"^\(\s*([A-RTUWYZ])\s*\)"
    NonRomanStyle4 = r"^(?!i\.)[a-rtuwyz]\.\s"   
    NonRomanStyle5 = r"^([a-rtuwyz])\)\s"      
    NonRomanStyle6 = r"^\(\s*([a-rtuwyz])\s*\)"

    margin = 5

    for index, row in df.iterrows():
        section_type = row['Section Type']
        current_x0 = row['x0']
        current_section = row['Section']  # Get the Section value
        
        if index > 0:
            previous_section_type = df.at[index - 1, 'Section Type']
            previous_text = df.at[index - 1, 'Text']
            previous_x0 = df.at[index - 1, 'x0']
            previous_section = df.at[index - 1, 'Section']
        
        
        if section_type in hierarchy_dict: 
            _, previous_index = hierarchy_dict[section_type]
            previous_x0_value = float(df.at[previous_index, 'x0'] or 0)
            hierarchy, _ = hierarchy_dict[section_type]
            previous_hierarchy = df.at[index - 1, 'Section Hierarchy']
            match_found = section_type == "^P\[\d+\]\s" or section_type == "^B\[\d+\]\s"
            if abs(current_x0 - previous_x0_value) >= margin and current_x0 >= previous_x0_value:
                hierarchy = last_hierarchy + 1
                hierarchy_dict[section_type] = (hierarchy, index)
                if str(section_type) == "^P\[\d+\]\s":
                     df.at[index, 'Section']= "P[1]"
                elif str(section_type) == "^B\[\d+\]\s":
                    df.at[index, 'Section']= "B[1]"
            else:
                if match_found:
                    for prev_idx in range(index - 1, -1, -1):
                        if df.at[prev_idx, 'Section Hierarchy'] == "0":
                            break  # Stop the loop when Section Hierarchy is "0"
                        if section_type in hierarchy_dict and abs(current_x0 - df.at[prev_idx, 'x0']) < 2:
                            last_matching_hierarchy = df.at[prev_idx, 'Section Hierarchy']
                            if last_matching_hierarchy < hierarchy:
                                df.at[index, 'Section Hierarchy'] = last_matching_hierarchy
                                hierarchy = last_matching_hierarchy
                                hierarchy_dict[section_type] = (hierarchy, index)
                                break
            if previous_hierarchy >= hierarchy:
                if ((str(section_type) == "^P\[\d+\]\s") or 
                    (str(section_type) == "^B\[\d+\]\s")):
                    last_matching_hierarchy = None      
                    prev_idx = None
                    last_matching_index = None

                    for prev_idx in range(index - 1, -1, -1):
                        df.at[index, 'Section Hierarchy']=hierarchy
                        last_matching_hierarchy = df.at[prev_idx, 'Section Hierarchy']
                        if df.at[prev_idx, 'Section Hierarchy'] == "0": 
                            break
                        elif ((str(df.at[prev_idx, 'Section Type']) == "^P\[\d+\]\s" or
                            str(df.at[prev_idx, 'Section Type']) == "^B\[\d+\]\s")):
                            #if last_matching_hierarchy == df.at[index, 'Section Hierarchy']: Condition does not work, don't know why
                            if str(df.at[index, 'Section Hierarchy'])== str(last_matching_hierarchy):
                                last_matching_index = prev_idx
                                break
                            
                    if last_matching_index is not None:
                        SectionValuePrevindex = df.at[last_matching_index, 'Section']
                        match = re.search(r"[PB]\[(\d+)\]", SectionValuePrevindex)
                        n_value_previous = int(match.group(1)) if match else 0
                        match = re.search(r"[PB]\[(\d+)\]", current_section)
                        n_value = int(match.group(1)) if match else 0
                        new_n_value = n_value_previous + 1
                        if str(section_type) == "^P\[\d+\]\s":
                            NewSection = f"P[{new_n_value}]"
                        elif str(section_type) == "^B\[\d+\]\s":
                            NewSection = f"B[{new_n_value}]"
                        df.at[index, 'Section'] = NewSection
                        hierarchy_dict[section_type] = (hierarchy, index)
        else:
            if last_type is None:
                hierarchy = 0
                hierarchy_dict[section_type] = (hierarchy, index)
            else: 
                #Need a condition here for B[N] to get reset to 1 if there is no B[N] in the dictionary 
                # Ex: 
                # P[1],B[1]
                # P[1],B[2]
                # P[2]
                # P[2],B[3] this is wrong (should be P[2],B[1])
                hierarchy = last_hierarchy + 1
                hierarchy_dict[section_type] = (hierarchy, index)
                if (str(previous_section_type) == "^P\[\d+\]\s" and not previous_text.endswith(":") or
                    str(previous_section_type) == "^B\[\d+\]\s" or #and not previous_text.endswith(":") or
                    str(previous_section_type) == "^\[Table\/Figure\]\s"
                ):
                    #Get the last key in the dictionary
                    last_key = list(hierarchy_dict.keys())[-1] 
                    last_item = hierarchy_dict.pop(last_key) # Remove last item that was added in the else statement above
                    last_key = list(hierarchy_dict.keys())[-1]
                    last_item = hierarchy_dict.pop(last_key) # Remove previous P[N] item 
                    hierarchy = last_hierarchy #update Hierarchy to previous
                    #hierarchy_dict[section_type] = hierarchy # add current element to the list so hierarchy and types are updated moving forward
                    hierarchy_dict[section_type] = (hierarchy, index)

        df.at[index, 'Section Hierarchy'] = hierarchy
        df.at[index, 'Section Type'] = section_type
                        ### Logic to deal with some of Roman Numeral Numbers issues
        if (RomanStyle1 == section_type and NonRomanStyle1 == previous_section_type or
            RomanStyle2 == section_type and NonRomanStyle2 == previous_section_type or
            RomanStyle3 == section_type and NonRomanStyle3 == previous_section_type or
            RomanStyle4 == section_type and NonRomanStyle4 == previous_section_type or 
            RomanStyle5 == section_type and NonRomanStyle5 == previous_section_type or
            RomanStyle6 == section_type and NonRomanStyle6 == previous_section_type
            ):
                    
            if abs(current_x0 - previous_x0) <= margin: # This is in case we have H then I but there was in indentation +> we ignore
                hierarchy = last_hierarchy
                section_type = previous_section_type
                        
                ### END OF ROMAN NUMERAL FIX ###
        # Update DataFrame
        df.at[index, 'Section Hierarchy'] = hierarchy
        df.at[index, 'Section Type'] = section_type

        # If current hierarchy is lower than the previous one, clear out the dictionary
        if last_hierarchy is not None and hierarchy < last_hierarchy:
            keys_to_remove = [key for key, value in hierarchy_dict.items() if value[0] > hierarchy]
            for key in keys_to_remove:
                del hierarchy_dict[key]

        if last_hierarchy is not None and hierarchy < last_hierarchy:
            keys_to_remove = [key for key, value in hierarchy_dict.items() if value[0] > hierarchy]
            for key in keys_to_remove:
                del hierarchy_dict[key]

        last_type = section_type
        last_hierarchy = hierarchy

    return df

# %% [markdown]
# ## Create Combined Section Column

# %%
def create_combined_section(df):
    # Reset index to handle indexing cleanly
    df = df.reset_index(drop=True)

    combined_sections = []
        
    for i in range(df.shape[0]):
        current_hierarchy = df.loc[i, 'Section Hierarchy']
        # Strip the section in case it has leading/trailing spaces
        current_section = str(df.loc[i, 'Section']).strip()
            
        if current_hierarchy == 0:
            # Top level just becomes itself
            combined_section = current_section
        else:
            # Get the last combined section for reference
            prev_combined = combined_sections[-1]
            prev_hierarchy = df.loc[i-1, 'Section Hierarchy']

            # Split the previous combined section by ", "
            prev_parts = prev_combined.split(", ")

            if current_hierarchy > prev_hierarchy:
                # Add a new "child" section
                combined_section = f"{prev_combined}, {current_section}"
            elif current_hierarchy == prev_hierarchy:
                # Replace the last section at this level
                combined_section = ", ".join(prev_parts[:-1]) + f", {current_section}"
            else:
                # Move up the hierarchy and replace the section at that level
                levels_to_keep = current_hierarchy
                # Keep only the first `levels_to_keep` parts and add the new one
                combined_section = ", ".join(prev_parts[:levels_to_keep]) + f", {current_section}"

        # Just to ensure no accidental extra whitespace creeps in
        combined_section = combined_section.strip()
        combined_sections.append(combined_section)
        
    # Insert the new column right before "Section" or wherever you'd like
    df.insert(df.columns.get_loc('Section'), 'Combined Section', combined_sections)

    return df

# %% [markdown]
# # -- GENERATE OUTPUTS --

# %% [markdown]
# ## Print Rectangles on PDF based on X,Y coordinates

# %%
def drawDetectedtext(df, doc, page_height, color, width, source_condition=None, condition_color=None, problematic_pages=None):
    # If problematic_pages not provided, default to pages with rotations
    if problematic_pages is None:
        problematic_pages = [page_num + 1 for page_num in range(len(doc)) if doc[page_num].rotation != 0]

    for _, row in df.iterrows():
        page_number = int(row["Page"]) - 1  # PyMuPDF uses zero-based page indexing
        page = doc.load_page(page_number)
        
        # Extract coordinates
        x0, y0, x1, y1 = row["x0"], row["y0"], row["x1"], row["y1"]

        # Apply coordinate transformations:
        # If the page is problematic, swap coordinates as done in redact_combined_text
        if (page_number + 1) in problematic_pages:
            rect = fitz.Rect(y0, x0, y1, x1)
        else:
            # Adjust the y-coordinates for PyMuPDF coordinate system on normal pages
            y0_adjusted = page_height - y1
            y1_adjusted = page_height - y0
            rect = fitz.Rect(x0, y0_adjusted, x1, y1_adjusted)
        
        # Highlight or draw rectangle based on the source condition
        if source_condition and row["Source"] == source_condition:
            # Highlight Tesseract text
            page.add_highlight_annot(rect)
        else:
            # Draw a rectangle for other text
            page.draw_rect(rect, color=color, width=width)


# %%
def drawHeaderFooterArea(doc, page_height, page_width, h_area, color, segment_length, width, spacing, problematic_pages=None):
    # If problematic_pages not provided, default to pages with rotations
    if problematic_pages is None:
        problematic_pages = [page_num + 1 for page_num in range(len(doc)) if doc[page_num].rotation != 0]

    # Calculate the y positions for the header and footer lines
    header_y = (h_area / 100) * page_height
    footer_y = page_height - header_y

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)

        # Check if this page is rotated
        if (page_number + 1) in problematic_pages:
            # For rotated pages, treat header_y and footer_y as x-coordinates for vertical lines
            # Instead of drawing lines across the width at a certain y, 
            # we draw them across the height at a certain x.

            # Draw vertical "header" segments
            x_pos = 0
            while x_pos < page_height:
                # Here, we are swapping x and y usage to draw vertical lines instead of horizontal
                page.draw_line(
                    p1=(header_y, x_pos),
                    p2=(header_y, x_pos + segment_length),
                    color=color,
                    width=width
                )
                x_pos += segment_length + spacing

            # Draw vertical "footer" segments
            x_pos = 0
            while x_pos < page_height:
                page.draw_line(
                    p1=(footer_y, x_pos),
                    p2=(footer_y, x_pos + segment_length),
                    color=color,
                    width=width
                )
                x_pos += segment_length + spacing

        else:
            # For non-rotated pages, draw horizontal lines as before.
            # Draw header line segments
            x_pos = 0
            while x_pos < page_width:
                page.draw_line(
                    p1=(x_pos, header_y),
                    p2=(x_pos + segment_length, header_y),
                    color=color,
                    width=width
                )
                x_pos += segment_length + spacing

            # Draw footer line segments
            x_pos = 0
            while x_pos < page_width:
                page.draw_line(
                    p1=(x_pos, footer_y),
                    p2=(x_pos + segment_length, footer_y),
                    color=color,
                    width=width
                )
                x_pos += segment_length + spacing


# %%
def drawHeaderFooterLines(df, doc, page_width, page_height, color, width, problematic_pages=None):
    # If problematic_pages not provided, default to pages with rotations
    if problematic_pages is None:
        problematic_pages = [page_num + 1 for page_num in range(len(doc)) if doc[page_num].rotation != 0]

    for _, row in df.iterrows():
        page_number = int(row["Page"]) - 1  # zero-based index for PyMuPDF
        header_line = row["Header Line"]
        footer_line = row["Footer Line"]

        page = doc.load_page(page_number)

        # Adjust the y-coordinates to PyMuPDF's coordinate system if lines exist
        header_line_adjusted = None if header_line == "No Header" else (page_height - header_line)
        footer_line_adjusted = None if footer_line == "No Footer" else (page_height - footer_line)

        # Check if this is a problematic (rotated) page
        if (page_number + 1) in problematic_pages:
            # On rotated pages, we treat the lines similarly to how text rectangles are treated:
            # Swap the coordinate axes. A horizontal line at y would become a vertical line at x.
            
            # Draw the header line if it exists
            if header_line_adjusted is not None:
                # Instead of drawing horizontally across page_width, draw vertically across page_height
                page.draw_line(
                    p1=(header_line_adjusted, 0),
                    p2=(header_line_adjusted, page_height),
                    color=color,
                    width=width
                )

            # Draw the footer line if it exists
            if footer_line_adjusted is not None:
                # Similarly, draw a vertical line for the footer
                page.draw_line(
                    p1=(footer_line_adjusted, 0),
                    p2=(footer_line_adjusted, page_height),
                    color=color,
                    width=width
                )
        else:
            # Non-rotated page: draw lines normally (horizontal)
            if header_line_adjusted is not None:
                page.draw_line(
                    p1=(0, header_line_adjusted),
                    p2=(page_width, header_line_adjusted),
                    color=color,
                    width=width
                )

            if footer_line_adjusted is not None:
                page.draw_line(
                    p1=(0, footer_line_adjusted),
                    p2=(page_width, footer_line_adjusted),
                    color=color,
                    width=width
                )


# %%
def printOnPDF(File2, File3, document_path, output_pdf_path, header_footer_lines, h_area):
    doc = fitz.open(document_path)
    page_height, page_width = doc[0].rect.height, doc[0].rect.width

    # Draw Detected Text
    drawDetectedtext(File2, doc, page_height, color=(1,0,0), width = 0.5, source_condition="Tesseract", condition_color=(1, 0.75, 0.8)) # print lines and Tesseract text 
    drawDetectedtext(File3, doc, page_height, color=(0, 0, 1), width=1) # print paragraphs
    # Draw Header and Footer Area
    h_area = 15  # Defines Header/Footer area (percentage of page height)
    drawHeaderFooterArea(doc, page_height, page_width, h_area, color=(0.5, 0.5, 0.5), segment_length=10, width=0.5, spacing=40)

    # Draw Header and Footer Lines
    drawHeaderFooterLines(header_footer_lines, doc, page_width, page_height, color=(0.75, 0, 0.75), width=1)
    
    # Save the annotated PDF
    doc.save(output_pdf_path)
    return

# %% [markdown]
# ## Print DF to Excel

# %%
def detect_missed_text_and_tables(document_path, dpi_value=300):
    document_dir = os.path.dirname(document_path)
    redacted_file_path = os.path.join(document_dir, "final_redacted.pdf")

    # Check if the redacted file exists
    if not os.path.isfile(redacted_file_path):
        print(f"Error: The file 'final_redacted.pdf' does not exist in {document_dir}.")
        return pd.DataFrame()

    missed_data_details = []

    try:
        print(f"Processing redacted document: {redacted_file_path}")
        doc = fitz.open(redacted_file_path)
        num_pages = len(doc)

        for page_num in range(num_pages):
            page_index = page_num + 1
            page = doc[page_num]

            # Render page to image
            pix = page.get_pixmap(dpi=dpi_value)
            img_bytes = pix.tobytes(output='png')
            img_array = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Preprocess image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 11, 4
            )

            # Remove horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

            # Remove vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

            # Check for possible tables
            has_horizontal_lines = np.count_nonzero(remove_horizontal) > 0
            has_vertical_lines = np.count_nonzero(remove_vertical) > 0
            has_table = has_horizontal_lines and has_vertical_lines

            # Combine horizontal and vertical lines
            lines = cv2.bitwise_or(remove_horizontal, remove_vertical)

            # Subtract lines from thresholded image
            detect_regions = cv2.subtract(thresh, lines)

            # Find contours of the regions
            contours, _ = cv2.findContours(detect_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            missed_areas_on_page = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter out small regions that are likely noise
                if w > 10 and h > 10:
                    aspect_ratio = w / float(h)
                    # Exclude horizontal lines and very elongated regions
                    if aspect_ratio < 5:
                        missed_areas_on_page += 1

            # Only add rows for pages with mistakes or tables
            if missed_areas_on_page > 0 or has_table:
                missed_data_details.append({
                    'Page': page_index,
                    '# of Mistakes on Page': missed_areas_on_page,
                    'Possible Table': 'Yes' if has_table else 'No',
                })

        print(f"Finished processing redacted document: {redacted_file_path}")

    except Exception as e:
        print(f"Error processing document '{redacted_file_path}': {e}")
        return pd.DataFrame()

    # Create a DataFrame with the results
    missed_data_df = pd.DataFrame(missed_data_details)
    return missed_data_df


# %%
def ToExcel(File, FileName, spell_checked_df=None, page_mistake_counts_df=None):
    """
    Exports main data to Excel and optionally adds spell check and benchmark data as new sheets.
    
    Parameters:
        File (list or DataFrame): Main data to export.
        FileName (str): Path to save the Excel file.
        spell_checked_df (DataFrame, optional): Spell check results to add as a new sheet.
        page_mistake_counts_df (DataFrame, optional): Benchmark data to add as a new sheet.
    """
    # Convert File to DataFrame
    try:
        Excel = pd.DataFrame(File)
    except ValueError as e:
        raise ValueError(f"Error converting input data to DataFrame: {e}")

    # Desired final columns
    final_columns = ['Page', 'Combined Section', 'Text']

    # Reindex to ensure only the desired final columns exist.
    # If a column does not exist, fill it with empty strings or NaN.
    Excel = Excel.reindex(columns=final_columns)

    # Write all data to the same Excel file using ExcelWriter
    with pd.ExcelWriter(FileName, engine='openpyxl') as writer:
        # Write Main Data
        if not Excel.empty:
            Excel.to_excel(writer, sheet_name='Main', index=False)
        else:
            print("Warning: Main data is empty. No data written to 'Main' sheet.")

        # Write Spell Check Results
        if spell_checked_df is not None and not spell_checked_df.empty:
            spell_checked_df.to_excel(writer, sheet_name='Spell Check', index=False)
        else:
            print("Warning: Spell check data is empty or None. No data written to 'Spell Check' sheet.")

        # Write Benchmark Data
        if page_mistake_counts_df is not None and not page_mistake_counts_df.empty:
            page_mistake_counts_df.to_excel(writer, sheet_name='Benchmark Pages with Mistakes', index=False)
        else:
            print("Warning: Benchmark data is empty or None. No data written to 'Benchmark Pages with Mistakes' sheet.")

    print(f"Excel file created: '{FileName}'.")


# %% [markdown]
# # SPELLCHEKER

# %%
def split_word(word, spell):
    # Try splitting the word into two words at every possible position
    for i in range(1, len(word)):
        first = word[:i]
        second = word[i:]
        if first.lower() in spell and second.lower() in spell:
            return f"{first} {second}"
    return None

def perform_spellcheckPrev(combined_df):
    spell = SpellChecker()
    spell_checked_data = []
    known_acronyms = {'ANSI', 'ACI', 'NSF', 'ASTM', 'ISO', 'IEC', 'IEEE', 'FDA', 'EPA', 'min.', 'submittals'} # Add more as needed
    for index, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0], desc="Spell Checking", unit="word"):
        word = row['Text']
        page_number = row['Page']
        # Remove leading and trailing punctuation
        word = word.strip(string.punctuation)
        # Skip if word is empty after stripping punctuation
        if not word:
            continue
        # Skip if word is a single character or numeric
        if len(word) <= 1 or word.isnumeric():
            continue
        # Skip if word is a known acronym
        if word.upper() in known_acronyms:
            continue
        # Skip if word is capitalized (proper noun) or all uppercase (acronym)
        if word[0].isupper() or word.isupper():
            continue
        # Check if word is in the dictionary
        if word.lower() in spell:
            continue  # Word is correct
        else:
            # Get the most likely correction
            correction = spell.correction(word)
            # Try splitting the word
            split_correction = split_word(word, spell)
            if split_correction:
                suggested_correction = split_correction
            elif correction != word:
                suggested_correction = correction
            else:
                suggested_correction = None
            if suggested_correction and suggested_correction.lower() != word.lower():
                spell_checked_data.append({
                    'Page': page_number,
                    'Text': row['Text'],  # Use the original text including punctuation
                    'Suggested Correction': suggested_correction
                })
    # Create a dataframe with the spellchecked words
    spell_checked_df = pd.DataFrame(spell_checked_data)
    return spell_checked_df

def perform_spellcheck(combined_df):
    spell = SpellChecker()
    spell_checked_data = []
    known_acronyms = {'ANSI', 'ACI', 'NSF', 'ASTM', 'ISO', 'IEC', 'IEEE', 'FDA', 'EPA', 'min.', 'submittals'}  # Add more as needed

    for index, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0], desc="Spell Checking", unit="word"):
        original_text = row['Text']  # Preserve original text

        # Handle non-string values and skip NaN values
        if not isinstance(original_text, str):
            if pd.isna(original_text):  # Skip NaN values
                continue
            original_text = str(original_text)  # Convert non-string values to strings

        word = original_text.strip(string.punctuation)  # Strip punctuation

        # Skip irrelevant or correct words
        if not word or len(word) <= 1 or word.isnumeric() or word.upper() in known_acronyms or word[0].isupper() or word.isupper():
            continue

        if word.lower() in spell:
            continue  # Word is correct
        else:
            correction = spell.correction(word)
            split_correction = split_word(word, spell)

            if split_correction:
                suggested_correction = split_correction
            elif correction != word:
                suggested_correction = correction
            else:
                suggested_correction = None

            if suggested_correction and suggested_correction.lower() != word.lower():
                spell_checked_data.append({
                    'Page': row['Page'],
                    'Text': original_text,  # Use preserved original text
                    'Suggested Correction': suggested_correction
                })

    # Create a dataframe with the spellchecked words
    spell_checked_df = pd.DataFrame(spell_checked_data)
    return spell_checked_df

# %% [markdown]
# # -- MAIN --

# %%
def runScript(document_path, output_excel, output_pdf_path, output_redacted_pdf_name, header_y0, footer_y1,
              auto_header_footer, h_area, x0_Left, x0_Right, tolerence, line_gap_margin, percentage_linegaps,
              mergeLineclassic, is_there_a_watermark, run_tesseract):
    # Define global variables so we can debug using Jupyter Variables
    global PdfData
    global Tesseract_file
    global File0   
    global File1   
    global File2   
    global File3  
    global File4
    global File5
    global File6
    global File7
    global File8
    global File9

    # Beginning of conversion
    PdfData = CreateDataFrameFromPDF(document_path)

    # Ensure the 'Source' column exists for PdfData
    if 'Source' not in PdfData.columns:
        PdfData['Source'] = 'PDFMiner'  # Set the default source as 'PDFMiner'

    Tesseract_file = tesseract_adjustment(PdfData, output_redacted_pdf_name, document_path, is_there_a_watermark, run_tesseract)
    if run_tesseract:
        # Filter rows with 'Source' == 'Tesseract'
        tesseract_only = Tesseract_file['Source'] == 'Tesseract'
        # Perform spellcheck on the filtered DataFrame
        spell_checked_df = perform_spellcheck(Tesseract_file.loc[tesseract_only, :])
        
        # Merge corrections back into Tesseract_file
        for index, row in spell_checked_df.iterrows():
            Tesseract_file.loc[index, 'Text'] = row['Suggested Correction'] if not pd.isna(row['Suggested Correction']) else row['Text']

    page_mistake_counts_df = detect_missed_text_and_tables(document_path, dpi_value=400)
    File0 = verticalOrdering(Tesseract_file)
    File1 = removeTextonTheLeftAndright(File0, x0_Left, x0_Right)
    File2 = merge_items_with_similar_y0(File1, tolerence)

    
    File2, header_footer_lines = remove_headers_and_footers(document_path, File0, File2, header_y0, footer_y1, auto_header_footer, h_area)
    line_gap, y_gap_counts, total_unique_count = calculate_line_gap(File2, line_gap_margin, percentage_linegaps)
    
    if mergeLineclassic:
        File3 = merge_lines(File2, line_gap)
    else:
        File3 = merge_linesWithPattern(File2, line_gap, patterns, bullet_patterns, table_figure_patterns)
    
    File4 = merge_paragraphs_with_page_break(File3)
    File5 = add_Artificial_Headers(File4, patterns, bullet_patterns, table_figure_patterns)
    File6 = move_headers_to_section(File5, patterns)
    File6 = File6.reset_index(drop=True)
    File7 = sectionType(File6, patterns)
    File8 = sectionHierarchy(File7)
    File9 = create_combined_section(File8)

    printOnPDF(File2, File3, document_path, output_pdf_path, header_footer_lines, h_area)
    ToExcel(File9, output_excel, 
            spell_checked_df=File9.loc[tesseract_only, :] if 'Source' in File9.columns else None, 
            page_mistake_counts_df=page_mistake_counts_df)
    return

# %% [markdown]
# # -- UI --

# %%
## TO UPDATE FOR EACH DOCUMENT ##
### Define Input/Output Paths ###
FolderPath = r"C:\Users\Jacqueline.chen\Downloads" # Path to find the PDF document to convert to RVTM
document_name = "Pages from VN8Q_Volume_4_General_Requirements_modified.pdf" # Name of the PDF document to convert to RVTM
#### Define Input PDF, Excel output and PDF with markup output path automatically based on Folder location and input PDF name
document_path = os.path.join(FolderPath, document_name)
output_name = os.path.splitext(document_name)[0] + "_Output.xlsx"
output_pdf_name = os.path.splitext(document_name)[0] + " image.pdf"
output_redacted_pdf_name = os.path.splitext(document_name)[0] + " redacted.pdf"
output = os.path.join(FolderPath, output_name)
output_pdf_path = os.path.join(FolderPath, output_pdf_name)

### Trim document ###
auto_header_footer = False #Default value is True - Program will detect header and footer auomatically. if false use header_y0 and footer_y1 will remove headers and footer
h_area = 15 # default value is 15 - program will focus on top and botom 15% of pages to detect headers and footers (based on first page size assuming consistent page size hrough document)
header_y0 = 720 # what is above will be removed - look at the last line of header y0 and pick a value just below it
footer_y1 = 70 # what is below will be removed - look at the first line of footer y1 and pick a value just above it
x0_Left = 0 # Default value is 0. What is on the left will be removed - use only when left of the pdf needs to be cropped
x0_Right = 99999 # Default value is 99999. What is on the Right will be removed - use only when Right of the pdf needs to be cropped

## TO UPDATE FOR VERY SPECIFIC DOCUMENTS - Default values should work for most documents ##
tolerence = 1 #default value is 1 - Merge Lines with no Ycoordinate gaps within the tolerence
line_gap_margin = 1 #Default value is 1 - Value to adjust based on PDF, it is added to line gap to include the highest values of line gap ideally if all line gaps were uniform throught the document this would be 0
percentage_linegaps = 30 #Default Value is 30 - % estimated line gaps compared to other Y gaps in PDF (for example Paragraph gaps)
mergeLineclassic = True #Default value is True - It means That paragraph y Gap is always bigger than line y Gaps
is_there_a_watermark = False # True or False

## TO UPDATE ONE TIME ##
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" #  #Update path with your Local Tesseract installation
run_tesseract = True
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Jacqueline.Chen\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# RUN SCRIPT #
runScript(document_path, output, output_pdf_path, output_redacted_pdf_name, header_y0, footer_y1, auto_header_footer, h_area, x0_Left, x0_Right, tolerence, line_gap_margin, percentage_linegaps, mergeLineclassic, is_there_a_watermark, run_tesseract)

# %%
def runScriptPreview(
        document_path,  # OK
        header_y0,      # OK
        footer_y1,      # OK
        x0_Left,        # OK
        x0_Right, 
        tolerence,      # OK
        line_gap_margin, # OK
        percentage_linegaps, # OK
        auto_header_footer,  # OK
        h_area):
    # Define global variables for debugging (not generally recommended, but kept for consistency)
    global PdfData, Tesseract_file, File0
    global y_gap_counts
    # Let's define a total number of steps for progress. This is arbitrary and depends on your processing:   
    try:
        # Start processing
        PdfData = CreateDataFrameFromPDF(document_path)

        if 'Source' not in PdfData.columns:
            PdfData['Source'] = 'PDFMiner'

        File0 = verticalOrdering(PdfData)
        File1 = removeTextonTheLeftAndright(File0, x0_Left, x0_Right)
        File2 = merge_items_with_similar_y0(File1, tolerence)
        File2, header_footer_lines = remove_headers_and_footers(document_path, File0, File2, header_y0, footer_y1, auto_header_footer, h_area)
        y_gap_counts = pd.DataFrame()
        line_gap, y_gap_counts, total_unique_count = calculate_line_gap(File2, line_gap_margin, percentage_linegaps)
        return True
    except Exception as e:
        print(f"Error in runScriptPreview: {e}")
        return False



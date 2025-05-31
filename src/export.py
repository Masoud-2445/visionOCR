import cv2
from reportlab.pdfgen import canvas
import os
import tempfile
from reportlab.lib.colors import Color
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.shared import OxmlElement
from typing import Dict
from docx.shared import Pt, Inches



def overlay_ocr_to_pdf(image_np, export_data, output_pdf_path):
    """
    Create a PDF with the input image as background and overlay text from OCR data
    with a semi-transparent background, similar to Google Lens Translate.
    
    Args:
        image_np (numpy.ndarray): OpenCV image array
        export_data (dict): Dictionary from OCREngine.prepare_for_export() method
                           Contains 'lines' with text items and 'image_size'
        output_pdf_path (str): Path to save the output PDF
    """
    height, width = image_np.shape[:2]

    width_pt = width * 0.75
    height_pt = height * 0.75

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img_file:
        temp_img_path = temp_img_file.name
        cv2.imwrite(temp_img_path, image_np)
    
    try:
        c = canvas.Canvas(output_pdf_path, pagesize=(width_pt, height_pt))
        
        c.drawImage(temp_img_path, 0, 0, width=width_pt, height=height_pt)

        for line in export_data["lines"]:
            line_text = line["text"]

            if not line_text or line_text.isspace():
                continue

            line_items = line["items"]
            if not line_items:
                continue

            min_left = min(item["bbox"][0] for item in line_items)
            min_top = min(item["bbox"][1] for item in line_items)
            max_right = max(item["bbox"][0] + item["bbox"][2] for item in line_items)
            max_bottom = max(item["bbox"][1] + item["bbox"][3] for item in line_items)

            left_px = min_left
            top_px = min_top
            width_px = max_right - min_left
            height_px = max_bottom - min_top

            left = left_px * 0.75
            box_width = width_px * 0.75
            box_height = height_px * 0.75
            top = height_pt - (top_px * 0.75) - box_height
            

            font_size = max(10, min(box_height * 0.7, 72))  # Min 10pt, max 72pt
            c.setFont("Helvetica", font_size)
                
            padding_x = box_width * 0.03
            padding_y = box_height * 0.1

            ext_left = left - padding_x
            ext_top = top - padding_y
            ext_width = box_width + (padding_x * 2)
            ext_height = box_height + (padding_y * 2)
            
            c.saveState()

            c.setFillColor(Color(1, 1, 1, alpha=0.7))
            c.setStrokeColor(Color(0.8, 0.8, 0.8, alpha=0.8))
            c.setLineWidth(0.5)
            
            corner_radius = min(5, ext_height/4)  # Reasonable corner radius
            c.roundRect(ext_left, ext_top, ext_width, ext_height, 
                       corner_radius, stroke=1, fill=1)

            c.setFillColor(Color(0, 0, 0, alpha=1))

            text_x = left + padding_x  # Add some padding from the left edge
            text_y = top + box_height * 0.3  # Adjust baseline position

            c.drawString(text_x, text_y, line_text)

            c.restoreState()

        c.save()   
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
    return output_pdf_path

def export_ocr_to_word(export_data: Dict, output_docx_path: str, rtl_support: bool = False) -> None:
    """
    Exports OCR result data into a Microsoft Word (.docx) document.

    Args:
        export_data (Dict): A dictionary containing OCR result data with line and word bounding boxes.
        output_docx_path (str): Path to save the generated .docx file.
        rtl_support (bool): Whether to apply right-to-left formatting (e.g., for Persian/Arabic text).
    """
    try:
        doc = Document()

        # Set narrow margins
        for section in doc.sections:
            section.top_margin = Inches(0.5)
            section.bottom_margin = Inches(0.5)
            section.left_margin = Inches(0.5)
            section.right_margin = Inches(0.5)

        avg_font_size = 11
        print(f"Using consistent font size: {avg_font_size}pt")

        lines = export_data.get("lines", [])

        # Sort lines by top bounding box coordinate if available
        if lines:
            lines_with_top = []
            for line in lines:
                if line.get("items") and line["items"][0].get("bbox"):
                    top_pos = line["items"][0]["bbox"][1]
                    lines_with_top.append((top_pos, line))
                else:
                    lines_with_top.append((0, line))  # fallback
            lines_with_top.sort(key=lambda x: x[0])
            sorted_lines = [line for _, line in lines_with_top]
        else:
            sorted_lines = lines

        for line in sorted_lines:
            line_text = line.get("text", "")
            items = line.get("items", [])

            if not line_text.strip():
                continue

            paragraph = doc.add_paragraph()

            if rtl_support:
                paragraph.paragraph_format.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT
                pPr = paragraph._element.get_or_add_pPr()
                bidi = OxmlElement("w:bidi")
                pPr.append(bidi)

            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(1)
            paragraph.paragraph_format.line_spacing = 1.1

            if items:
                for idx, item in enumerate(items):
                    item_text = item.get("text", "")
                    if not item_text.strip():
                        continue

                    run = paragraph.add_run(item_text)
                    run.font.size = Pt(avg_font_size)
                    run.font.bold = False
                    run.font.italic = False
                    run.font.underline = False

                    if idx < len(items) - 1:
                        space = paragraph.add_run(" ")
                        space.font.size = Pt(avg_font_size)
                        space.font.bold = False
            else:
                run = paragraph.add_run(line_text)
                run.font.size = Pt(avg_font_size)
                run.font.bold = False
                run.font.italic = False
                run.font.underline = False

        doc.save(output_docx_path)
        print(f"Word document saved to: {output_docx_path}")

    except Exception as e:
        print("Error exporting OCR result to Word:")
        print(f"Exception: {e}")

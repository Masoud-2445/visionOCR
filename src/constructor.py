from docx import Document
from docx.shared import Pt, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import cv2
import os
import uuid
from docx.oxml.ns import qn
from docx.oxml import OxmlElement



def save_temp_image(cv_image):
    """Saves an OpenCV image to a temporary file and returns the path."""
    temp_folder = "temp_images"
    os.makedirs(temp_folder, exist_ok=True)
    file_path = os.path.join(temp_folder, f"{uuid.uuid4()}.png")
    cv2.imwrite(file_path, cv_image)
    return file_path



def text_constructor(container, element, space_before_pt = None):
    try :
        text = container.add_paragraph()

        if space_before_pt:
            text.paragraph_format.space_before = Pt(space_before_pt)

        props = element.get("font_properties", {})
        content = element.get("content", "")

        lines = content.split('\n')
        
        alignment_str = element.get("alignment", "left")

        if alignment_str == "center":
            text.alignment = WD_ALIGN_PARAGRAPH.CENTER

        elif alignment_str == "right":
            text.alignment = WD_ALIGN_PARAGRAPH.RIGHT

        else:
            text.alignment = WD_ALIGN_PARAGRAPH.LEFT


        is_rtl = any('\u0600' <= char <= '\u06FF' for char in content)
        if is_rtl:
            pPr = text._p.get_or_add_pPr()
            bidi = OxmlElement('w:bidi')
            bidi.set(qn('w:val'), '1')
            pPr.append(bidi)
        lines = content.split('\n')

        for i, line in enumerate(lines) :
            run = text.add_run(line)
        
            if props.get("weight") == "bold":
                run.bold = True

            font_size_str = props.get("size", "medium")

            if font_size_str == "large":
                run.font.size = Pt(16)

            elif font_size_str == "small":
                run.font.size = Pt(10)

            else:
                run.font.size = Pt(12)

            if i < len(lines) - 1 : 
                run.add_break()

    except Exception as e :
        print(f"Error in text_constructor: {e}")


def visual_constructor(container, element, cv_image, page_width_inches, page_height_inches, space_before_pt = None):
    tmp_path = None
    try :
        img = cv_image
        abs_box = element["absolute_bbox"]
        rel_box = element["relative_bbox"]

        img_h, img_w = cv_image.shape[:2]
        x1 = max(0, abs_box["x_start"])
        y1 = max(0, abs_box["y_start"])
        x2 = min(img_w, abs_box["x_end"])
        y2 = min(img_h, abs_box["y_end"])


        cropped = img[y1:y2, x1:x2]

        tmp_path = save_temp_image(cropped)


        text = container.add_paragraph()
        if space_before_pt:
            text.paragraph_format.space_before = Pt(space_before_pt)
        text.paragraph_format.space_after = Pt(0)


        alignment_str = element.get("alignment", "left")
        if alignment_str == "center":
            text.alignment = WD_ALIGN_PARAGRAPH.CENTER

        elif alignment_str == "right":
            text.alignment = WD_ALIGN_PARAGRAPH.RIGHT

        else:
            text.alignment = WD_ALIGN_PARAGRAPH.LEFT

        relative_width = rel_box["x_end"] - rel_box["x_start"]
        image_width_inches = page_width_inches * relative_width

        relative_height = rel_box["y_end"] - rel_box["y_start"]
        image_height_inches = page_height_inches * relative_height


        run = text.add_run()
        run.add_picture(tmp_path, width=Inches(image_width_inches), height=Inches(image_height_inches))


    except Exception as e :
        print(f"Error in visual_construct : {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)



def table_constructor(container, element, page_width_inches, space_before_pt = None):
    try :

        if space_before_pt:
            text = container.add_paragraph()
            text.paragraph_format.space_before = Pt(space_before_pt)
            text.paragraph_format.space_after = Pt(0)

        table_data = element.get("table_structure", [])

        if not table_data or not table_data[0]:
            print("table with no structure: Skipping.")
            return
        
        rows = len(table_data)
        cols = len(table_data[0]) if rows > 0 else 0
    
        table = container.add_table(rows=rows, cols=cols)
        table.style = "Table Grid"
        alignment_str = element.get("alignment", "left")


        if alignment_str == "center":
            table.alignment = WD_TABLE_ALIGNMENT.CENTER

        elif alignment_str == "right":
            table.alignment = WD_TABLE_ALIGNMENT.RIGHT

        else:
            table.alignment = WD_TABLE_ALIGNMENT.LEFT

        
        rel_box = element.get("relative_bbox")
        if rel_box:
            
            relative_width = rel_box["x_end"] - rel_box["x_start"]
            total_table_width = Inches(page_width_inches * relative_width)
            table.width = total_table_width
            table.autofit = False 
            try :
                table.allow_autofit = False
            except Exception :
                pass

            first_row_cells = table_data[0]
            for i, cell_data in enumerate(first_row_cells):
                cell_rel_box = cell_data.get("relative_bbox")
                if cell_rel_box and relative_width > 0:
                    cell_relative_width = cell_rel_box["x_end"] - cell_rel_box["x_start"]
                    table.columns[i].width = Emu(int(total_table_width.emu * (cell_relative_width / relative_width)))
            
    
        for r, row in enumerate(table_data):
            for c, cell in enumerate(row):
                table.cell(r, c).text = cell.get("text", "")

    except Exception as e :
        print(f"failed to make a table construct : {e}")




def add_horizontal_group(doc, group, cv_image, usable_page_width_inches, usable_page_height_inches, space_before_pt = None):
    if space_before_pt:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(space_before_pt)
        p.paragraph_format.space_after = Pt(0)

    group.sort(key=lambda e: e['absolute_bbox']['x_start'])
    
    num_columns = len(group)
    table = doc.add_table(rows=1, cols=num_columns)

    for row in table.rows:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = OxmlElement('w:tcBorders')
            for border_name in ['top','left','bottom','right','insideH','insideV']:
                b = OxmlElement(f'w:{border_name}')
                b.set(qn('w:val'), 'nil')
                tcBorders.append(b)
            tcPr.append(tcBorders)

    for i, element in enumerate(group):
        cell = table.cell(0, i)
        element_type = element.get("type")
        
        if element_type in ["header", "paragraph", "footer"]:
            text_constructor(cell, element)

        elif element_type in ["stamp", "signature", "person_headshot", "signed_headshot", "signed_stamp"]:
            visual_constructor(cell, element, cv_image, usable_page_width_inches, usable_page_height_inches)

        elif element_type == "table":
             table_constructor(cell, element, usable_page_width_inches, 0)



def docx_constructor(layout_data, image_target, output_path):

    try:
        doc = Document()
        
        section = doc.sections[0]
        usable_page_width = section.page_width.inches - section.left_margin.inches - section.right_margin.inches
        usable_page_height = section.page_height.inches - section.top_margin.inches - section.bottom_margin.inches

        elements = layout_data.get("elements", [])
        if not elements:
            print("No elements to process.")
            doc.save(output_path)
            return

        elements.sort(key=lambda e: (e['absolute_bbox']['y_start'], e['absolute_bbox']['x_start']))
        print("starting document construction")

        last_element_y_end = 0
        inc = 0

        while inc < len(elements):
            current_element = elements[inc]

            gap_points = 0
            if last_element_y_end > 0:
                gap_pixels = current_element['absolute_bbox']['y_start'] - last_element_y_end
                
                if gap_pixels > 20:
                    gap_points = (gap_pixels / 96.0) * 72.0
            
            
            horizontal_group = [current_element]
            next_element = inc + 1

            while next_element < len(elements):
                y1, y2 = current_element['absolute_bbox']['y_start'], current_element['absolute_bbox']['y_end']
                y3, y4 = elements[next_element]['absolute_bbox']['y_start'], elements[next_element]['absolute_bbox']['y_end']
                
                overlap = max(0, min(y2, y4) - max(y1, y3))
                min_height = min(y2 - y1, y4 - y3)
                
                if min_height > 0 and (overlap / min_height) > 0.35:
                    horizontal_group.append(elements[next_element])
                    next_element += 1
                else:
                    break

            if len(horizontal_group) > 1:
                print(f"  - Found a horizontal group with {len(horizontal_group)} elements.")

                add_horizontal_group(doc, horizontal_group, image_target, usable_page_width, usable_page_height, gap_points)
                
                last_element_y_end = max(e['absolute_bbox']['y_end'] for e in horizontal_group)
                inc += len(horizontal_group)
            else:
                
                element_type = current_element.get("type")

                if element_type in ["header", "paragraph", "footer"]:
                    text_constructor(doc, current_element, gap_points)

                elif element_type == "table":
                    table_constructor(doc, current_element, usable_page_width, gap_points)

                elif element_type in ["stamp", "signature", "person_headshot", "signed_headshot", "signed_stamp"]:
                    visual_constructor(doc, current_element, image_target, usable_page_width, usable_page_height, gap_points)
                
                
                last_element_y_end = current_element['absolute_bbox']['y_end']
                inc += 1

        doc.save(output_path)
        print(f"Document with precise spacing successfully saved in: {output_path}")

    except Exception as e:
        print(f"failed to make document construct : {e}")
        return None

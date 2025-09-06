import json

# this module use for enriching api output
def post_process_layout(output_json_path: str):

    with open(output_json_path, "r", encoding="utf-8") as f:
        api_output = json.load(f)

    image_width = api_output["analyzed_image_size"]["width"]
    image_height = api_output["analyzed_image_size"]["height"]


    elements = api_output.get("elements", [])

    for element in elements:
        abs_box = element.get("absolute_bbox")
        if abs_box:
            element["relative_bbox"] = {
                "x_start": abs_box["x_start"] / image_width,
                "x_end": abs_box["x_end"] / image_width,
                "y_start": abs_box["y_start"] / image_height,
                "y_end": abs_box["y_end"] / image_height,
            }


    for el in elements:
        if el.get("type") in ["visual_element", "table"] and "alignment" not in el:
            abs_box = el["absolute_bbox"]
            center_x = (abs_box["x_start"] + abs_box["x_end"]) / 2
            if center_x < image_width * 0.33:
                el["alignment"] = "left"
            elif center_x > image_width * 0.66:
                el["alignment"] = "right"
            else:
                el["alignment"] = "center"

    elements_sorted = sorted(elements, key=lambda e: (e["absolute_bbox"]["y_start"], e["absolute_bbox"]["x_start"]))

    for idx, el in enumerate(elements_sorted, start=1):
        el["reading_order_index"] = idx




    for el in elements:
        if el.get("type") == "table" and "table_content" in el:
            table_content = el["table_content"]
            table_structure = []
            for row in table_content:
                row_cells = [{"text": cell} for cell in row]
                table_structure.append(row_cells)
            el["table_structure"] = table_structure

    api_output["elements"] = elements_sorted

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(api_output, f, ensure_ascii=False, indent=4)

    print(f"[layout_postprocess] Updated JSON saved to: {output_json_path}")
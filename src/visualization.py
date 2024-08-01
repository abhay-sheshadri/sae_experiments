import html
import matplotlib.pyplot as plt


def interpolate_color(start_color, end_color, factor):
    """
    Interpolate between two colors.
    
    Parameters:
    start_color (tuple): RGB values for the starting color (light orange)
    end_color (tuple): RGB values for the ending color (dark orange)
    factor (float): A value between 0 and 1 indicating the interpolation factor
    
    Returns:
    tuple: Interpolated RGB color
    """
    return tuple(int(start + factor * (end - start)) for start, end in zip(start_color, end_color))


def generate_highlighted_html(tokens, acts):
    """
    Generates HTML for a single example with highlighted text.

    Parameters:
    tokens (list): A list of string tokens representing the text.
    acts (list): A list of float values representing the activation (highlight intensity) for each token.

    Returns:
    str: HTML string with highlighted text.

    Raises:
    ValueError: If the number of tokens doesn't match the number of activation values.
    """
    if len(tokens) != len(acts):
        raise ValueError("The number of tokens and activations must match.")

    html_content = "<div class='text-content'>"
    for token, act in zip(tokens, acts):
        if act > 0:
            # Normalize activation, max at 1
            factor = min(act / 10, 1)
            
            # Interpolate color
            light_orange = (255, 237, 160)  # RGB for very light orange
            dark_orange = (240, 134, 0)    # RGB for dark, saturated orange
            color = interpolate_color(light_orange, dark_orange, factor)
            
            html_content += f'<span class="highlight" style="background-color: rgb{color};">{html.escape(token)}</span>'
        else:
            html_content += html.escape(token)
    html_content += "</div>"

    return html_content


def generate_categorized_examples(categorized_examples):
    """
    Generates complete HTML content for multiple categories of examples.

    Parameters:
    categorized_examples (dict): A dictionary where keys are category names (strings) and 
                                 values are lists of tuples. Each tuple contains two lists: 
                                 tokens and their corresponding activation values.

    Returns:
    str: Complete HTML string with all categories and their examples, including CSS styling.

    This function creates a structured HTML document with:
    - Overall styling for the document
    - Sections for each category
    - Individual examples within each category, using the generate_highlighted_html function
    """
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
            line-height: 1.3;
            margin: 0;
            padding: 10px;
        }
        .highlight {
            border-radius: 2px;
            padding: 0 1px;
        }
        .category {
            margin-bottom: 20px;
        }
        .category-label {
            font-weight: bold;
            font-size: 16px;
            color: #333;
            margin-bottom: 10px;
            border-bottom: 2px solid #333;
            padding-bottom: 5px;
        }
        .example {
            margin-bottom: 10px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        .example-label {
            font-weight: bold;
            font-size: 12px;
            color: #666;
            margin-bottom: 2px;
        }
        .text-content {
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
    </style>
    """

    html_content = "<div class='examples-container'>"
    
    for category, examples in categorized_examples.items():
        
        html_content += f'<div class="category"><div class="category-label">{html.escape(category)}</div>'
        for i, (tokens, acts) in enumerate(examples, 1):
            html_content += f'<div class="example"><div class="example-label">Example {i}:</div>'
            html_content += generate_highlighted_html(tokens, acts)
            html_content += '</div>'
        html_content += '</div>'

    html_content += "</div>"
    full_html = f"<!DOCTYPE html><html><head>{css}</head><body>{html_content}</body></html>"
    return full_html

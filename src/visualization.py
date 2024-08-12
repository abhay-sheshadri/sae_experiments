import html
import matplotlib.pyplot as plt


def interpolate_color(start_color, end_color, factor):
    """
    Interpolate between two colors.
    """
    # For each pair of corresponding color values in start_color and end_color:
    # 1. Calculate the difference: (end - start)
    # 2. Multiply this difference by the factor
    # 3. Add the result to the start value
    # 4. Convert the result to an integer
    # Return the resulting color as a tuple
    return tuple(int(start + factor * (end - start)) for start, end in zip(start_color, end_color))


def light_mode(html):
    # Define HTML wrapper for light mode display
    light_mode_wrapper = """
    <div style="background-color: white; color: black; padding: 20px;">
        {}
    </div>
    """
    # Generate the categorized examples HTML and wrap it in the light mode div
    return light_mode_wrapper.format(html)


def generate_highlighted_html(tokens, acts):
    """
    Generates HTML for a single example with highlighted text.
    """
    # Make sure there are an equal number of tokens and activations
    if len(tokens) != len(acts):
        raise ValueError("The number of tokens and activations must match.")
    # Generate html content
    html_content = "<div class='text-content'>"
    for token, act in zip(tokens, acts):
        # Highlight token if act > 0
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


def feature_centric_view(feature, short=False):
    if short:
        # Just add the top 5 max activating examples
        max_activation_examples = feature.get_max_activating(5)
        # Create a dictionary with these top activations
        display_dict = {"Top Activations": [ex.get_tokens_feature_lists(feature) for ex in max_activation_examples]}
    else:
        # Get the top 15 examples that maximally activate this feature
        max_activation_examples = feature.get_max_activating(15)
        # Create a dictionary with these top activations
        display_dict = {"Top Activations": [ex.get_tokens_feature_lists(feature) for ex in max_activation_examples]}
        # Get 8 quantiles with 5 examples each
        quantiles = feature.get_quantiles(8, 5)
        # Iterate through the quantiles in reverse order (highest to lowest)
        for i, (lower, upper) in enumerate(list(quantiles)[::-1]):
            # Get examples for this quantile
            examples = quantiles[(lower, upper)]
            # Add examples for this quantile to the display dictionary
            # The key is formatted as "Interval {i} - (lower_bound, upper_bound)"
            display_dict[f"Interval {i} - ({lower:.2f}, {upper:.2f})"] = [ex.get_tokens_feature_lists(feature) for ex in examples]    
    return light_mode(generate_categorized_examples(display_dict))
    # Define HTML wrapper for light mode display
    light_mode_wrapper = """
    <div style="background-color: white; color: black; padding: 20px;">
        {}
    </div>
    """
    # Generate the categorized examples HTML and wrap it in the light mode div
    return light_mode_wrapper.format()

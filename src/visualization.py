import html

import matplotlib.pyplot as plt


def _interpolate_color(start_color, end_color, factor):
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


def _light_mode(html):
    # Define HTML wrapper for light mode display
    light_mode_wrapper = """
    <div style="background-color: white; color: black; padding: 20px;">
        {}
    </div>
    """
    # Generate the categorized examples HTML and wrap it in the light mode div
    return light_mode_wrapper.format(html)


def _generate_highlighted_html(tokens, acts, use_orange_highlight=True):
    """
    Generates HTML for a single example with highlighted text.
    """
    # Make sure there are an equal number of tokens and activations
    if len(tokens) != len(acts):
        raise ValueError("The number of tokens and activations must match.")
    # Generate html content
    html_content = "<div class='text-content'>"
    for token, act in zip(tokens, acts):
        # Highlight token if act != 0
        if act != 0:
            # Normalize activation, max at 1
            factor = min(abs(act) / 10, 1)
            # Interpolate color
            if use_orange_highlight:
                light_orange = (255, 237, 160)  # RGB for very light orange
                dark_orange = (240, 134, 0)    # RGB for dark, saturated orange
                color = _interpolate_color(light_orange, dark_orange, factor)
            else:
                if act > 0:
                    color = _interpolate_color((255, 255, 255), (0, 0, 255), factor)  # White to Blue
                else:
                    color = _interpolate_color((255, 255, 255), (255, 0, 0), factor)  # White to Red
            html_content += f'<span class="highlight" style="background-color: rgb{color};">{html.escape(token)}</span>'
        else:
            html_content += html.escape(token)
    html_content += "</div>"
    return html_content


def _generate_categorized_examples(categorized_examples):
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
            html_content += _generate_highlighted_html(tokens, acts)
            html_content += '</div>'
        html_content += '</div>'

    html_content += "</div>"
    full_html = f"<!DOCTYPE html><html><head>{css}</head><body>{html_content}</body></html>"
    return full_html


def _generate_logits_table(negative_logits, positive_logits):
    """
    Generate HTML content for a table of negative and positive logits
    """
    css = """
    <style>
        .logits-container {
            display: flex;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 20px auto;
        }
        .logits-column {
            flex: 1;
            margin: 0 5px;
        }
        .logits-column h2 {
            background-color: #f0f0f0;
            margin: 0;
            padding: 10px;
            border-radius: 5px 5px 0 0;
            font-size: 16px;
            font-weight: normal;
        }
        .logits-list {
            list-style-type: none;
            padding: 0;
            margin: 0;
        }
        .logits-list li {
            display: flex;
            justify-content: space-between;
            padding: 5px 10px;
        }
        .negative .logits-list li:nth-child(odd) {
            background-color: #ffe6e6;
        }
        .negative .logits-list li:nth-child(even) {
            background-color: #ffcccc;
        }
        .positive .logits-list li:nth-child(odd) {
            background-color: #e6e6ff;
        }
        .positive .logits-list li:nth-child(even) {
            background-color: #ccccff;
        }
        .token {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: rgba(0, 0, 0, 0.1);
            padding: 2px 4px;
            border-radius: 3px;
            color: #333;
            font-weight: bold;  /* Make token text bold */
        }
        .value {
            font-weight: normal;
        }
    </style>
    """
    html_content = f"{css}<div class='logits-container'>"
    # Negative logits column
    html_content += """
    <div class='logits-column negative'>
        <h2>NEGATIVE LOGITS</h2>
        <ul class='logits-list'>
    """
    for token, value in negative_logits:
        html_content += f"<li><span class='token'>{html.escape(token)}</span><span class='value'>{value:.5f}</span></li>"
    html_content += "</ul></div>"
    # Positive logits column
    html_content += """
    <div class='logits-column positive'>
        <h2>POSITIVE LOGITS</h2>
        <ul class='logits-list'>
    """
    for token, value in positive_logits:
        html_content += f"<li><span class='token'>{html.escape(token)}</span><span class='value'>{value:.5f}</span></li>"
    html_content += "</ul></div>"

    html_content += "</div>"
    return html_content


def _combine_html_contents(*html_contents, title="Combined View"):
    """
    Combine multiple HTML contents into a single HTML document with a dropdown to select which content to display.
    """
    # Start the HTML document
    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #2c3e50;
                padding-bottom: 10px;
                font-size: 24px;  /* Reduced from the default (usually around 32px) */
                margin-top: 10px;  /* Reduced top margin */
                margin-bottom: 15px;  /* Adjusted bottom margin */
            }}
            #content-selector {{
                margin-bottom: 20px;
                padding: 10px;
                font-size: 16px;
            }}
            .content {{
                display: none;
            }}
            .content.active {{
                display: block;
            }}
        </style>
        <script>
            function showContent(id) {{
                var contents = document.getElementsByClassName('content');
                for (var i = 0; i < contents.length; i++) {{
                    contents[i].classList.remove('active');
                }}
                document.getElementById(id).classList.add('active');
            }}
            
            // Function to show the first content by default
            window.onload = function() {{
                var firstContentId = document.getElementById('content-selector').value;
                showContent(firstContentId);
            }};
        </script>
    </head>
    <body>
        <h1>{title}</h1>
        <select id="content-selector" onchange="showContent(this.value)">
    """
    # Add dropdown options for each section
    for i, (section_title, _) in enumerate(html_contents):
        selected = "selected" if i == 0 else ""
        combined_html += f'<option value="content_{i}" {selected}>{section_title}</option>'
    combined_html += """
        </select>
    """
    # Add the HTML content for each section
    for i, (section_title, content) in enumerate(html_contents):
        active_class = "active" if i == 0 else ""
        combined_html += f"""
        <div id="content_{i}" class="content {active_class}">
            <h2>{section_title}</h2>
            {content}
        </div>
        """
    # Close the HTML document
    combined_html += """
    </body>
    </html>
    """
    return combined_html


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
    # Generate HTML content for the categorized examples
    categorized_examples_html = _generate_categorized_examples(display_dict)
    # Get the top and bottom 8 logits
    top_logits, bottom_logits = feature.get_logits(8)
    # Generate HTML content for the logits table
    logits_table_html = _generate_logits_table(bottom_logits, top_logits)
    # Add the logits table as a seperate table in the HTML content accessible via a dropdown
    combined_html = _combine_html_contents(
            ("Categorized Examples", categorized_examples_html),
            ("Logits Analysis", logits_table_html),
            title=f"Feature-Centric View - Feature {feature.feature_id}, Hook Point {feature.hook_name}"
        )
    # Return the combined HTML content
    return _light_mode(combined_html)


def prompt_centric_view(examples, feature):
    # Generate a view for a single example or a list of examples and a feature.
    if not isinstance(examples, list):
        examples = [examples]
    html_content = f"<h2>Prompt-Centric View - Feature {feature.feature_id}, Hook Point {feature.hook_name}</h2>"
    html_content += "<div class='examples-container'>"
    # Iterate through each example, and create div
    for i, example in enumerate(examples, 1):
        acts = example.get_feature_activation(feature)
        html_content += f'<div class="example"><div class="example-label">Example {i}:</div>'
        html_content += _generate_highlighted_html(example.str_tokens, acts.tolist())
        html_content += '</div>'
    html_content += "</div>"
    return _light_mode(html_content)


def prompt_centric_view_direction(examples, encoder, hook_name, direction):
    # Generate a view for a single example or a list of examples and a direction.
    if not isinstance(examples, list):
        examples = [examples]
    html_content = f"<h2>Prompt-Centric View - Hook Point {hook_name}</h2>"
    html_content += "<div class='examples-container'>"
    # Iterate through each example, and create div
    for i, example in enumerate(examples, 1):
        str_tokens, acts = example.get_tokens_direction_scores(encoder, direction, hook_name)
        html_content += f'<div class="example"><div class="example-label">Example {i}:</div>'
        html_content += _generate_highlighted_html(str_tokens, acts, use_orange_highlight=False)
        html_content += '</div>'
    html_content += "</div>"
    return _light_mode(html_content)    
    
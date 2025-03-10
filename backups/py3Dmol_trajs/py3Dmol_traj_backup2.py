import os
import re
import py3Dmol
from IPython.display import HTML
import base64

def parse_pdb_models(pdb_file):
    """
    Parse a PDB file into separate models.
    
    Args:
        pdb_file (str): Path to the PDB file
        
    Returns:
        list: A list of models, where each model is a list of PDB lines
    """
    models = []
    current_model = []
    model_titles = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('TITLE'):
                # Store the title for each model
                model_titles.append(line.strip()[6:].strip())
            
            if line.startswith('ENDMDL'):
                current_model.append(line)
                models.append('\n'.join(current_model))
                current_model = []
            elif line.startswith('MODEL') and current_model:
                # If we find a new MODEL line but haven't closed the previous one,
                # implicitly close the previous model
                models.append('\n'.join(current_model))
                current_model = [line]
            else:
                current_model.append(line)
    
    # Add the last model if it exists and wasn't closed with ENDMDL
    if current_model:
        models.append('\n'.join(current_model))
    
    # If no titles were found, generate default ones
    if len(model_titles) < len(models):
        model_titles.extend([f"Model {i+1}" for i in range(len(model_titles), len(models))])
    
    return models, model_titles

def create_py3dmol_views(models):
    """
    Create py3Dmol views for each model in the trajectory.
    
    Args:
        models (list): List of model strings
        
    Returns:
        list: List of py3Dmol view HTML strings
    """
    view_htmls = []
    
    for i, model_str in enumerate(models):
        # Create a new py3Dmol view
        view = py3Dmol.view(width=600, height=400)
        view.addModel(model_str, 'pdb')
        
        # Set style to show backbone with AlphaFold pLDDT coloring
        view.setStyle({'cartoon': {'color': 'spectrum', 'colorscheme': {
            'prop': 'b',
            'gradient': 'roygb',
            'min': 0,
            'max': 100
        }}})
        
        # Set default orientation
        view.zoomTo()
        view.setBackgroundColor('white')
        
        # Get HTML representation
        html = view._make_html()
        
        # Fix style issues by adjusting CSS
        html = html.replace('<style>', '<style>.viewer_3Dmoljs{position:relative;width:600px;height:400px;} .mol-container{width:100%;height:100%;position:relative;} ')
        
        # Make view initially hidden except for the first one
        if i > 0:
            html = html.replace('<div', '<div style="display:none;"', 1)
        
        # Add a data-index attribute to help with navigation
        html = html.replace('<div', f'<div data-model-index="{i}"', 1)
        
        view_htmls.append(html)
    
    return view_htmls

def generate_html_slider_page(view_htmls, model_titles):
    """
    Generate an HTML page with the py3Dmol views and a slider control.
    
    Args:
        view_htmls (list): List of py3Dmol view HTML strings
        model_titles (list): List of model titles
        
    Returns:
        str: Complete HTML page
    """
    # Properly handle JSON escaping for the model titles
    import json
    json_model_titles = json.dumps(model_titles)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDB Trajectory Viewer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .container {{
                width: 650px;
                margin: 0 auto;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                padding: 20px;
                border-radius: 8px;
            }}
            .viewer-container {{
                position: relative;
                width: 600px;
                height: 400px;
                margin: 20px auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .controls {{
                display: flex;
                justify-content: center;
                align-items: center;
                margin-top: 10px;
                gap: 10px;
            }}
            .slider-container {{
                width: 100%;
                padding: 10px 0;
            }}
            button {{
                padding: 8px 12px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            button:disabled {{
                background-color: #cccccc;
                cursor: not-allowed;
            }}
            .slider {{
                width: 100%;
            }}
            .model-info {{
                text-align: center;
                margin: 10px 0;
                font-weight: bold;
            }}
            .speed-control {{
                display: flex;
                align-items: center;
                gap: 5px;
                margin-left: 20px;
            }}
        </style>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>PDB Trajectory Viewer</h1>
            <div class="viewer-container" id="viewer-container">
                {"".join(view_htmls)}
            </div>
            <div class="model-info" id="model-info">
                Model 1 of {len(view_htmls)}: <span id="model-title">{model_titles[0]}</span>
            </div>
            <div class="slider-container">
                <input type="range" min="1" max="{len(view_htmls)}" value="1" class="slider" id="model-slider">
            </div>
            <div class="controls">
                <button id="prev-btn">&lt; Previous</button>
                <button id="play-btn">Play</button>
                <button id="pause-btn" disabled>Pause</button>
                <button id="next-btn">Next &gt;</button>
                <button id="reset-btn">Reset</button>
                <div class="speed-control">
                    <label for="speed-control">Speed:</label>
                    <select id="speed-control">
                        <option value="2000">Slow</option>
                        <option value="1000" selected>Normal</option>
                        <option value="500">Fast</option>
                        <option value="250">Very Fast</option>
                    </select>
                </div>
            </div>
        </div>

        <script>
            // Get elements
            const viewerContainer = document.getElementById('viewer-container');
            const modelViews = viewerContainer.querySelectorAll('div[data-model-index]');
            const modelSlider = document.getElementById('model-slider');
            const modelInfo = document.getElementById('model-info');
            const modelTitle = document.getElementById('model-title');
            const playBtn = document.getElementById('play-btn');
            const pauseBtn = document.getElementById('pause-btn');
            const prevBtn = document.getElementById('prev-btn');
            const nextBtn = document.getElementById('next-btn');
            const resetBtn = document.getElementById('reset-btn');
            const speedControl = document.getElementById('speed-control');
            
            // Model titles
            const modelTitles = {json_model_titles};
            
            let currentModelIndex = 0;
            let isPlaying = false;
            let playInterval;
            
            // Function to show a specific model
            function showModel(index) {{
                // Hide all models
                modelViews.forEach(view => {{
                    view.style.display = 'none';
                }});
                
                // Show the selected model
                modelViews[index].style.display = 'block';
                
                // Update slider value
                modelSlider.value = index + 1;
                
                // Update model info
                modelInfo.innerHTML = `Model ${{index + 1}} of ${{modelViews.length}}: <span id="model-title">${{modelTitles[index]}}</span>`;
                
                // Update current index
                currentModelIndex = index;
                
                // Update button states
                prevBtn.disabled = currentModelIndex === 0;
                nextBtn.disabled = currentModelIndex === modelViews.length - 1;
            }}
            
            // Initialize with the first model
            showModel(0);
            
            // Slider change event
            modelSlider.addEventListener('input', function() {{
                const index = parseInt(this.value) - 1;
                showModel(index);
                
                // If playing, pause
                if (isPlaying) {{
                    pausePlayback();
                }}
            }});
            
            // Previous button
            prevBtn.addEventListener('click', function() {{
                if (currentModelIndex > 0) {{
                    showModel(currentModelIndex - 1);
                }}
                
                // If playing, pause
                if (isPlaying) {{
                    pausePlayback();
                }}
            }});
            
            // Next button
            nextBtn.addEventListener('click', function() {{
                if (currentModelIndex < modelViews.length - 1) {{
                    showModel(currentModelIndex + 1);
                }}
                
                // If playing, pause
                if (isPlaying) {{
                    pausePlayback();
                }}
            }});
            
            // Reset button
            resetBtn.addEventListener('click', function() {{
                showModel(0);
                
                // If playing, pause
                if (isPlaying) {{
                    pausePlayback();
                }}
            }});
            
            // Function to start playback
            function startPlayback() {{
                isPlaying = true;
                playBtn.disabled = true;
                pauseBtn.disabled = false;
                
                // Get speed from control
                const speed = parseInt(speedControl.value);
                
                // Start interval
                playInterval = setInterval(function() {{
                    // Increment model index
                    let nextIndex = currentModelIndex + 1;
                    
                    // Loop back to the beginning if at the end
                    if (nextIndex >= modelViews.length) {{
                        nextIndex = 0;
                    }}
                    
                    showModel(nextIndex);
                }}, speed);
            }}
            
            // Function to pause playback
            function pausePlayback() {{
                isPlaying = false;
                playBtn.disabled = false;
                pauseBtn.disabled = true;
                
                // Clear interval
                clearInterval(playInterval);
            }}
            
            // Play button
            playBtn.addEventListener('click', startPlayback);
            
            // Pause button
            pauseBtn.addEventListener('click', pausePlayback);
            
            // Speed control change
            speedControl.addEventListener('change', function() {{
                if (isPlaying) {{
                    // Restart playback with new speed
                    pausePlayback();
                    startPlayback();
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

def create_trajectory_viewer(pdb_file, output_html):
    """
    Create an HTML trajectory viewer for a PDB file.
    
    Args:
        pdb_file (str): Path to the PDB file
        output_html (str): Path for the output HTML file
    """
    # Parse PDB models
    models, model_titles = parse_pdb_models(pdb_file)
    
    print(f"Found {len(models)} models in the PDB file")
    
    # Create py3Dmol views
    view_htmls = create_py3dmol_views(models)
    
    # Generate HTML with slider
    html_content = generate_html_slider_page(view_htmls, model_titles)
    
    # Write to file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"Trajectory viewer created at: {output_html}")
    
    return output_html

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a PDB trajectory viewer.')
    parser.add_argument('pdb_file', help='Path to the PDB file with multiple models')
    parser.add_argument('--output', '-o', default='trajectory_viewer.html', 
                        help='Output HTML file path (default: trajectory_viewer.html)')
    
    args = parser.parse_args()
    
    create_trajectory_viewer(args.pdb_file, args.output)


import os
import re
import py3Dmol
from IPython.display import HTML
import base64
import json

def parse_pdb_models(pdb_file):
    """
    Parse a PDB file into separate models.
    
    Args:
        pdb_file (str): Path to the PDB file
        
    Returns:
        list: A list of models, where each model is a list of PDB lines
    """
    models = []
    current_model = []
    model_titles = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('TITLE'):
                # Store the title for each model
                model_titles.append(line.strip()[6:].strip())
            
            if line.startswith('ENDMDL'):
                current_model.append(line)
                models.append('\n'.join(current_model))
                current_model = []
            elif line.startswith('MODEL') and current_model:
                # If we find a new MODEL line but haven't closed the previous one,
                # implicitly close the previous model
                models.append('\n'.join(current_model))
                current_model = [line]
            else:
                current_model.append(line)
    
    # Add the last model if it exists and wasn't closed with ENDMDL
    if current_model:
        models.append('\n'.join(current_model))
    
    # If no titles were found, generate default ones
    if len(model_titles) < len(models):
        model_titles.extend([f"Model {i+1}" for i in range(len(model_titles), len(models))])
    
    return models, model_titles

def generate_html_slider_page(models, model_titles):
    """
    Generate an HTML page with py3Dmol views and slider controls.
    
    Args:
        models (list): List of model strings
        model_titles (list): List of model titles
        
    Returns:
        str: Complete HTML page
    """
    # Convert model titles to JSON for JavaScript use
    json_model_titles = json.dumps(model_titles)
    
    # Convert models to JSON for JavaScript use
    json_models = json.dumps(models)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDB Trajectory Viewer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .container {{
                width: 700px;
                margin: 0 auto;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                padding: 20px;
                border-radius: 8px;
            }}
            .viewer-container {{
                position: relative;
                width: 600px;
                height: 400px;
                margin: 20px auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .controls {{
                display: flex;
                justify-content: center;
                align-items: center;
                margin-top: 10px;
                gap: 10px;
                flex-wrap: wrap;
            }}
            .slider-container {{
                width: 100%;
                padding: 10px 0;
            }}
            button {{
                padding: 8px 12px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #45a049;
            }}
            button:disabled {{
                background-color: #cccccc;
                cursor: not-allowed;
            }}
            .slider {{
                width: 100%;
            }}
            .model-info {{
                text-align: center;
                margin: 10px 0;
                font-weight: bold;
            }}
            .speed-control {{
                display: flex;
                align-items: center;
                gap: 5px;
                margin-left: 10px;
            }}
            #loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-weight: bold;
                color: #666;
            }}
        </style>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>PDB Trajectory Viewer</h1>
            <div class="viewer-container" id="viewer-container">
                <div id="loading">Loading visualization...</div>
                <div id="3dmol-viewer" style="width: 600px; height: 400px; position: relative;"></div>
            </div>
            <div class="model-info" id="model-info">
                Loading models...
            </div>
            <div class="slider-container">
                <input type="range" min="1" max="1" value="1" class="slider" id="model-slider" disabled>
            </div>
            <div class="controls">
                <button id="prev-btn" disabled>&lt; Previous</button>
                <button id="play-btn" disabled>Play</button>
                <button id="pause-btn" disabled>Pause</button>
                <button id="next-btn" disabled>Next &gt;</button>
                <button id="reset-btn" disabled>Reset</button>
                <div class="speed-control">
                    <label for="speed-control">Speed:</label>
                    <select id="speed-control" disabled>
                        <option value="2000">Slow</option>
                        <option value="1000" selected>Normal</option>
                        <option value="500">Fast</option>
                        <option value="250">Very Fast</option>
                    </select>
                </div>
            </div>
        </div>

        <script>
            // Get elements
            const viewerContainer = document.getElementById('3dmol-viewer');
            const loadingElement = document.getElementById('loading');
            const modelSlider = document.getElementById('model-slider');
            const modelInfo = document.getElementById('model-info');
            const playBtn = document.getElementById('play-btn');
            const pauseBtn = document.getElementById('pause-btn');
            const prevBtn = document.getElementById('prev-btn');
            const nextBtn = document.getElementById('next-btn');
            const resetBtn = document.getElementById('reset-btn');
            const speedControl = document.getElementById('speed-control');
            
            // Model data
            const modelTitles = {json_model_titles};
            const models = {json_models};
            
            let currentModelIndex = 0;
            let isPlaying = false;
            let playInterval;
            let viewer = null;
            
            // Create viewer
            function initializeViewer() {{
                // Create 3Dmol viewer
                viewer = $3Dmol.createViewer(jQuery("#3dmol-viewer"), {{
                    backgroundColor: "white"
                }});
                
                // Enable slider and controls once viewer is ready
                modelSlider.max = models.length;
                modelSlider.disabled = false;
                playBtn.disabled = false;
                resetBtn.disabled = false;
                speedControl.disabled = false;
                
                // Show first model
                showModel(0);
                
                // Hide loading message
                loadingElement.style.display = "none";
            }}
            
            // Function to show a specific model
            function showModel(index) {{
                if (!viewer) return;
                
                // Clear previous models
                viewer.clear();
                
                // Add current model
                viewer.addModel(models[index], "pdb");
                
                // Set style to show backbone with AlphaFold pLDDT coloring
                viewer.setStyle({{"cartoon": {{"color": "spectrum", "colorscheme": {{
                    "prop": "b",
                    "gradient": "roygb",
                    "min": 0,
                    "max": 100
                }}}}}});
                
                // Set view
                viewer.zoomTo();
                
                // Render
                viewer.render();
                
                // Update slider value
                modelSlider.value = index + 1;
                
                // Update model info
                modelInfo.innerText = `Model ${{index + 1}} of ${{models.length}}: ${{modelTitles[index]}}`;
                
                // Update current index
                currentModelIndex = index;
                
                // Update button states
                prevBtn.disabled = currentModelIndex === 0;
                nextBtn.disabled = currentModelIndex === models.length - 1;
            }}
            
            // Initialize the viewer when the page loads
            window.onload = function() {{
                // Short delay to ensure the container is ready
                setTimeout(initializeViewer, 100);
            }};
            
            // Slider change event
            modelSlider.addEventListener('input', function() {{
                const index = parseInt(this.value) - 1;
                showModel(index);
                
                // If playing, pause
                if (isPlaying) {{
                    pausePlayback();
                }}
            }});
            
            // Previous button
            prevBtn.addEventListener('click', function() {{
                if (currentModelIndex > 0) {{
                    showModel(currentModelIndex - 1);
                }}
                
                // If playing, pause
                if (isPlaying) {{
                    pausePlayback();
                }}
            }});
            
            // Next button
            nextBtn.addEventListener('click', function() {{
                if (currentModelIndex < models.length - 1) {{
                    showModel(currentModelIndex + 1);
                }}
                
                // If playing, pause
                if (isPlaying) {{
                    pausePlayback();
                }}
            }});
            
            // Reset button
            resetBtn.addEventListener('click', function() {{
                showModel(0);
                
                // If playing, pause
                if (isPlaying) {{
                    pausePlayback();
                }}
            }});
            
            // Function to start playback
            function startPlayback() {{
                isPlaying = true;
                playBtn.disabled = true;
                pauseBtn.disabled = false;
                
                // Get speed from control
                const speed = parseInt(speedControl.value);
                
                // Start interval
                playInterval = setInterval(function() {{
                    // Increment model index
                    let nextIndex = currentModelIndex + 1;
                    
                    // Loop back to the beginning if at the end
                    if (nextIndex >= models.length) {{
                        nextIndex = 0;
                    }}
                    
                    showModel(nextIndex);
                }}, speed);
            }}
            
            // Function to pause playback
            function pausePlayback() {{
                isPlaying = false;
                playBtn.disabled = false;
                pauseBtn.disabled = true;
                
                // Clear interval
                clearInterval(playInterval);
            }}
            
            // Play button
            playBtn.addEventListener('click', startPlayback);
            
            // Pause button
            pauseBtn.addEventListener('click', pausePlayback);
            
            // Speed control change
            speedControl.addEventListener('change', function() {{
                if (isPlaying) {{
                    // Restart playback with new speed
                    pausePlayback();
                    startPlayback();
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

def create_trajectory_viewer(pdb_file, output_html):
    """
    Create an HTML trajectory viewer for a PDB file.
    
    Args:
        pdb_file (str): Path to the PDB file
        output_html (str): Path for the output HTML file
    """
    # Parse PDB models
    models, model_titles = parse_pdb_models(pdb_file)
    
    print(f"Found {len(models)} models in the PDB file")
    
    # Generate HTML with slider
    html_content = generate_html_slider_page(models, model_titles)
    
    # Write to file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"Trajectory viewer created at: {output_html}")
    
    return output_html

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a PDB trajectory viewer.')
    parser.add_argument('pdb_file', help='Path to the PDB file with multiple models')
    parser.add_argument('--output', '-o', default='trajectory_viewer.html', 
                        help='Output HTML file path (default: trajectory_viewer.html)')
    
    args = parser.parse_args()
    
    create_trajectory_viewer(args.pdb_file, args.output)
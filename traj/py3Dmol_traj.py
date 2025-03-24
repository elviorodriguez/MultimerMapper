
import os
import re
import py3Dmol
from IPython.display import HTML
import json

def parse_pdb_models(pdb_file):
    """
    Parse a PDB file into separate models with improved validation.
    Specifically optimized for PDBs where:
    - Each model starts with ATOM records
    - Ends with TER, ENDMDL, followed by a TITLE line
    
    Args:
        pdb_file (str): Path to the PDB file
        
    Returns:
        list: A list of models, where each model is a string of PDB content
        list: A list of model titles
    """
    # Read the entire file content
    with open(pdb_file, 'r') as f:
        content = f.read()
    
    # For debugging
    # print(f"Read PDB file: {pdb_file}, size: {len(content)} bytes")
    
    # Split the content into models based on ENDMDL + TITLE pattern
    models_raw = []
    model_titles = []
    
    # Use regex to split by ENDMDL followed by TITLE
    model_pattern = r'(ATOM[\s\S]*?ENDMDL\s*\n\s*TITLE\s+[^\n]*)'
    model_matches = re.findall(model_pattern, content, re.DOTALL)
    
    if model_matches:
        # print(f"Found {len(model_matches)} models using ENDMDL+TITLE pattern")
        
        for i, model_text in enumerate(model_matches):
            # Extract the title from the end of the model text
            title_match = re.search(r'TITLE\s+([^\n]*)', model_text)
            if title_match:
                model_title = title_match.group(1).strip()
            else:
                model_title = f"Model {i+1}"
            
            # Remove the TITLE line from the model text to prevent duplication
            model_content = re.sub(r'TITLE\s+[^\n]*$', '', model_text).strip()
            
            models_raw.append(model_content)
            model_titles.append(model_title)
    else:
        # Fallback method: Try to split by just ENDMDL
        # print("No models found with ENDMDL+TITLE pattern. Trying alternative approach...")
        
        sections = content.split('ENDMDL')
        
        for i, section in enumerate(sections):
            if i == len(sections) - 1:  # Skip last section if it's empty
                continue
                
            section = section.strip() + "\nENDMDL"  # Add back the ENDMDL
            
            if 'ATOM' in section:
                models_raw.append(section)
                
                # Try to find title after this section
                next_title_match = re.search(r'TITLE\s+([^\n]*)', sections[i+1])
                if next_title_match:
                    model_titles.append(next_title_match.group(1).strip())
                else:
                    # Look for title within this section
                    title_match = re.search(r'TITLE\s+([^\n]*)', section)
                    if title_match:
                        model_titles.append(title_match.group(1).strip())
                    else:
                        model_titles.append(f"Model {i+1}")
        
        # print(f"Found {len(models_raw)} models using ENDMDL-only pattern")
    
    # If still no models found, try another approach based on ATOM pattern
    if not models_raw:
        # print("Still no models found. Trying to parse based on ATOM sections...")
        
        # Look for patterns where ATOM 1 appears (beginning of each model)
        atom_starts = [m.start() for m in re.finditer(r'ATOM\s+1\s+N\s+MET', content)]
        
        if len(atom_starts) > 1:
            # print(f"Found {len(atom_starts)} potential model starts with 'ATOM 1'")
            
            for i in range(len(atom_starts)):
                start = atom_starts[i]
                if i < len(atom_starts) - 1:
                    end = atom_starts[i+1]
                    model_text = content[start:end].strip()
                else:
                    model_text = content[start:].strip()
                
                # Extract title if present
                title_match = re.search(r'TITLE\s+([^\n]*)', model_text)
                if title_match:
                    model_title = title_match.group(1).strip()
                    # Remove the title from the model text
                    model_text = re.sub(r'TITLE\s+[^\n]*', '', model_text).strip()
                else:
                    model_title = f"Model {i+1}"
                
                models_raw.append(model_text)
                model_titles.append(model_title)
    
    # Print some diagnostic info about the models
    # print(f"Found {len(models_raw)} models in the PDB file")
    for i, model in enumerate(models_raw):
        atom_count = model.count("ATOM")
        hetatm_count = model.count("HETATM")
        # print(f"Model {i+1}: {atom_count} ATOM records, {hetatm_count} HETATM records")
        # print(f"Title: {model_titles[i]}")
        
        # Print the first few lines of each model for verification
        first_lines = model.split('\n')[:2]
        # print(f"First lines: {first_lines}")
        
        # Check if model starts with ATOM and contains ENDMDL
        if not model.lstrip().startswith("ATOM"):
            print(f"WARNING: Model {i+1} does not start with ATOM")
        if "ENDMDL" not in model:
            print(f"WARNING: Model {i+1} does not contain ENDMDL")
            # Add ENDMDL if missing
            models_raw[i] = model + "\nENDMDL"
    
    return models_raw, model_titles

def create_trajectory_viewer(pdb_file, output_html):
    """
    Create an HTML trajectory viewer for a PDB file.
    
    Args:
        pdb_file (str): Path to the PDB file
        output_html (str): Path for the output HTML file
    """
    # Parse PDB models
    models, model_titles = parse_pdb_models(pdb_file)
    
    # print(f"Found {len(models)} models in the PDB file")
    
    # Embed models and titles in the HTML
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>PDB Trajectory Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        h2 {
            text-align: center;
        }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            width: 650px;
            margin: 0 auto;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            padding: 20px;
            border-radius: 8px;
        }
        .viewer-container {
            width: 600px;
            height: 600px;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            position: relative;
        }
        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
            gap: 10px;
        }
        .slider-container {
            width: 100%;
            padding: 10px 0;
        }
        button {
            padding: 8px 12px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .slider {
            width: 100%;
        }
        .model-info {
            text-align: center;
            margin: 10px 0;
            font-weight: bold;
        }
        .speed-control {
            display: flex;
            align-items: center;
            gap: 5px;
            margin-left: 20px;
        }
        #status {
            color: red;
            text-align: center;
            margin: 10px 0;
            min-height: 20px;
        }
        .model-style-control {
            display: flex;
            justify-content: center;
            margin: 10px 0;
            gap: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>PDB Trajectory Viewer</h2>
        <div id="status"></div>
        <div class="viewer-container" id="viewer-container"></div>
        <div class="model-info" id="model-info">Loading...</div>
        <div class="slider-container">
            <input type="range" min="1" max="1" value="1" class="slider" id="model-slider" disabled>
        </div>
        <div class="model-style-control">
            <label>Style:</label>
            <select id="style-select">
                <option value="cartoon">Cartoon</option>
                <option value="line">Line</option>
                <option value="stick">Stick</option>
                <option value="sphere">Sphere</option>
            </select>
            <label>Color:</label>
            <select id="color-select">
                <option value="spectrum">Spectrum</option>
                <option value="chain">Chain</option>
                <option value="residue">Residue</option>
                <option value="secondary">Secondary Structure</option>
                <option value="plddt" selected>pLDDT</option>
            </select>
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
                    <option value="250" selected>Very Fast</option>
                </select>
            </div>
        </div>
    </div>

    <script>
        // Define custom colorscale for pLDDT
        const plddt_colorscale = [
            [0.0, "#FF0000"],
            [0.4, "#FFA500"],
            [0.6, "#FFFF00"],
            [0.8, "#ADD8E6"],
            [1.0, "#00008B"]
        ];

        // Helper function to interpolate colors in the scale
        function getColorFromScale(value, scale) {
            // Find the appropriate color range
            let lowerIndex = 0;
            for (let i = 0; i < scale.length; i++) {
                if (value <= scale[i][0]) {
                    break;
                }
                lowerIndex = i;
            }
            
            // If at the end of the scale, return the last color
            if (lowerIndex >= scale.length - 1) {
                return scale[scale.length - 1][1];
            }
            
            // Interpolate between the two nearest colors
            const lowerValue = scale[lowerIndex][0];
            const upperValue = scale[lowerIndex + 1][0];
            const valueFraction = (value - lowerValue) / (upperValue - lowerValue);
            
            const lowerColor = hexToRgb(scale[lowerIndex][1]);
            const upperColor = hexToRgb(scale[lowerIndex + 1][1]);
            
            // Linear interpolation of RGB values
            const r = Math.round(lowerColor.r + valueFraction * (upperColor.r - lowerColor.r));
            const g = Math.round(lowerColor.g + valueFraction * (upperColor.g - lowerColor.g));
            const b = Math.round(lowerColor.b + valueFraction * (upperColor.b - lowerColor.b));
            
            return rgbToHex(r, g, b);
        }

        // Helper function to convert hex to rgb
        function hexToRgb(hex) {
            const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
            return result ? {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            } : {r: 0, g: 0, b: 0};
        }

        // Helper function to convert rgb to hex
        function rgbToHex(r, g, b) {
            return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
        }
        // Function to show status messages
        function showStatus(message, isError = false) {
            const statusElement = document.getElementById('status');
            statusElement.textContent = message;
            statusElement.style.color = isError ? 'red' : 'green';
            console.log((isError ? 'ERROR: ' : 'INFO: ') + message);
        }

        // Model data
        const modelData = """ + json.dumps(models) + """;
        const modelTitles = """ + json.dumps(model_titles) + """;
        const totalModels = modelData.length;

        // Set up variables
        let viewer = null;
        let currentModelIndex = 0;
        let isPlaying = false;
        let playInterval = null;

        // Get UI elements
        const viewerContainer = document.getElementById('viewer-container');
        const modelSlider = document.getElementById('model-slider');
        const modelInfo = document.getElementById('model-info');
        const playBtn = document.getElementById('play-btn');
        const pauseBtn = document.getElementById('pause-btn');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const resetBtn = document.getElementById('reset-btn');
        const speedControl = document.getElementById('speed-control');
        const styleSelect = document.getElementById('style-select');
        const colorSelect = document.getElementById('color-select');

        // Function to apply current style settings
        function applyCurrentStyle() {
            if (!viewer) return;
            
            const style = styleSelect.value;
            const colorScheme = colorSelect.value;
            
            // Remove all styles first
            viewer.setStyle({}, {});
            
            // Apply the selected style
            const styleObj = {};
            
            if (colorScheme === 'spectrum') {
                styleObj[style] = {color: 'spectrum'};
            } else if (colorScheme === 'chain') {
                styleObj[style] = {colorscheme: 'chainHetatm'};
            } else if (colorScheme === 'residue') {
                styleObj[style] = {colorscheme: 'amino'};
            } else if (colorScheme === 'secondary') {
                styleObj[style] = {colorscheme: 'ssPyMOL'};
            } else if (colorScheme === 'plddt') {
                // Apply pLDDT coloring based on B-factor values
                styleObj[style] = {colorfunc: function(atom) {
                    // Normalize B-factor value (clamped between 0 and 100)
                    const bfactor = Math.max(0, Math.min(100, atom.b));
                    const normalizedValue = bfactor / 100;
                    
                    // Find the color using the pLDDT colorscale
                    return getColorFromScale(normalizedValue, plddt_colorscale);
                }};
            }
            
            viewer.setStyle({}, styleObj);
            viewer.render();
        }

        // Function to display a model
        function displayModel(index) {
            try {
                if (!viewer) {
                    showStatus("Viewer not initialized", true);
                    return;
                }

                showStatus("Loading model " + (index + 1) + "...");
                
                // Clear current model
                viewer.clear();
                
                // Add new model with simpler options
                try {
                    viewer.addModel(modelData[index], 'pdb');
                    
                    // Apply current style settings
                    applyCurrentStyle();
                    
                    // Adjust view
                    viewer.addModel(modelData[index], 'pdb');
                    applyCurrentStyle();                    
                    viewer.render();
                    
                    // Update UI
                    currentModelIndex = index;
                    modelSlider.value = index + 1;
                    modelInfo.textContent = `Model ${index + 1} of ${totalModels}: ${modelTitles[index]}`;
                    
                    // Update button states
                    prevBtn.disabled = (index === 0);
                    nextBtn.disabled = (index === totalModels - 1);
                    
                    showStatus("Model loaded successfully");
                } catch (renderError) {
                    showStatus("Error rendering model: " + renderError.message, true);
                    console.error("Render error details:", renderError);
                }
            } catch (error) {
                showStatus("Error displaying model: " + error.message, true);
                console.error("Error displaying model:", error);
            }
        }

        // Function to initialize the viewer
        function initViewer() {
            try {
                showStatus("Initializing viewer...");
                
                // Update slider
                modelSlider.max = totalModels;
                modelSlider.disabled = false;
                
                // Create viewer with explicit size
                let config = {
                    backgroundColor: 'white',
                    antialias: true
                };
                
                if ($(viewerContainer).length > 0) {
                    // First create a clear container
                    $(viewerContainer).empty();
                    
                    viewer = $3Dmol.createViewer($(viewerContainer), config);
                    
                    if (viewer) {
                        // Enable controls after successful viewer creation
                        playBtn.disabled = false;
                        pauseBtn.disabled = true;
                        resetBtn.disabled = false;
                        speedControl.disabled = false;
                        prevBtn.disabled = false;
                        nextBtn.disabled = false;
                        
                        // Wait a moment before loading the first model
                        setTimeout(function() {
                            displayModel(0);
                        }, 500);
                        
                        showStatus("Viewer ready");
                    } else {
                        throw new Error("Failed to create 3Dmol viewer object");
                    }
                } else {
                    throw new Error("Viewer container element not found");
                }
            } catch (error) {
                showStatus("Error initializing viewer: " + error.message, true);
                console.error("Error initializing viewer:", error);
            }
        }

        // Setup playback controls
        function startPlayback() {
            try {
                isPlaying = true;
                playBtn.disabled = true;
                pauseBtn.disabled = false;
                
                const speed = parseInt(speedControl.value);
                
                playInterval = setInterval(function() {
                    let nextIndex = currentModelIndex + 1;
                    if (nextIndex >= totalModels) {
                        nextIndex = 0;
                    }
                    displayModel(nextIndex);
                }, speed);
            } catch (error) {
                showStatus("Error starting playback: " + error.message, true);
                stopPlayback();
            }
        }

        function stopPlayback() {
            isPlaying = false;
            playBtn.disabled = false;
            pauseBtn.disabled = true;
            
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
            }
        }

        // Attach event listeners
        modelSlider.addEventListener('input', function() {
            const index = parseInt(this.value) - 1;
            if (isPlaying) {
                stopPlayback();
            }
            displayModel(index);
        });

        playBtn.addEventListener('click', startPlayback);
        pauseBtn.addEventListener('click', stopPlayback);

        prevBtn.addEventListener('click', function() {
            if (currentModelIndex > 0) {
                if (isPlaying) {
                    stopPlayback();
                }
                displayModel(currentModelIndex - 1);
            }
        });

        nextBtn.addEventListener('click', function() {
            if (currentModelIndex < totalModels - 1) {
                if (isPlaying) {
                    stopPlayback();
                }
                displayModel(currentModelIndex + 1);
            }
        });

        resetBtn.addEventListener('click', function() {
            if (isPlaying) {
                stopPlayback();
            }
            displayModel(0);
        });

        speedControl.addEventListener('change', function() {
            if (isPlaying) {
                stopPlayback();
                startPlayback();
            }
        });
        
        // Style controls
        styleSelect.addEventListener('change', function() {
            applyCurrentStyle();
        });
        
        colorSelect.addEventListener('change', function() {
            applyCurrentStyle();
        });

        // Initialize when the page loads
        window.onload = function() {
            showStatus("Page loaded, starting initialization");
            
            // Small delay to ensure DOM is fully loaded
            setTimeout(function() {
                if (!modelData || modelData.length === 0) {
                    showStatus("No models found in the PDB file", true);
                } else {
                    initViewer();
                }
            }, 1000);
        };
    </script>
</body>
</html>
    """
    
    # Write to file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    # print(f"Trajectory viewer created at: {output_html}")
    return output_html

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a PDB trajectory viewer.')
    parser.add_argument('pdb_file', help='Path to the PDB file with multiple models')
    parser.add_argument('--output', '-o', default='trajectory_viewer.html', 
                        help='Output HTML file path (default: trajectory_viewer.html)')
    
    args = parser.parse_args()
    
    create_trajectory_viewer(args.pdb_file, args.output)
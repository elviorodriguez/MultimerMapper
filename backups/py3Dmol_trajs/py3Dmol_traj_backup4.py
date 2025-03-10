import os
import re
import py3Dmol
from IPython.display import HTML
import json

def parse_pdb_models(pdb_file):
    """
    Parse a PDB file into separate models with improved validation.
    
    Args:
        pdb_file (str): Path to the PDB file
        
    Returns:
        list: A list of models, where each model is a string of PDB content
        list: A list of model titles
    """
    # Read the entire file content
    with open(pdb_file, 'r') as f:
        content = f.read()
    
    # Split the content into models more reliably
    models_raw = []
    model_titles = []
    
    # Split by MODEL/ENDMDL sections
    current_model = []
    current_title = None
    in_model = False
    
    for line in content.splitlines():
        if line.startswith('MODEL'):
            in_model = True
            if current_model:  # If we have collected lines for a previous model
                models_raw.append('\n'.join(current_model))
                model_titles.append(current_title if current_title else f"Model {len(models_raw)}")
            current_model = [line]
            current_title = None
        elif line.startswith('TITLE') and in_model:
            current_title = line.replace('TITLE', '').strip()
        elif line.startswith('ENDMDL'):
            current_model.append(line)
            in_model = False
            models_raw.append('\n'.join(current_model))
            model_titles.append(current_title if current_title else f"Model {len(models_raw)}")
            current_model = []
            current_title = None
        elif in_model:  # Only append if we're within a model
            current_model.append(line)
    
    # Add the last model if it wasn't terminated with ENDMDL
    if current_model:
        models_raw.append('\n'.join(current_model))
        model_titles.append(current_title if current_title else f"Model {len(models_raw)}")
    
    # If no models were found, try an alternative approach
    if not models_raw:
        print("No models found with MODEL/ENDMDL tags. Trying alternative parsing...")
        
        # Try to split by TER followed by TITLE/MODEL
        sections = re.split(r'(TER.*?\n(?:TITLE|MODEL))', content, flags=re.DOTALL)
        
        current_section = []
        for i, section in enumerate(sections):
            if i == 0 and 'ATOM' in section:  # First section with atoms
                current_section.append(section)
            elif section.startswith('TER') and i+1 < len(sections):  # TER followed by TITLE/MODEL
                current_section.append(section)
                models_raw.append('\n'.join(current_section))
                
                # Extract title if available
                title_match = re.search(r'TITLE\s+(.*?)(?:\n|$)', '\n'.join(current_section))
                if title_match:
                    model_titles.append(title_match.group(1).strip())
                else:
                    model_titles.append(f"Model {len(model_titles) + 1}")
                    
                current_section = []
            elif 'ATOM' in section:  # Any section with atoms
                current_section.append(section)
        
        # Add the last section if it contains atoms
        if current_section and any('ATOM' in line for line in current_section):
            models_raw.append('\n'.join(current_section))
            
            # Extract title if available
            title_match = re.search(r'TITLE\s+(.*?)(?:\n|$)', '\n'.join(current_section))
            if title_match:
                model_titles.append(title_match.group(1).strip())
            else:
                model_titles.append(f"Model {len(model_titles) + 1}")
    
    # If still no models found, try another approach based on your sample
    if not models_raw:
        print("Still no models found. Trying to parse based on TITLE and ENDMDL...")
        sections = content.split('TITLE')
        
        for i, section in enumerate(sections):
            if i == 0:  # Skip the first empty split
                continue
                
            if 'ATOM' in section:
                models_raw.append("TITLE" + section)
                
                # Extract title
                title_line = section.strip().split('\n')[0]
                model_titles.append(title_line.strip())
    
    # Print some diagnostic info about the models
    print(f"Found {len(models_raw)} models in the PDB file")
    for i, model in enumerate(models_raw):
        atom_count = model.count("ATOM")
        hetatm_count = model.count("HETATM")
        print(f"Model {i+1}: {atom_count} ATOM records, {hetatm_count} HETATM records")
    
    return models_raw, model_titles

def create_trajectory_viewer(pdb_file, output_html):
    """
    Create an HTML trajectory viewer for a PDB file with improved visualization.
    
    Args:
        pdb_file (str): Path to the PDB file
        output_html (str): Path for the output HTML file
    """
    # Parse PDB models
    models, model_titles = parse_pdb_models(pdb_file)
    
    # Embed models and titles in the HTML
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>PDB Trajectory Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
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
            height: 400px;
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
        .debug-info {
            margin-top: 20px;
            padding: 10px;
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PDB Trajectory Viewer</h1>
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
                    <option value="250">Very Fast</option>
                </select>
            </div>
        </div>
        <div id="debug-info" class="debug-info"></div>
    </div>

    <script>
        // Function to show status messages
        function showStatus(message, isError = false) {
            const statusElement = document.getElementById('status');
            statusElement.textContent = message;
            statusElement.style.color = isError ? 'red' : 'green';
            console.log((isError ? 'ERROR: ' : 'INFO: ') + message);
        }

        // Function to log debug information
        function logDebug(message) {
            const debugElement = document.getElementById('debug-info');
            const timestamp = new Date().toLocaleTimeString();
            debugElement.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            // Auto-scroll to bottom
            debugElement.scrollTop = debugElement.scrollHeight;
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

        // Function to check if model content has required elements
        function validateModel(modelContent) {
            // Get atom count by counting lines that start with ATOM
            const lines = modelContent.split('\n');
            const atomCount = lines.filter(line => line.trim().startsWith('ATOM')).length;
            logDebug(`Model validation: ${atomCount} ATOM records found`);
            
            // Check if model has ATOM records
            if (atomCount === 0) {
                return false;
            }
            return true;
        }

        // Function to apply current style settings with enhanced error handling
        function applyCurrentStyle() {
            if (!viewer) {
                showStatus("Viewer not available for styling", true);
                return;
            }
            
            try {
                const style = styleSelect.value;
                const colorScheme = colorSelect.value;
                
                logDebug(`Applying style: ${style}, colorScheme: ${colorScheme}`);
                
                // First, clear all styles
                viewer.removeAllModels();
                viewer.removeAllLabels();
                viewer.removeAllShapes();
                viewer.removeAllSurfaces();
                
                // Re-add current model
                try {
                    const modelContent = modelData[currentModelIndex];
                    viewer.addModel(modelContent, "pdb");
                    logDebug("Model re-added for styling");
                } catch (modelError) {
                    logDebug(`Error re-adding model: ${modelError.message}`);
                    showStatus("Error re-adding model for styling", true);
                    return;
                }
                
                // Apply style based on selections
                try {
                    if (style === 'cartoon') {
                        if (colorScheme === 'spectrum') {
                            viewer.setStyle({}, {cartoon: {color: 'spectrum'}});
                        } else if (colorScheme === 'chain') {
                            viewer.setStyle({}, {cartoon: {colorscheme: 'chainHetatm'}});
                        } else if (colorScheme === 'residue') {
                            viewer.setStyle({}, {cartoon: {colorscheme: 'amino'}});
                        } else if (colorScheme === 'secondary') {
                            viewer.setStyle({}, {cartoon: {colorscheme: 'ssPyMOL'}});
                        }
                    } else if (style === 'line') {
                        if (colorScheme === 'spectrum') {
                            viewer.setStyle({}, {line: {color: 'spectrum'}});
                        } else if (colorScheme === 'chain') {
                            viewer.setStyle({}, {line: {colorscheme: 'chainHetatm'}});
                        } else if (colorScheme === 'residue') {
                            viewer.setStyle({}, {line: {colorscheme: 'amino'}});
                        } else if (colorScheme === 'secondary') {
                            viewer.setStyle({}, {line: {colorscheme: 'ssPyMOL'}});
                        }
                    } else if (style === 'stick') {
                        if (colorScheme === 'spectrum') {
                            viewer.setStyle({}, {stick: {color: 'spectrum'}});
                        } else if (colorScheme === 'chain') {
                            viewer.setStyle({}, {stick: {colorscheme: 'chainHetatm'}});
                        } else if (colorScheme === 'residue') {
                            viewer.setStyle({}, {stick: {colorscheme: 'amino'}});
                        } else if (colorScheme === 'secondary') {
                            viewer.setStyle({}, {stick: {colorscheme: 'ssPyMOL'}});
                        }
                    } else if (style === 'sphere') {
                        if (colorScheme === 'spectrum') {
                            viewer.setStyle({}, {sphere: {color: 'spectrum'}});
                        } else if (colorScheme === 'chain') {
                            viewer.setStyle({}, {sphere: {colorscheme: 'chainHetatm'}});
                        } else if (colorScheme === 'residue') {
                            viewer.setStyle({}, {sphere: {colorscheme: 'amino'}});
                        } else if (colorScheme === 'secondary') {
                            viewer.setStyle({}, {sphere: {colorscheme: 'ssPyMOL'}});
                        }
                    }
                    
                    // Add protein backbone separately for visibility
                    viewer.addStyle({"atom":"CA"}, {line:{color:"#cccccc",width:1.0}});
                    viewer.addStyle({"elem":"P"}, {sphere:{color:"orange",radius:1.0}});
                    
                    // Explicitly select protein and apply style
                    viewer.setStyle({"hetflag":false}, {cartoon:{style:"oval"}});
                    viewer.setStyle({"hetflag":true}, {stick:{radius:0.15, colorscheme:"greenCarbon"}});
                    
                    logDebug("Styles applied successfully");
                } catch (styleError) {
                    logDebug(`Error applying style: ${styleError.message}`);
                    showStatus(`Error applying style: ${styleError.message}`, true);
                    
                    // Apply a very basic style as fallback
                    try {
                        viewer.setStyle({}, {line: {}});
                        logDebug("Applied fallback line style");
                    } catch (fallbackError) {
                        logDebug(`Even fallback style failed: ${fallbackError.message}`);
                    }
                }
                
                // Adjust view and render
                try {
                    viewer.zoomTo();
                    viewer.zoom(0.8);  // Zoom out slightly to see the whole structure
                    viewer.render();
                    logDebug("View adjusted and rendered");
                } catch (viewError) {
                    logDebug(`View adjustment error: ${viewError.message}`);
                }
            } catch (error) {
                showStatus("Error in style application: " + error.message, true);
                logDebug(`General style error: ${error.message}`);
            }
        }

        // Function to display a model with improved visualization
        function displayModel(index) {
            try {
                if (!viewer) {
                    showStatus("Viewer not initialized", true);
                    return;
                }

                showStatus("Loading model " + (index + 1) + "...");
                logDebug(`Loading model ${index + 1} of ${totalModels}`);
                
                // Validate model content
                if (!validateModel(modelData[index])) {
                    showStatus("Model " + (index + 1) + " appears to be invalid (no ATOM records)", true);
                    logDebug(`Model ${index + 1} validation failed - no ATOM records`);
                    return;
                }
                
                // Clear current model
                viewer.removeAllModels();
                viewer.removeAllLabels();
                viewer.removeAllShapes();
                viewer.removeAllSurfaces();
                
                // Add new model with enhanced error handling
                try {
                    // Add the model with parsing options
                    viewer.addModel(modelData[index], 'pdb', {doAssembly: true, noSecondaryStructure: false});
                    logDebug(`Model ${index + 1} added successfully`);
                    
                    // Apply styles immediately
                    applyCurrentStyle();
                    
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
                    logDebug(`Render error details: ${renderError.message}`);
                    console.error("Render error details:", renderError);
                    
                    // Try a more basic approach as fallback
                    try {
                        // First clear everything
                        viewer.removeAllModels();
                        
                        // Try adding with different options
                        viewer.addModel(modelData[index], 'pdb', {keepH: true, doAssembly: false});
                        viewer.setStyle({}, {stick: {radius: 0.15, colorscheme: "default"}});
                        viewer.zoomTo();
                        viewer.render();
                        logDebug("Applied simplified fallback rendering");
                        showStatus("Using simplified rendering due to error with normal mode", true);
                    } catch (fallbackError) {
                        logDebug(`Even fallback rendering failed: ${fallbackError.message}`);
                        
                        // Try with a very basic dummy structure
                        try {
                            viewer.addModel("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C", 'pdb');
                            viewer.setStyle({}, {sphere: {radius: 2.0}});
                            viewer.zoomTo();
                            viewer.render();
                            logDebug("Loaded dummy atom as last resort");
                            showStatus("Unable to render model, showing placeholder", true);
                        } catch (dummyError) {
                            logDebug(`Failed to load even dummy structure: ${dummyError.message}`);
                            showStatus("Rendering system failed completely", true);
                        }
                    }
                }
            } catch (error) {
                showStatus("Error displaying model: " + error.message, true);
                logDebug(`General error displaying model: ${error.message}`);
                console.error("Error displaying model:", error);
            }
        }

        // Function to initialize the viewer with improved visualization capabilities
        function initViewer() {
            try {
                showStatus("Initializing viewer...");
                logDebug("Starting viewer initialization");
                
                // Update slider
                modelSlider.max = totalModels;
                modelSlider.disabled = false;
                
                // Create viewer with enhanced configuration
                let config = {
                    backgroundColor: 'white',
                    antialias: true,
                    disableFog: true,
                    outline: true,                     // Add outline to make structure more visible
                    outlineWidth: 0.1,                 // Thin outline
                    outlineColor: new $3Dmol.Color(0x000000), // Black outline
                    defaultcolors: $3Dmol.rasmolElementColors // Use RasMol color scheme
                };
                
                if ($(viewerContainer).length > 0) {
                    // First create a clear container
                    $(viewerContainer).empty();
                    logDebug("Viewer container cleared");
                    
                    try {
                        viewer = $3Dmol.createViewer($(viewerContainer), config);
                        
                        if (viewer) {
                            logDebug("Viewer created successfully");
                            
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
                    } catch (viewerError) {
                        showStatus("Error creating 3Dmol viewer: " + viewerError.message, true);
                        logDebug(`Viewer creation error: ${viewerError.message}`);
                        
                        // Try again with minimal configuration
                        try {
                            logDebug("Attempting fallback viewer creation");
                            viewer = $3Dmol.createViewer($(viewerContainer));
                            showStatus("Created viewer with minimal configuration", true);
                            
                            // Enable basic controls
                            playBtn.disabled = false;
                            resetBtn.disabled = false;
                            
                            // Load first model
                            setTimeout(function() {
                                displayModel(0);
                            }, 500);
                        } catch (retryError) {
                            logDebug(`Fallback viewer creation also failed: ${retryError.message}`);
                            showStatus("Failed to create viewer. Browser may not support WebGL.", true);
                        }
                    }
                } else {
                    throw new Error("Viewer container element not found");
                }
            } catch (error) {
                showStatus("Error initializing viewer: " + error.message, true);
                logDebug(`General initialization error: ${error.message}`);
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
                logDebug(`Starting playback with speed: ${speed}ms`);
                
                playInterval = setInterval(function() {
                    let nextIndex = currentModelIndex + 1;
                    if (nextIndex >= totalModels) {
                        nextIndex = 0;
                    }
                    displayModel(nextIndex);
                }, speed);
            } catch (error) {
                showStatus("Error starting playback: " + error.message, true);
                logDebug(`Playback start error: ${error.message}`);
                stopPlayback();
            }
        }

        function stopPlayback() {
            isPlaying = false;
            playBtn.disabled = false;
            pauseBtn.disabled = true;
            logDebug("Playback stopped");
            
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
            
            // Completely reinitialize the viewer
            logDebug("Reset requested - reinitializing viewer");
            initViewer();
        });

        speedControl.addEventListener('change', function() {
            if (isPlaying) {
                stopPlayback();
                startPlayback();
            }
        });
        
        // Style controls
        styleSelect.addEventListener('change', function() {
            logDebug(`Style changed to: ${this.value}`);
            applyCurrentStyle();
        });
        
        colorSelect.addEventListener('change', function() {
            logDebug(`Color scheme changed to: ${this.value}`);
            applyCurrentStyle();
        });

        // Initialize when the page loads
        window.onload = function() {
            showStatus("Page loaded, starting initialization");
            logDebug("Page loaded");
            
            // Verify that models are available
            if (!modelData || modelData.length === 0) {
                showStatus("No models found in the PDB file", true);
                logDebug("No models found in data");
                return;
            }
            
            // Check for 3Dmol availability
            if (typeof $3Dmol === 'undefined') {
                showStatus("3Dmol.js library not loaded properly. Check your internet connection and try again.", true);
                logDebug("3Dmol library not available");
                return;
            }
            
            // Debug model data
            logDebug(`Found ${modelData.length} models`);
            for (let i = 0; i < modelData.length; i++) {
                const atomCount = (modelData[i].match(/ATOM/g) || []).length;
                logDebug(`Model ${i+1}: ${atomCount} ATOM records, ~${modelData[i].length} characters`);
            }
            
            // Small delay to ensure DOM is fully loaded
            setTimeout(function() {
                initViewer();
            }, 1000);
        };
    </script>
</body>
</html>
    """
    
    # Write to file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"Enhanced trajectory viewer created at: {output_html}")
    return output_html

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a PDB trajectory viewer.')
    parser.add_argument('pdb_file', help='Path to the PDB file with multiple models')
    parser.add_argument('--output', '-o', default='trajectory_viewer.html', 
                        help='Output HTML file path (default: trajectory_viewer.html)')
    
    args = parser.parse_args()
    
    create_trajectory_viewer(args.pdb_file, args.output)
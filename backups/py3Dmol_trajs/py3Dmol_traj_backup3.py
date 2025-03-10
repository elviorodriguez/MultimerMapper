# import os
# import re
# import py3Dmol
# from IPython.display import HTML
# import base64
# import json

# def parse_pdb_models(pdb_file):
#     """
#     Parse a PDB file into separate models.
    
#     Args:
#         pdb_file (str): Path to the PDB file
        
#     Returns:
#         list: A list of models, where each model is a string of PDB content
#     """
#     # Read the entire file content
#     with open(pdb_file, 'r') as f:
#         content = f.read()
    
#     # Check if the file has MODEL/ENDMDL tags
#     if 'MODEL' in content and 'ENDMDL' in content:
#         # Split by MODEL/ENDMDL
#         pattern = r'(MODEL\s+\d+.*?ENDMDL)'
#         models_raw = re.findall(pattern, content, re.DOTALL)
        
#         if not models_raw:  # If regex failed, fallback to simpler approach
#             models_raw = []
#             current_model = []
#             for line in content.splitlines():
#                 if line.startswith('MODEL'):
#                     if current_model:  # If we have collected lines for a model
#                         models_raw.append('\n'.join(current_model))
#                     current_model = [line]
#                 elif line.startswith('ENDMDL'):
#                     current_model.append(line)
#                     models_raw.append('\n'.join(current_model))
#                     current_model = []
#                 elif current_model:  # Only append if we're within a model
#                     current_model.append(line)
            
#             # Add the last model if it wasn't terminated with ENDMDL
#             if current_model:
#                 models_raw.append('\n'.join(current_model))
#     else:
#         # Treat the entire file as a single model
#         models_raw = [content]
    
#     # Extract titles if present
#     model_titles = []
#     for model in models_raw:
#         title_match = re.search(r'TITLE\s+(.*?)(?:\n|$)', model)
#         if title_match:
#             model_titles.append(title_match.group(1).strip())
#         else:
#             model_titles.append(f"Model {len(model_titles) + 1}")
    
#     # If no titles were extracted, create default ones
#     if not model_titles or len(model_titles) < len(models_raw):
#         model_titles = [f"Model {i+1}" for i in range(len(models_raw))]
    
#     return models_raw, model_titles

# def create_trajectory_viewer(pdb_file, output_html):
#     """
#     Create an HTML trajectory viewer for a PDB file.
    
#     Args:
#         pdb_file (str): Path to the PDB file
#         output_html (str): Path for the output HTML file
#     """
#     # Parse PDB models
#     models, model_titles = parse_pdb_models(pdb_file)
    
#     print(f"Found {len(models)} models in the PDB file")
    
#     # Embed models and titles in the HTML
#     html_content = """
# <!DOCTYPE html>
# <html>
# <head>
#     <title>PDB Trajectory Viewer</title>
#     <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
#     <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             margin: 0;
#             padding: 20px;
#             display: flex;
#             flex-direction: column;
#             align-items: center;
#         }
#         .container {
#             width: 650px;
#             margin: 0 auto;
#             box-shadow: 0 0 10px rgba(0,0,0,0.1);
#             padding: 20px;
#             border-radius: 8px;
#         }
#         .viewer-container {
#             width: 600px;
#             height: 400px;
#             margin: 20px auto;
#             border: 1px solid #ddd;
#             border-radius: 4px;
#             position: relative;
#         }
#         .controls {
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             margin-top: 10px;
#             gap: 10px;
#         }
#         .slider-container {
#             width: 100%;
#             padding: 10px 0;
#         }
#         button {
#             padding: 8px 12px;
#             background-color: #4CAF50;
#             color: white;
#             border: none;
#             border-radius: 4px;
#             cursor: pointer;
#         }
#         button:hover {
#             background-color: #45a049;
#         }
#         button:disabled {
#             background-color: #cccccc;
#             cursor: not-allowed;
#         }
#         .slider {
#             width: 100%;
#         }
#         .model-info {
#             text-align: center;
#             margin: 10px 0;
#             font-weight: bold;
#         }
#         .speed-control {
#             display: flex;
#             align-items: center;
#             gap: 5px;
#             margin-left: 20px;
#         }
#         #status {
#             color: red;
#             text-align: center;
#             margin: 10px 0;
#             min-height: 20px;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>PDB Trajectory Viewer</h1>
#         <div id="status"></div>
#         <div class="viewer-container" id="viewer-container"></div>
#         <div class="model-info" id="model-info">Loading...</div>
#         <div class="slider-container">
#             <input type="range" min="1" max="1" value="1" class="slider" id="model-slider" disabled>
#         </div>
#         <div class="controls">
#             <button id="prev-btn" disabled>&lt; Previous</button>
#             <button id="play-btn" disabled>Play</button>
#             <button id="pause-btn" disabled>Pause</button>
#             <button id="next-btn" disabled>Next &gt;</button>
#             <button id="reset-btn" disabled>Reset</button>
#             <div class="speed-control">
#                 <label for="speed-control">Speed:</label>
#                 <select id="speed-control" disabled>
#                     <option value="2000">Slow</option>
#                     <option value="1000" selected>Normal</option>
#                     <option value="500">Fast</option>
#                     <option value="250">Very Fast</option>
#                 </select>
#             </div>
#         </div>
#     </div>

#     <script>
#         // Function to show status messages
#         function showStatus(message, isError = false) {
#             const statusElement = document.getElementById('status');
#             statusElement.textContent = message;
#             statusElement.style.color = isError ? 'red' : 'green';
#         }

#         // Model data
#         const modelData = """ + json.dumps(models) + """;
#         const modelTitles = """ + json.dumps(model_titles) + """;
#         const totalModels = modelData.length;

#         // Set up variables
#         let viewer = null;
#         let currentModelIndex = 0;
#         let isPlaying = false;
#         let playInterval = null;

#         // Get UI elements
#         const viewerContainer = document.getElementById('viewer-container');
#         const modelSlider = document.getElementById('model-slider');
#         const modelInfo = document.getElementById('model-info');
#         const playBtn = document.getElementById('play-btn');
#         const pauseBtn = document.getElementById('pause-btn');
#         const prevBtn = document.getElementById('prev-btn');
#         const nextBtn = document.getElementById('next-btn');
#         const resetBtn = document.getElementById('reset-btn');
#         const speedControl = document.getElementById('speed-control');

#         // Function to display a model
#         function displayModel(index) {
#             try {
#                 if (!viewer) {
#                     showStatus("Viewer not initialized", true);
#                     return;
#                 }

#                 // Clear current model
#                 viewer.clear();
                
#                 // Add new model
#                 viewer.addModel(modelData[index], 'pdb');
                
#                 // Style the model
#                 viewer.setStyle({cartoon: {color: 'spectrum'}});
                
#                 // Adjust view
#                 viewer.zoomTo();
#                 viewer.render();
                
#                 // Update UI
#                 currentModelIndex = index;
#                 modelSlider.value = index + 1;
#                 modelInfo.textContent = `Model ${index + 1} of ${totalModels}: ${modelTitles[index]}`;
                
#                 // Update button states
#                 prevBtn.disabled = (index === 0);
#                 nextBtn.disabled = (index === totalModels - 1);
                
#                 showStatus("");
#             } catch (error) {
#                 showStatus("Error displaying model: " + error.message, true);
#                 console.error("Error displaying model:", error);
#             }
#         }

#         // Function to initialize the viewer
#         function initViewer() {
#             try {
#                 showStatus("Initializing viewer...");
                
#                 // Update slider
#                 modelSlider.max = totalModels;
#                 modelSlider.disabled = false;
                
#                 // Enable controls
#                 playBtn.disabled = false;
#                 pauseBtn.disabled = true;
#                 resetBtn.disabled = false;
#                 speedControl.disabled = false;
                
#                 // Create viewer
#                 viewer = $3Dmol.createViewer($(viewerContainer), {
#                     backgroundColor: 'white'
#                 });
                
#                 if (!viewer) {
#                     throw new Error("Failed to create 3Dmol viewer");
#                 }
                
#                 // Display first model
#                 displayModel(0);
                
#                 showStatus("Viewer ready");
#             } catch (error) {
#                 showStatus("Error initializing viewer: " + error.message, true);
#                 console.error("Error initializing viewer:", error);
#             }
#         }

#         // Setup playback controls
#         function startPlayback() {
#             try {
#                 isPlaying = true;
#                 playBtn.disabled = true;
#                 pauseBtn.disabled = false;
                
#                 const speed = parseInt(speedControl.value);
                
#                 playInterval = setInterval(function() {
#                     let nextIndex = currentModelIndex + 1;
#                     if (nextIndex >= totalModels) {
#                         nextIndex = 0;
#                     }
#                     displayModel(nextIndex);
#                 }, speed);
#             } catch (error) {
#                 showStatus("Error starting playback: " + error.message, true);
#                 stopPlayback();
#             }
#         }

#         function stopPlayback() {
#             isPlaying = false;
#             playBtn.disabled = false;
#             pauseBtn.disabled = true;
            
#             if (playInterval) {
#                 clearInterval(playInterval);
#                 playInterval = null;
#             }
#         }

#         // Attach event listeners
#         modelSlider.addEventListener('input', function() {
#             const index = parseInt(this.value) - 1;
#             if (isPlaying) {
#                 stopPlayback();
#             }
#             displayModel(index);
#         });

#         playBtn.addEventListener('click', startPlayback);
#         pauseBtn.addEventListener('click', stopPlayback);

#         prevBtn.addEventListener('click', function() {
#             if (currentModelIndex > 0) {
#                 if (isPlaying) {
#                     stopPlayback();
#                 }
#                 displayModel(currentModelIndex - 1);
#             }
#         });

#         nextBtn.addEventListener('click', function() {
#             if (currentModelIndex < totalModels - 1) {
#                 if (isPlaying) {
#                     stopPlayback();
#                 }
#                 displayModel(currentModelIndex + 1);
#             }
#         });

#         resetBtn.addEventListener('click', function() {
#             if (isPlaying) {
#                 stopPlayback();
#             }
#             displayModel(0);
#         });

#         speedControl.addEventListener('change', function() {
#             if (isPlaying) {
#                 stopPlayback();
#                 startPlayback();
#             }
#         });

#         // Initialize when the page loads
#         window.onload = function() {
#             showStatus("Page loaded, starting initialization");
            
#             // Small delay to ensure DOM is fully loaded
#             setTimeout(function() {
#                 if (modelData.length === 0) {
#                     showStatus("No models found in the PDB file", true);
#                 } else {
#                     initViewer();
#                 }
#             }, 500);
#         };
#     </script>
# </body>
# </html>
#     """
    
#     # Write to file
#     with open(output_html, 'w') as f:
#         f.write(html_content)
    
#     print(f"Trajectory viewer created at: {output_html}")
#     return output_html

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Create a PDB trajectory viewer.')
#     parser.add_argument('pdb_file', help='Path to the PDB file with multiple models')
#     parser.add_argument('--output', '-o', default='trajectory_viewer.html', 
#                         help='Output HTML file path (default: trajectory_viewer.html)')
    
#     args = parser.parse_args()
    
#     create_trajectory_viewer(args.pdb_file, args.output)


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
        list: A list of models, where each model is a string of PDB content
    """
    # Read the entire file content
    with open(pdb_file, 'r') as f:
        content = f.read()
    
    # Check if the file has MODEL/ENDMDL tags
    if 'MODEL' in content and 'ENDMDL' in content:
        # Split by MODEL/ENDMDL
        pattern = r'(MODEL\s+\d+.*?ENDMDL)'
        models_raw = re.findall(pattern, content, re.DOTALL)
        
        if not models_raw:  # If regex failed, fallback to simpler approach
            models_raw = []
            current_model = []
            for line in content.splitlines():
                if line.startswith('MODEL'):
                    if current_model:  # If we have collected lines for a model
                        models_raw.append('\n'.join(current_model))
                    current_model = [line]
                elif line.startswith('ENDMDL'):
                    current_model.append(line)
                    models_raw.append('\n'.join(current_model))
                    current_model = []
                elif current_model:  # Only append if we're within a model
                    current_model.append(line)
            
            # Add the last model if it wasn't terminated with ENDMDL
            if current_model:
                models_raw.append('\n'.join(current_model))
    else:
        # Treat the entire file as a single model
        models_raw = [content]
    
    # Extract titles if present
    model_titles = []
    for model in models_raw:
        title_match = re.search(r'TITLE\s+(.*?)(?:\n|$)', model)
        if title_match:
            model_titles.append(title_match.group(1).strip())
        else:
            model_titles.append(f"Model {len(model_titles) + 1}")
    
    # If no titles were extracted, create default ones
    if not model_titles or len(model_titles) < len(models_raw):
        model_titles = [f"Model {i+1}" for i in range(len(models_raw))]
    
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
    
    print(f"Found {len(models)} models in the PDB file")
    
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
    </div>

    <script>
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
                    viewer.zoomTo();
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
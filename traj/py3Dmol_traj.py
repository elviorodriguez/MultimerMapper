
import re
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
    <meta charset="UTF-8">
    <title>PDB Trajectory Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/ffmpeg/0.11.6/ffmpeg.min.js"></script>
    <style>
        h3 {
            text-align: center;
            margin-block-start: 0em;
            margin-block-end: 0em;
        }
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 5px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh; 
        }
        .container {
            width: 100%;
            max-width: 900px;
            margin: 0 auto;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            padding: 20px;
            border-radius: 8px;
        }
        .viewer-container {
            width: 100%;
            height: 600px;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        /* Ensure the 3Dmol viewer takes full width and height of container */
        .viewer-container canvas {
            width: 100% !important;
            height: 100% !important;
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
            font-size: 14px;
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
        <h3>PDB Trajectory Viewer</h3>
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
            <button id="prev-btn" disabled>&lt; Prev.</button>
            <button id="play-btn" disabled>Play</button>
            <button id="play-reverse-btn" disabled>Reverse</button>
            <button id="pause-btn" disabled>Pause</button>
            <button id="next-btn" disabled>Next &gt;</button>
            <button id="reset-btn" disabled>Reset</button>
            <button id="save-gif-btn" disabled>Save Video</button>
            <div class="speed-control">
                <label for="speed-control">Speed:</label>
                <select id="speed-control" disabled>
                    <option value="2000">Very Slow</option>
                    <option value="1000">Slow</option>
                    <option value="500">Normal</option>
                    <option value="250" selected>Fast</option>
                    <option value="100" >Very Fast</option>
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
        let isPlayingReverse = false;
        let playInterval = null;
        let isInitialLoad = true;

        // Get UI elements
        const viewerContainer = document.getElementById('viewer-container');
        const modelSlider = document.getElementById('model-slider');
        const modelInfo = document.getElementById('model-info');
        const playBtn = document.getElementById('play-btn');
        const playReverseBtn = document.getElementById('play-reverse-btn');
        const pauseBtn = document.getElementById('pause-btn');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const resetBtn = document.getElementById('reset-btn');
        const saveGifBtn = document.getElementById('save-gif-btn');
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
                    
                    // Adjust view
                    viewer.addModel(modelData[index], 'pdb');
                    applyCurrentStyle();

                    // Adjust view to fit the model properly (only on very first load)
                    if (isInitialLoad) {
                        viewer.zoomTo();
                        isInitialLoad = false;
                    }
                    
                    // Render
                    viewer.render();
                    
                    // Update UI
                    currentModelIndex = index;
                    modelSlider.value = index + 1;
                    modelInfo.innerHTML = `Model ${index + 1} of ${totalModels}<br>${modelTitles[index]}`;
                    
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
                    antialias: true,
                    disableFog: true,
                    cameraNear: 0.1,
                    cameraFar: 1000
                };
                
                if ($(viewerContainer).length > 0) {
                    // First create a clear container
                    $(viewerContainer).empty();
                    
                    viewer = $3Dmol.createViewer($(viewerContainer), config);
                    
                    if (viewer) {
                        // Enable controls after successful viewer creation
                        playBtn.disabled = false;
                        playReverseBtn.disabled = false;
                        pauseBtn.disabled = true;
                        resetBtn.disabled = false;
                        saveGifBtn.disabled = false;
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
        function startPlayback(reverse = false) {
            try {
                isPlaying = true;
                isPlayingReverse = reverse;
                playBtn.disabled = true;
                playReverseBtn.disabled = true;
                pauseBtn.disabled = false;
                
                const speed = parseInt(speedControl.value);
                
                playInterval = setInterval(function() {
                    let nextIndex;
                    if (reverse) {
                        nextIndex = currentModelIndex - 1;
                        if (nextIndex < 0) {
                            nextIndex = totalModels - 1;
                        }
                    } else {
                        nextIndex = currentModelIndex + 1;
                        if (nextIndex >= totalModels) {
                            nextIndex = 0;
                        }
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
            isPlayingReverse = false;
            playBtn.disabled = false;
            playReverseBtn.disabled = false;
            pauseBtn.disabled = true;
            
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
            }
        }

        // Function to save Video
        async function saveGIF() {
            try {
                // Ask user for filename
                const filename = prompt("Enter filename for the video (without extension):", "trajectory_animation");
                if (!filename) {
                    showStatus("Video creation cancelled");
                    return;
                }
                
                showStatus("Preparing video creation...");
                saveGifBtn.disabled = true;
                
                // Get current speed setting for frame rate calculation
                const speed = parseInt(speedControl.value);
                const fps = Math.max(1, Math.min(30, Math.round(1000 / speed))); // Convert speed to reasonable FPS
                
                // Capture all frames first
                const frames = [];
                const originalIndex = currentModelIndex; // Save current model
                
                for (let i = 0; i < totalModels; i++) {
                    showStatus(`Capturing frame ${i + 1} of ${totalModels}...`);
                    
                    displayModel(i);
                    
                    // Wait for model to render
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    
                    try {
                        const canvas = viewer.getCanvas();
                        if (canvas) {
                            // Convert canvas to data URL (simpler approach)
                            const dataURL = canvas.toDataURL('image/png');
                            frames.push(dataURL);
                        }
                    } catch (e) {
                        showStatus("Error capturing frame " + (i + 1) + ": " + e.message, true);
                    }
                }
                
                if (frames.length === 0) {
                    throw new Error("No frames captured");
                }
                
                showStatus("Creating video...");
                
                // Create video using frames
                await createVideo(frames, filename, fps);
                
                showStatus("Video saved successfully!");
                
                // Restore original model
                displayModel(originalIndex);
                
            } catch (error) {
                showStatus("Error creating video: " + error.message, true);
            } finally {
                saveGifBtn.disabled = false;
            }
        }

        // Helper function to create video from frames
        async function createVideo(frames, filename, fps) {
            try {
                // Use MediaRecorder approach since frames are data URLs
                await createVideoWithMediaRecorder(frames, filename, fps);
            } catch (error) {
                showStatus("Video creation failed, downloading frames as ZIP...", true);
                // Fallback to original ZIP method
                await createFramesZip(frames, filename);
            }
        }

        // MediaRecorder implementation using data URLs
        async function createVideoWithMediaRecorder(frames, filename, fps) {
            return new Promise(async (resolve, reject) => {
                try {
                    // Create a temporary canvas to draw frames
                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    
                    // Set canvas size based on first frame
                    const firstImg = new Image();
                    firstImg.src = frames[0];
                    await new Promise(imgResolve => {
                        firstImg.onload = imgResolve;
                        firstImg.onerror = () => reject(new Error("Failed to load first frame"));
                    });
                    
                    canvas.width = firstImg.width;
                    canvas.height = firstImg.height;
                    
                    const stream = canvas.captureStream(fps);
                    const recorder = new MediaRecorder(stream, {
                        mimeType: 'video/webm;codecs=vp8'
                    });
                    
                    const chunks = [];
                    recorder.ondataavailable = (e) => chunks.push(e.data);
                    
                    recorder.onstop = () => {
                        const blob = new Blob(chunks, { type: 'video/webm' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = filename.endsWith('.webm') ? filename : filename + '.webm';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                        resolve();
                    };
                    
                    recorder.onerror = (e) => reject(e);
                    
                    recorder.start();
                    
                    const frameDuration = 1000 / fps;
                    for (let i = 0; i < frames.length; i++) {
                        const img = new Image();
                        img.src = frames[i];
                        await new Promise(imgResolve => {
                            img.onload = imgResolve;
                            img.onerror = () => reject(new Error(`Failed to load frame ${i + 1}`));
                        });
                        
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(img, 0, 0);
                        
                        await new Promise(frameResolve => setTimeout(frameResolve, frameDuration));
                    }
                    
                    recorder.stop();
                    
                } catch (error) {
                    reject(error);
                }
            });
        }


        // Helper function to create ZIP with frames
        async function createFramesZip(frames, filename) {
            try {
                // Load JSZip library
                if (typeof JSZip === 'undefined') {
                    await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js');
                }
                
                const zip = new JSZip();
                
                // Add each frame to the zip
                for (let i = 0; i < frames.length; i++) {
                    const frameNumber = String(i + 1).padStart(3, '0');
                    const base64Data = frames[i].split(',')[1]; // Remove data:image/png;base64,
                    zip.file(`frame_${frameNumber}.png`, base64Data, {base64: true});
                }
                
                // Create instructions file
                const instructions = `Instructions for creating GIF:

        1. Extract all PNG files from this ZIP
        2. Use an online GIF maker like:
        - https://ezgif.com/maker
        - https://giphy.com/create
        - Or use ImageMagick: convert -delay 25 -loop 0 frame_*.png animation.gif

        Total frames: ${frames.length}
        Recommended delay: 250ms between frames
        `;
                
                zip.file('README.txt', instructions);
                
                // Generate ZIP and download
                const content = await zip.generateAsync({type: 'blob'});
                const url = URL.createObjectURL(content);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename.endsWith('.zip') ? filename : filename + '_frames.zip';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                
            } catch (error) {
                // Fallback: download frames individually
                showStatus("ZIP creation failed, downloading frames individually...");
                
                for (let i = 0; i < frames.length; i++) {
                    const frameNumber = String(i + 1).padStart(3, '0');
                    const a = document.createElement('a');
                    a.href = frames[i];
                    a.download = `${filename}_frame_${frameNumber}.png`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    
                    // Small delay between downloads
                    await new Promise(resolve => setTimeout(resolve, 100));
                }
            }
        }

        // Helper function to load scripts
        function loadScript(src) {
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = src;
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }

        // Attach event listeners
        modelSlider.addEventListener('input', function() {
            const index = parseInt(this.value) - 1;
            if (isPlaying) {
                stopPlayback();
            }
            displayModel(index);
        });

        playBtn.addEventListener('click', () => startPlayback(false));
        playReverseBtn.addEventListener('click', () => startPlayback(true));
        pauseBtn.addEventListener('click', stopPlayback);
        saveGifBtn.addEventListener('click', saveGIF);

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
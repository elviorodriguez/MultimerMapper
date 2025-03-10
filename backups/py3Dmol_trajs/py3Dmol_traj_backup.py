# #!/usr/bin/env python3
# """
# PDB Trajectory Visualizer

# This module reads multi-model PDB files (like MD trajectories) and generates
# an interactive HTML py3Dmol visualization directly without intermediate steps.

# Usage:
#     python py3Dmol_traj.py input.pdb [--output output.html] [--width 800] [--height 600] [--speed 1.0]

# Author: Claude
# Date: March 9, 2025
# """

# import os
# import sys
# import argparse
# import numpy as np
# from collections import defaultdict

# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description='Visualize PDB trajectory with py3Dmol')
#     parser.add_argument('pdb_file', help='Input multi-model PDB file')
#     parser.add_argument('--output', help='Output HTML file path')
#     parser.add_argument('--width', type=int, default=800, help='Visualization width (default: 800)')
#     parser.add_argument('--height', type=int, default=600, help='Visualization height (default: 600)')
#     parser.add_argument('--speed', type=float, default=1.0, help='Animation speed multiplier (default: 1.0)')
#     return parser.parse_args()

# def create_visualization(pdb_file, output_path, width=800, height=600, speed=1.0):
#     """
#     Create an HTML file with py3Dmol visualization directly from the PDB file.
    
#     Args:
#         pdb_file (str): Path to the multi-model PDB file
#         output_path (str): Path to save the HTML file
#         width (int): Visualization width
#         height (int): Visualization height
#         speed (float): Animation speed multiplier
#     """
#     print(f"Creating visualization from {pdb_file}...")
    
#     # Read the PDB file directly - no need for intermediate processing
#     with open(pdb_file, 'r') as f:
#         pdb_content = f.read()

#     # Generate HTML with embedded PDB data
#     html_content = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>PDB Trajectory Visualization</title>
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
#         <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js"></script>
#         <style>
#             body {{ font-family: Arial, sans-serif; margin: 20px; }}
#             .container {{ margin: 0 auto; max-width: 1000px; }}
#             .controls {{ margin-top: 20px; }}
#             button {{ padding: 10px; margin-right: 10px; cursor: pointer; }}
#             .spinner {{ display: none; width: 40px; height: 40px; position: absolute; top: 50%; left: 50%; 
#                       margin-top: -20px; margin-left: -20px; border: 4px solid #f3f3f3; 
#                       border-top: 4px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite; }}
#             @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
#             .legend {{ margin-top: 15px; display: flex; align-items: center; }}
#             .legend-item {{ display: flex; align-items: center; margin-right: 15px; }}
#             .legend-color {{ width: 20px; height: 20px; margin-right: 5px; }}
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <h2>PDB Trajectory Visualization</h2>
#             <div id="viewer" style="width: {width}px; height: {height}px; position: relative;">
#                 <div class="spinner" id="loading"></div>
#             </div>
            
#             <div class="legend">
#                 <div class="legend-item"><div class="legend-color" style="background: blue;"></div>Very high (90-100)</div>
#                 <div class="legend-item"><div class="legend-color" style="background: cyan;"></div>High (70-90)</div>
#                 <div class="legend-item"><div class="legend-color" style="background: green;"></div>Medium (50-70)</div>
#                 <div class="legend-item"><div class="legend-color" style="background: yellow;"></div>Low (0-50)</div>
#             </div>
            
#             <div class="controls">
#                 <button id="play">Play</button>
#                 <button id="pause">Pause</button>
#                 <button id="reset">Reset</button>
#                 <button id="prev">Previous</button>
#                 <button id="next">Next</button>
#                 <span style="margin-left: 20px;">Speed: </span>
#                 <input type="range" id="speed" min="1" max="10" value="{int(speed*5)}" />
#                 <span style="margin-left: 20px;">Frame: </span>
#                 <span id="frameCounter">1</span>
#             </div>
#         </div>
        
#         <script>
#             // Store PDB data as a string directly in the page
#             const pdbData = `{pdb_content}`;
            
#             $(document).ready(function() {{
#                 $('#loading').show();
                
#                 // Create 3Dmol viewer
#                 let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
                
#                 // Add models as frames - this happens asynchronously
#                 viewer.addModelsAsFrames(pdbData, "pdb");
                
#                 // Style with AlphaFold pLDDT color scheme
#                 viewer.setStyle({{}}, {{
#                     cartoon: {{
#                         colorscheme: {{
#                             prop: 'b',
#                             gradient: 'roygb',
#                             min: 0,
#                             max: 100
#                         }}
#                     }}
#                 }});
                
#                 // Setup view
#                 viewer.zoomTo();
#                 viewer.render();
                
#                 // Hide loading spinner
#                 $('#loading').hide();
                
#                 // Animation settings
#                 let animationOptions = {{
#                     'loop': 'forward',
#                     'reps': 0,
#                     'step': {int(5 / speed)}
#                 }};
                
#                 // Frame counter
#                 let currentFrame = 0;
#                 let totalFrames = viewer.getNumFrames();
                
#                 function updateFrameCounter() {{
#                     $('#frameCounter').text((currentFrame + 1) + " / " + totalFrames);
#                 }}
                
#                 // Event listener for frame changes
#                 viewer.addFrameCallback(function(frameNum) {{
#                     currentFrame = frameNum;
#                     updateFrameCounter();
#                 }});
                
#                 updateFrameCounter();
                
#                 // Controls
#                 $("#play").click(function() {{
#                     viewer.animate(animationOptions);
#                 }});
                
#                 $("#pause").click(function() {{
#                     viewer.pause();
#                 }});
                
#                 $("#reset").click(function() {{
#                     viewer.stop();
#                     viewer.setFrame(0);
#                     viewer.render();
#                 }});
                
#                 $("#prev").click(function() {{
#                     viewer.stop();
#                     let newFrame = (currentFrame - 1 + totalFrames) % totalFrames;
#                     viewer.setFrame(newFrame);
#                     viewer.render();
#                 }});
                
#                 $("#next").click(function() {{
#                     viewer.stop();
#                     let newFrame = (currentFrame + 1) % totalFrames;
#                     viewer.setFrame(newFrame);
#                     viewer.render();
#                 }});
                
#                 $("#speed").on("input", function() {{
#                     let speed = $(this).val() / 5;
#                     animationOptions.step = Math.round(5 / speed);
                    
#                     // If animation is running, restart with new speed
#                     if (!viewer.isAnimated()) return;
#                     viewer.stop();
#                     viewer.animate(animationOptions);
#                 }});
                
#                 // Start animation by default
#                 viewer.animate(animationOptions);
#             }});
#         </script>
#     </body>
#     </html>
#     """
    
#     # Save HTML file
#     with open(output_path, 'w') as f:
#         f.write(html_content)
    
#     print(f"Visualization saved to {output_path}")
#     print(f"Open this file in a web browser to view the animation.")

# def main():
#     """Main function to run the module from the command line."""
#     args = parse_args()
    
#     # Set the output file path
#     if args.output:
#         output_path = args.output
#     else:
#         base_name = os.path.splitext(os.path.basename(args.pdb_file))[0]
#         output_path = f"{base_name}_visualization.html"
    
#     # Validate input file
#     if not os.path.exists(args.pdb_file):
#         print(f"Error: Input file '{args.pdb_file}' not found.")
#         sys.exit(1)
    
#     # Check if the file is a PDB file
#     if not args.pdb_file.lower().endswith('.pdb'):
#         print(f"Warning: Input file '{args.pdb_file}' does not have a .pdb extension.")
    
#     # Process the PDB file
#     try:
#         create_visualization(
#             args.pdb_file, 
#             output_path, 
#             width=args.width, 
#             height=args.height, 
#             speed=args.speed
#         )
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
PDB Trajectory Visualizer

This module reads multi-model PDB files (like MD trajectories), aligns all models
to the first model, and generates an HTML py3Dmol visualization.

Usage:
    python py3Dmol_traj.py input.pdb [--output output.html] [--width 800] [--height 600] [--speed 1.0]
"""

import os
import sys
import argparse
import numpy as np
from io import StringIO
import re

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize PDB trajectory with py3Dmol')
    parser.add_argument('pdb_file', help='Input multi-model PDB file')
    parser.add_argument('--output', help='Output HTML file path')
    parser.add_argument('--width', type=int, default=800, help='Visualization width (default: 800)')
    parser.add_argument('--height', type=int, default=600, help='Visualization height (default: 600)')
    parser.add_argument('--speed', type=float, default=1.0, help='Animation speed multiplier (default: 1.0)')
    return parser.parse_args()

def parse_pdb_models(pdb_file):
    """
    Parse multiple models from a PDB file.
    
    Args:
        pdb_file (str): Path to the PDB file
        
    Returns:
        list: List of models, where each model is a list of atoms (dictionaries)
    """
    models = []
    current_model = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                current_model = []
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse relevant atom information
                atom = {
                    'record_type': line[0:6].strip(),
                    'atom_num': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'alt_loc': line[16:17].strip(),
                    'res_name': line[17:20].strip(),
                    'chain_id': line[21:22].strip(),
                    'res_num': int(line[22:26]),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]) if line[54:60].strip() else 1.0,
                    'temp_factor': float(line[60:66]) if line[60:66].strip() else 0.0,
                    'element': line[76:78].strip() if len(line) >= 78 else '',
                    'charge': line[78:80].strip() if len(line) >= 80 else '',
                    'line': line  # Store the original line for reconstruction
                }
                current_model.append(atom)
            elif line.startswith('ENDMDL'):
                if current_model:
                    models.append(current_model)
                    
    # If there's only one model without MODEL/ENDMDL markers
    if not models and current_model:
        models.append(current_model)
        
    print(f"Parsed {len(models)} models from PDB file.")
    return models

def get_ca_atoms(model):
    """
    Extract CA atoms from a model for alignment.
    
    Args:
        model (list): List of atom dictionaries
        
    Returns:
        list: List of CA atom dictionaries
    """
    return [atom for atom in model if atom['atom_name'] == 'CA']

def compute_centroid(atoms):
    """
    Compute the centroid of a list of atoms.
    
    Args:
        atoms (list): List of atom dictionaries
        
    Returns:
        tuple: (x, y, z) coordinates of the centroid
    """
    coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in atoms])
    return np.mean(coords, axis=0)

def kabsch_align(mobile_coords, reference_coords):
    """
    Perform Kabsch alignment to find the optimal rotation matrix.
    
    Args:
        mobile_coords (numpy.ndarray): Coordinates to be aligned
        reference_coords (numpy.ndarray): Reference coordinates
        
    Returns:
        tuple: (rotation_matrix, translation_vector)
    """
    # Calculate centroids
    mobile_centroid = np.mean(mobile_coords, axis=0)
    reference_centroid = np.mean(reference_coords, axis=0)
    
    # Center the coordinates
    mobile_centered = mobile_coords - mobile_centroid
    reference_centered = reference_coords - reference_centroid
    
    # Calculate the covariance matrix
    covariance = np.dot(mobile_centered.T, reference_centered)
    
    # Singular Value Decomposition
    u, s, vt = np.linalg.svd(covariance)
    
    # Calculate the rotation matrix
    rotation = np.dot(vt.T, u.T)
    
    # Ensure proper rotation (no reflection)
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = np.dot(vt.T, u.T)
    
    # Calculate the translation
    translation = reference_centroid - np.dot(mobile_centroid, rotation)
    
    return rotation, translation

def align_models(models):
    """
    Align all models to the first model.
    
    Args:
        models (list): List of models, where each model is a list of atoms
        
    Returns:
        list: List of aligned models
    """
    if not models:
        return []
        
    # Use the first model as reference
    reference_model = models[0]
    reference_ca_atoms = get_ca_atoms(reference_model)
    
    if not reference_ca_atoms:
        print("Warning: No CA atoms found in reference model. Using all atoms.")
        reference_ca_atoms = reference_model
    
    reference_coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in reference_ca_atoms])
    
    aligned_models = [reference_model]  # The reference model is already "aligned"
    
    for i, model in enumerate(models[1:], 1):
        print(f"Aligning model {i+1}/{len(models)}...", end='\r')
        
        # Get CA atoms for the current model
        mobile_ca_atoms = get_ca_atoms(model)
        
        if not mobile_ca_atoms:
            print(f"\nWarning: No CA atoms found in model {i+1}. Using all atoms.")
            mobile_ca_atoms = model
        
        # Extract coordinates
        mobile_coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in mobile_ca_atoms])
        
        # Find common residues for alignment
        ref_res_ids = {(atom['chain_id'], atom['res_num']) for atom in reference_ca_atoms}
        mobile_res_ids = {(atom['chain_id'], atom['res_num']) for atom in mobile_ca_atoms}
        common_res_ids = ref_res_ids.intersection(mobile_res_ids)
        
        if len(common_res_ids) < 3:
            print(f"\nWarning: Less than 3 common residues found for model {i+1}. Skipping alignment.")
            aligned_models.append(model)
            continue
            
        # Filter to common residues
        ref_filtered = [atom for atom in reference_ca_atoms if (atom['chain_id'], atom['res_num']) in common_res_ids]
        mobile_filtered = [atom for atom in mobile_ca_atoms if (atom['chain_id'], atom['res_num']) in common_res_ids]
        
        # Sort by residue number to ensure correspondence
        ref_filtered.sort(key=lambda atom: (atom['chain_id'], atom['res_num']))
        mobile_filtered.sort(key=lambda atom: (atom['chain_id'], atom['res_num']))
        
        ref_coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in ref_filtered])
        mob_coords = np.array([[atom['x'], atom['y'], atom['z']] for atom in mobile_filtered])
        
        # Calculate alignment
        rotation, translation = kabsch_align(mob_coords, ref_coords)
        
        # Apply transformation to all atoms in the model
        aligned_model = []
        for atom in model:
            # Create a copy of the atom
            new_atom = atom.copy()
            
            # Apply rotation and translation
            coords = np.array([atom['x'], atom['y'], atom['z']])
            new_coords = np.dot(coords, rotation) + translation
            
            # Update coordinates
            new_atom['x'] = new_coords[0]
            new_atom['y'] = new_coords[1]
            new_atom['z'] = new_coords[2]
            
            # Update the line with new coordinates
            line = list(atom['line'])
            line[30:38] = f"{new_atom['x']:8.3f}"
            line[38:46] = f"{new_atom['y']:8.3f}"
            line[46:54] = f"{new_atom['z']:8.3f}"
            new_atom['line'] = ''.join(line)
            
            aligned_model.append(new_atom)
        
        aligned_models.append(aligned_model)
    
    print("\nAlignment complete.                 ")
    return aligned_models

def models_to_pdb_string(models):
    """
    Convert list of models to PDB string format.
    
    Args:
        models (list): List of aligned models
        
    Returns:
        str: PDB string with multiple models
    """
    pdb_lines = []
    
    for i, model in enumerate(models):
        pdb_lines.append(f"MODEL {i+1}")
        
        for atom in model:
            pdb_lines.append(atom['line'].rstrip())
            
        pdb_lines.append("ENDMDL")
    
    return '\n'.join(pdb_lines)

def create_visualization(pdb_string, output_path, width=800, height=600, speed=1.0):
    """
    Create an HTML file with py3Dmol visualization of the trajectory.
    
    Args:
        pdb_string (str): Multi-model PDB string
        output_path (str): Path to save the HTML file
        width (int): Visualization width
        height (int): Visualization height
        speed (float): Animation speed multiplier
    """
    # Generate HTML with embedded PDB data
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDB Trajectory Visualization</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ margin: 0 auto; max-width: 1000px; }}
            .controls {{ margin-top: 20px; }}
            button {{ padding: 10px; margin-right: 10px; cursor: pointer; }}
            .spinner {{ display: none; width: 40px; height: 40px; position: absolute; top: 50%; left: 50%; 
                      margin-top: -20px; margin-left: -20px; border: 4px solid #f3f3f3; 
                      border-top: 4px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite; }}
            @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
            .legend {{ margin-top: 15px; display: flex; align-items: center; }}
            .legend-item {{ display: flex; align-items: center; margin-right: 15px; }}
            .legend-color {{ width: 20px; height: 20px; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>PDB Trajectory Visualization</h2>
            <div id="viewer" style="width: {width}px; height: {height}px; position: relative;">
                <div class="spinner" id="loading"></div>
            </div>
            
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: blue;"></div>Very high (90-100)</div>
                <div class="legend-item"><div class="legend-color" style="background: cyan;"></div>High (70-90)</div>
                <div class="legend-item"><div class="legend-color" style="background: green;"></div>Medium (50-70)</div>
                <div class="legend-item"><div class="legend-color" style="background: yellow;"></div>Low (0-50)</div>
            </div>
            
            <div class="controls">
                <button id="play">Play</button>
                <button id="pause">Pause</button>
                <button id="reset">Reset</button>
                <button id="prev">Previous</button>
                <button id="next">Next</button>
                <span style="margin-left: 20px;">Speed: </span>
                <input type="range" id="speed" min="1" max="10" value="{int(speed*5)}" />
                <span style="margin-left: 20px;">Frame: </span>
                <span id="frameCounter">1</span>
            </div>
        </div>
        
        <script>
            // Create blob URL for the PDB data (more efficient for large files)
            const pdbBlob = new Blob([`{pdb_string}`], {{type: 'text/plain'}});
            const pdbUrl = URL.createObjectURL(pdbBlob);
            
            $(document).ready(function() {{
                $('#loading').show();
                
                // Create 3Dmol viewer
                let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
                
                // Load PDB data from blob URL
                $.get(pdbUrl, function(data) {{
                    // Add models as frames
                    viewer.addModelsAsFrames(data, "pdb");
                    
                    // Style with AlphaFold pLDDT color scheme
                    viewer.setStyle({{}}, {{
                        cartoon: {{
                            colorscheme: {{
                                prop: 'b',
                                gradient: 'roygb',
                                min: 0,
                                max: 100
                            }}
                        }}
                    }});
                    
                    // Setup view
                    viewer.zoomTo();
                    viewer.render();
                    
                    // Hide loading spinner
                    $('#loading').hide();
                    
                    // Start animation
                    let animationOptions = {{
                        'loop': 'forward',
                        'reps': 0,
                        'step': {int(5 / speed)}
                    }};
                    viewer.animate(animationOptions);
                    
                    // Frame counter
                    let currentFrame = 0;
                    let totalFrames = viewer.getNumFrames();
                    
                    function updateFrameCounter() {{
                        $('#frameCounter').text((currentFrame + 1) + " / " + totalFrames);
                    }}
                    
                    // Event listener for frame changes
                    viewer.addFrameCallback(function(frameNum) {{
                        currentFrame = frameNum;
                        updateFrameCounter();
                    }});
                    
                    updateFrameCounter();
                    
                    // Controls
                    $("#play").click(function() {{
                        viewer.animate(animationOptions);
                    }});
                    
                    $("#pause").click(function() {{
                        viewer.pause();
                    }});
                    
                    $("#reset").click(function() {{
                        viewer.stop();
                        viewer.setFrame(0);
                        viewer.render();
                    }});
                    
                    $("#prev").click(function() {{
                        viewer.stop();
                        let newFrame = (currentFrame - 1 + totalFrames) % totalFrames;
                        viewer.setFrame(newFrame);
                        viewer.render();
                    }});
                    
                    $("#next").click(function() {{
                        viewer.stop();
                        let newFrame = (currentFrame + 1) % totalFrames;
                        viewer.setFrame(newFrame);
                        viewer.render();
                    }});
                    
                    $("#speed").on("input", function() {{
                        let speed = $(this).val() / 5;
                        animationOptions.step = Math.round(5 / speed);
                        
                        // If animation is running, restart with new speed
                        if (!viewer.isAnimated()) return;
                        viewer.stop();
                        viewer.animate(animationOptions);
                    }});
                }});
                
                // Clean up blob URL when page is unloaded
                $(window).on('unload', function() {{
                    URL.revokeObjectURL(pdbUrl);
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Save HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization saved to {output_path}")
    print(f"Open this file in a web browser to view the animation.")

def main():
    """Main function to run the module from the command line."""
    args = parse_args()
    
    # Set the output file path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.pdb_file))[0]
        output_path = f"{base_name}_visualization.html"
    
    # Validate input file
    if not os.path.exists(args.pdb_file):
        print(f"Error: Input file '{args.pdb_file}' not found.")
        sys.exit(1)
    
    # Process the PDB file
    try:
        print(f"Reading PDB file: {args.pdb_file}")
        models = parse_pdb_models(args.pdb_file)
        
        if not models:
            print("Error: No models found in the PDB file.")
            sys.exit(1)
            
        print(f"Aligning models to reference (model 1)...")
        aligned_models = align_models(models)
        
        print("Converting aligned models to PDB format...")
        pdb_string = models_to_pdb_string(aligned_models)
        
        print("Creating visualization...")
        create_visualization(
            pdb_string, 
            output_path, 
            width=args.width, 
            height=args.height, 
            speed=args.speed
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
import os
import re
import json
import html
import numpy as np
from Bio.PDB import PDBParser


def get_residue_centroid(residue):
    """Calculate the centroid of a residue's atoms."""
    coords = []
    for atom in residue:
        coords.append(atom.get_coord())
    if coords:
        return np.mean(coords, axis=0)
    return None


def get_backbone_atom_coord(residue):
    """Get the coordinate of the backbone atom (CA for amino acids)."""
    if 'CA' in residue:
        return residue['CA'].get_coord()
    elif 'C1' in residue:  # For nucleotides
        return residue['C1'].get_coord()
    return None


def parse_contact_matrix(contact_matrix, chains_in_model, L1, L2):
    """
    Parse contact matrix to extract residue pairs and their contact frequencies.
    
    Args:
        contact_matrix: 2D numpy array with contact frequencies
        chains_in_model: List of chain IDs in the model
        L1: Length of first chain
        L2: Length of second chain
    
    Returns:
        List of tuples: (chain1, res1, chain2, res2, frequency)
    """
    contacts = []
    
    # Define which chain is which
    n_rows, n_cols = contact_matrix.shape
    if n_rows == L1 and n_cols == L2:
        chain_1_idx = 0
        chain_2_idx = 1
    elif n_rows == L2 and n_cols == L1:
        chain_1_idx = 1
        chain_2_idx = 0
    else:
        print("ERROR!!! Chains Lengths do not match! Assigning indexes 0 to chain 1 and 1 to chain 2")
        chain_1_idx = 0
        chain_2_idx = 1
    
    # Check if this is a homodimer (same chain IDs) or heterodimer
    is_homodimer = (len(set(chains_in_model)) == 1) or (chains_in_model[0] == chains_in_model[1])
    
    # Find all non-zero contacts
    rows, cols = np.where(contact_matrix > 0)
    
    for i, j in zip(rows, cols):
        frequency = contact_matrix[i, j]
        
        # For homodimers, only take upper triangle to avoid duplicates
        # For heterodimers, take all contacts since matrix is not symmetric
        if is_homodimer and i >= j:
            continue
        
        # Map matrix indices to chain and residue numbers
        # Row index corresponds to chain_1_idx, column index to chain_2_idx
        chain1, res1 = map_index_to_chain_residue(i, chains_in_model, chain_1_idx)
        chain2, res2 = map_index_to_chain_residue(j, chains_in_model, chain_2_idx)
        
        contacts.append((chain1, res1, chain2, res2, frequency))
    
    return contacts


def map_index_to_chain_residue(index, chains_in_model, chain_idx):
    """
    Map matrix index to chain and residue number.
    """
    chain = chains_in_model[chain_idx]
    residue = index + 1
    return chain, residue


def create_contact_visualization(pdb_file, contact_matrix, chains_in_model, output_html, 
                               protein1_name, protein2_name, cluster_id, logger):
    """
    Create an HTML visualization for protein contacts.
    
    Args:
        pdb_file (str): Path to the PDB file
        contact_matrix (numpy.ndarray): Contact frequency matrix
        chains_in_model (list): List of chain IDs in the model
        output_html (str): Path for the output HTML file
        protein1_name (str): Name of protein 1
        protein2_name (str): Name of protein 2
        cluster_id (int): Cluster identifier
    """
    
    # Read PDB file content
    with open(pdb_file, 'r') as f:
        pdb_content = f.read()
    
    # Parse PDB to get residue information
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]

    # Get chain lengths
    chain_lengths = {}
    for chain_idx, chain in enumerate(model):
        chain_id = chain.get_id()
        chain_lengths[chain_idx] = len(chain)

    # Parse contact matrix
    contacts = parse_contact_matrix(contact_matrix, chains_in_model, L1 = chain_lengths[0], L2 = chain_lengths[1])

    # Identify all residues involved in contacts
    contact_residues = set()
    for chain1, res1, chain2, res2, _ in contacts:
        contact_residues.add((str(chain1), int(res1)))
        contact_residues.add((str(chain2), int(res2)))
    
    # Extract centroid and backbone information
    centroids_data = []
    backbone_data = []
    hex_palette = [
        "#DA70D6",  # Orchid
        "#ADFF2F",  # Green Yellow
        "#646464",  # Python Dove Gray
        "#4584B6",  # Python Steel Blue
        "#FFDE57",  # Python Mustard
        "#8A2BE2",  # Blue Violet
        "#FF7F50",  # Coral
        "#20B2AA",  # Light Sea Green
        "#FFD700",  # Gold
        "#FF4500"   # Orange Red
    ]
    
    for chain_idx, chain in enumerate(model):
        chain_id = chain.get_id()
        for residue in chain:
            res_id = residue.get_id()[1]
            
            # Add centroids only for contact residues
            if (chain_id, res_id) in contact_residues:
                # Get centroid
                centroid = get_residue_centroid(residue)
                if centroid is not None:
                    centroids_data.append({
                        'chain': str(chain_id),  # Ensure string
                        'residue': int(res_id),  # Ensure native int
                        'x': float(centroid[0]),
                        'y': float(centroid[1]),
                        'z': float(centroid[2]),
                        'color': hex_palette[chain_idx]
                    })
                
                # Get backbone atom
                backbone_coord = get_backbone_atom_coord(residue)
                if backbone_coord is not None:
                    backbone_data.append({
                        'chain': str(chain_id),  # Ensure string
                        'residue': int(res_id),  # Ensure native int
                        'x': float(backbone_coord[0]),
                        'y': float(backbone_coord[1]),
                        'z': float(backbone_coord[2]),
                        'color': hex_palette[chain_idx]
                    })
    
    # Prepare contacts data for JavaScript with explicit type conversion
    contacts_data = []
    for chain1, res1, chain2, res2, frequency in contacts:
        # Find centroids for both residues
        centroid1 = next((c for c in centroids_data if c['chain'] == str(chain1) and c['residue'] == int(res1)), None)
        centroid2 = next((c for c in centroids_data if c['chain'] == str(chain2) and c['residue'] == int(res2)), None)
        
        if centroid1 and centroid2:
            contacts_data.append({
                'chain1': str(chain1),
                'res1': int(res1),
                'chain2': str(chain2),
                'res2': int(res2),
                'frequency': float(frequency),  # Convert to native float
                'x1': float(centroid1['x']),
                'y1': float(centroid1['y']),
                'z1': float(centroid1['z']),
                'x2': float(centroid2['x']),
                'y2': float(centroid2['y']),
                'z2': float(centroid2['z'])
            })
    
    # print(f"Contacts data length: {len(contacts_data)}")
    
    # Convert all data to JSON-serializable format
    def convert_to_json_serializable(obj):
        """Convert numpy types to native Python types recursively"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    # Ensure all data is JSON serializable
    centroids_data = convert_to_json_serializable(centroids_data)
    backbone_data = convert_to_json_serializable(backbone_data)
    contacts_data = convert_to_json_serializable(contacts_data)
    
    # Generate HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Protein Contact Visualization - {protein1_name} vs {protein2_name} (Cluster {cluster_id})</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 5px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 5px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h3 {{
            text-align: center;
            margin-block-start: 0em;
            margin-block-end: 0em;
        }}
        .viewer-container {{
            width: 800px;
            height: 600px;
            margin: 10px auto;
            border: 2px solid #ddd;
            border-radius: 8px;
            position: relative;
            background-color: #000;
        }}
        .controls {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }}
        .control-group label {{
            font-weight: bold;
            color: #555;
        }}
        button {{
            padding: 10px 15px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s;
        }}
        button:hover {{
            background-color: #0056b3;
        }}
        button:disabled {{
            background-color: #6c757d;
            cursor: not-allowed;
        }}
        select {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        .toggle-button {{
            background-color: #28a745;
        }}
        .toggle-button:hover {{
            background-color: #218838;
        }}
        .toggle-button.active {{
            background-color: #dc3545;
        }}
        .toggle-button.active:hover {{
            background-color: #c82333;
        }}
        .info-panel {{
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 10px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h3>{protein1_name} vs {protein2_name} - Cluster {cluster_id}</h3>
        
        <div class="viewer-container" id="viewer-container"></div>
        
        <div class="controls">
            <div class="control-group">
                <label>Protein Style:</label>
                <select id="style-select">
                    <option value="cartoon">Cartoon</option>
                    <option value="line">Line</option>
                    <option value="stick">Stick</option>
                    <option value="sphere">Sphere</option>
                    <option value="surface">Surface</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Color Scheme:</label>
                <select id="color-select">
                    <option value="chain">Chain</option>
                    <option value="spectrum">Spectrum</option>
                    <option value="residue">Residue</option>
                    <option value="secondary">Secondary Structure</option>
                    <option value="plddt">pLDDT</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Surface Options:</label>
                <select id="surface-select">
                    <option value="none">None</option>
                    <option value="VDW">Van der Waals</option>
                    <option value="SAS">Solvent Accessible</option>
                    <option value="MS">Molecular Surface</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Contact Features:</label>
                <button id="centroids-toggle" class="toggle-button">Show Centroids</button>
                <button id="contacts-toggle" class="toggle-button">Show Contacts</button>
            </div>
        </div>
        
        <div class="info-panel">
            <div class="stats">
                <div class="stat">
                    <div class="stat-value" id="total-residues">-</div>
                    <div class="stat-label">Total Residues<br>in contact</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="total-contacts">-</div>
                    <div class="stat-label">Total Contacts</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="avg-frequency">-</div>
                    <div class="stat-label">Avg. Contact<br>Frequency</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data from Python
        const pdbData = `{pdb_content}`;
        const centroidsData = {json.dumps(centroids_data)};
        const backboneData = {json.dumps(backbone_data)};
        const contactsData = {json.dumps(contacts_data)};
        
        // Global variables
        let viewer = null;
        let centroidsVisible = false;
        let contactsVisible = false;
        
        // pLDDT color scale
        const plddt_colorscale = [
            [0.0, "#FF0000"],
            [0.4, "#FFA500"],
            [0.6, "#FFFF00"],
            [0.8, "#ADD8E6"],
            [1.0, "#00008B"]
        ];
        
        function getColorFromScale(value, scale) {{
            let lowerIndex = 0;
            for (let i = 0; i < scale.length; i++) {{
                if (value <= scale[i][0]) {{
                    break;
                }}
                lowerIndex = i;
            }}
            
            if (lowerIndex >= scale.length - 1) {{
                return scale[scale.length - 1][1];
            }}
            
            const lowerValue = scale[lowerIndex][0];
            const upperValue = scale[lowerIndex + 1][0];
            const valueFraction = (value - lowerValue) / (upperValue - lowerValue);
            
            const lowerColor = hexToRgb(scale[lowerIndex][1]);
            const upperColor = hexToRgb(scale[lowerIndex + 1][1]);
            
            const r = Math.round(lowerColor.r + valueFraction * (upperColor.r - lowerColor.r));
            const g = Math.round(lowerColor.g + valueFraction * (upperColor.g - lowerColor.g));
            const b = Math.round(lowerColor.b + valueFraction * (upperColor.b - lowerColor.b));
            
            return rgbToHex(r, g, b);
        }}
        
        function hexToRgb(hex) {{
            const result = /^#?([a-f\\d]{{2}})([a-f\\d]{{2}})([a-f\\d]{{2}})$/i.exec(hex);
            return result ? {{
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16)
            }} : {{r: 0, g: 0, b: 0}};
        }}
        
        function rgbToHex(r, g, b) {{
            return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
        }}
        
        function initViewer() {{
            const config = {{
                backgroundColor: 'white',
                antialias: true
            }};
            
            viewer = $3Dmol.createViewer($("#viewer-container"), config);
            viewer.addModel(pdbData, 'pdb');
            
            applyCurrentStyle();
            updateStats();
            
            viewer.zoomTo();
            viewer.render();
        }}
        
        function applyCurrentStyle() {{
            if (!viewer) return;
            
            const style = document.getElementById('style-select').value;
            const colorScheme = document.getElementById('color-select').value;
            const surfaceType = document.getElementById('surface-select').value;
            
            // Clear current styles
            viewer.setStyle({{}}, {{}});
            
            // Apply main style
            let styleObj = {{}};
            
            if (colorScheme === 'chain') {{
                styleObj[style] = {{colorscheme: 'chainHetatm'}};
            }} else if (colorScheme === 'spectrum') {{
                styleObj[style] = {{color: 'spectrum'}};
            }} else if (colorScheme === 'residue') {{
                styleObj[style] = {{colorscheme: 'amino'}};
            }} else if (colorScheme === 'secondary') {{
                styleObj[style] = {{colorscheme: 'ssPyMOL'}};
            }} else if (colorScheme === 'plddt') {{
                styleObj[style] = {{colorfunc: function(atom) {{
                    const bfactor = Math.max(0, Math.min(100, atom.b));
                    const normalizedValue = bfactor / 100;
                    return getColorFromScale(normalizedValue, plddt_colorscale);
                }}}};
            }}
            
            viewer.setStyle({{}}, styleObj);
            
            // Clear everything and re-add model with surfaces
            viewer.removeAllSurfaces(); 

            // Apply surface if selected
            if (surfaceType !== 'none') {{
                viewer.addSurface(surfaceType, {{opacity: 0.8}});
            }}
            
            viewer.render();
        }}
        
        function toggleCentroids() {{
            centroidsVisible = !centroidsVisible;
            const button = document.getElementById('centroids-toggle');
            
            if (centroidsVisible) {{
                // Add centroids as spheres
                centroidsData.forEach(centroid => {{
                    viewer.addSphere({{
                        center: {{x: centroid.x, y: centroid.y, z: centroid.z}},
                        radius: 1.0,
                        color: centroid.color,
                        alpha: 1.0
                    }});
                }});
                
                // Add sticks from backbone to centroids
                backboneData.forEach(backbone => {{
                    const centroid = centroidsData.find(c => 
                        c.chain === backbone.chain && c.residue === backbone.residue
                    );
                    if (centroid) {{
                        viewer.addCylinder({{
                            start: {{x: backbone.x, y: backbone.y, z: backbone.z}},
                            end: {{x: centroid.x, y: centroid.y, z: centroid.z}},
                            radius: 0.3,
                            color: backbone.color,
                            alpha: 1.0
                        }});
                    }}
                }});
                
                button.textContent = 'Hide Centroids';
                button.classList.add('active');
            }} else {{
                // Remove centroids and sticks (requires re-rendering)
                viewer.removeAllShapes();
                if (contactsVisible) {{
                    showContacts();
                }}
                button.textContent = 'Show Centroids';
                button.classList.remove('active');
            }}
            
            viewer.render();
        }}
        
        function toggleContacts() {{
            contactsVisible = !contactsVisible;
            const button = document.getElementById('contacts-toggle');
            
            if (contactsVisible) {{
                showContacts();
                button.textContent = 'Hide Contacts';
                button.classList.add('active');
            }} else {{
                viewer.removeAllShapes();
                if (centroidsVisible) {{
                    toggleCentroids();
                    toggleCentroids();
                }}
                button.textContent = 'Show Contacts';
                button.classList.remove('active');
            }}
            
            viewer.render();
        }}
        
        function showContacts() {{
            const maxFreq = Math.max(...contactsData.map(c => c.frequency));
            const minFreq = Math.min(...contactsData.map(c => c.frequency));
            
            contactsData.forEach(contact => {{
                const normalizedFreq = (contact.frequency - minFreq) / (maxFreq - minFreq);
                const radius = 0.05 + normalizedFreq * 0.20; // Scale thickness
                const intensity = Math.floor(255 * normalizedFreq);
                const color = `rgb(${{255 - intensity}}, ${{intensity}}, 0)`; // Red to Green gradient

                // OTHER CONTACT COLOR SCHEMES OPTIONS
                // OPTION 1: Black to White gradient
                // const color = `rgb(${{intensity}}, ${{intensity}}, ${{intensity}})`;
                // OPTION 2: Blue to Orange gradient (complementary colors)
                // const color = `rgb(${{Math.floor(intensity)}}, ${{Math.floor(intensity * 0.5)}}, ${{Math.floor(255 - intensity)}})`;
                // OPTION 3: Purple to Yellow gradient (complementary colors)
                // const color = `rgb(${{Math.floor(128 + intensity * 0.5)}}, ${{Math.floor(intensity)}}, ${{Math.floor(255 - intensity)}})`;
                // OPTION 4: Red to Cyan gradient (complementary colors)
                // const color = `rgb(${{Math.floor(255 - intensity)}}, ${{Math.floor(intensity)}}, ${{Math.floor(intensity)}})`;
                // OPTION 5: Green to Magenta gradient (complementary colors)
                // const color = `rgb(${{Math.floor(intensity)}}, ${{Math.floor(255 - intensity)}}, ${{Math.floor(intensity)}})`;
                // OPTION 6: Dark Blue to Bright Yellow gradient
                // const color = `rgb(${{Math.floor(intensity)}}, ${{Math.floor(intensity)}}, ${{Math.floor(255 - intensity)}})`;
                // OPTION 7: Dark Purple to Bright Green gradient
                // const color = `rgb(${{Math.floor(255 - intensity)}}, ${{Math.floor(intensity)}}, ${{Math.floor(128 - intensity * 0.5)}})`;
                // OPTION 8: Brown to Cyan gradient
                // const color = `rgb(${{Math.floor(165 - intensity * 0.6)}}, ${{Math.floor(42 + intensity * 0.8)}}, ${{Math.floor(42 + intensity * 0.8)}})`;
                // OPTION 9: Navy to Gold gradient
                // const color = `rgb(${{Math.floor(intensity)}}, ${{Math.floor(intensity * 0.8)}}, ${{Math.floor(128 - intensity * 0.5)}})`;
                // OPTION 10: Teal to Coral gradient
                // const color = `rgb(${{Math.floor(64 + intensity * 0.7)}}, ${{Math.floor(224 - intensity * 0.6)}}, ${{Math.floor(208 - intensity * 0.6)}})`;

                // DEBUGGING: To see what intensity values you're actually getting, add this line:
                console.log('Intensity range:', Math.min(...contactsData.map(c => c.frequency)), 'to', Math.max(...contactsData.map(c => c.frequency)));
                console.log('Current intensity:', intensity);
                
                viewer.addCylinder({{
                    start: {{x: contact.x1, y: contact.y1, z: contact.z1}},
                    end: {{x: contact.x2, y: contact.y2, z: contact.z2}},
                    radius: radius,
                    color: color,
                    alpha: 1
                }});
            }});
        }}
        
        function updateStats() {{
            document.getElementById('total-residues').textContent = centroidsData.length;
            document.getElementById('total-contacts').textContent = contactsData.length;
            
            if (contactsData.length > 0) {{
                const avgFreq = contactsData.reduce((sum, c) => sum + c.frequency, 0) / contactsData.length;
                document.getElementById('avg-frequency').textContent = avgFreq.toFixed(2);
            }} else {{
                document.getElementById('avg-frequency').textContent = '0.00';
            }}
        }}
        
        // Event listeners
        document.getElementById('style-select').addEventListener('change', applyCurrentStyle);
        document.getElementById('color-select').addEventListener('change', applyCurrentStyle);
        document.getElementById('surface-select').addEventListener('change', applyCurrentStyle);
        document.getElementById('centroids-toggle').addEventListener('click', toggleCentroids);
        document.getElementById('contacts-toggle').addEventListener('click', toggleContacts);
        
        // Initialize when page loads
        $(document).ready(function() {{
            initViewer();
        }});
    </script>
</body>
</html>
    """
    
    # Write to file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    logger.info(f"      Contact visualization created: {output_html}")
    return output_html


def create_contact_visualizations_for_clusters(clusters, mm_output, representative_pdbs_dir, logger):
    """
    Create HTML visualizations for all protein pair clusters.
    
    Args:
        clusters: Clusters dictionary
        mm_output: mm_output dictionary
        representative_pdbs_dir: Directory containing the representative PDB files
    """
    
    html_files = []

    # Create folder to store the representative visualizations
    representative_htmls_dir = mm_output['out_path'] + '/contact_clusters/representative_htmls'
    os.makedirs(representative_htmls_dir, exist_ok=True)

    logger.info('INITIALIZING: py3Dmol contacts visualizations for clusters...')

    for pair in clusters:
        logger.info(f'   Creating visualizations for pair: {pair}')
        
        for cluster_n in clusters[pair]:
            representative_model = clusters[pair][cluster_n]["representative"]
            combo = representative_model[0]
            chains = representative_model[1]
            rank_val = representative_model[2]
            
            # Get the contact matrix
            contact_matrix = clusters[pair][cluster_n]['average_matrix']
            
            # Construct PDB file path
            pdb_file = f'{representative_pdbs_dir}/{pair[0]}__vs__{pair[1]}-cluster_{cluster_n}.pdb'
            
            # Create output HTML file path
            output_html = f'{representative_htmls_dir}/{pair[0]}__vs__{pair[1]}-cluster_{cluster_n}.html'
            
            if os.path.exists(pdb_file):
                try:
                    create_contact_visualization(
                        pdb_file=pdb_file,
                        contact_matrix=contact_matrix,
                        chains_in_model=chains,
                        output_html=output_html,
                        protein1_name=pair[0],
                        protein2_name=pair[1],
                        cluster_id=cluster_n,
                        logger = logger
                    )
                    html_files.append(output_html)
                    logger.info(f'      ✓ Created visualization for cluster {cluster_n}')
                except Exception as e:
                    logger.error(f'      ✗ Error creating visualization for cluster {cluster_n}: {e}')
            else:
                logger.error(f'      ✗ PDB file not found: {pdb_file}')
    
    logger.info(f"FINISHED: Created {len(html_files)} py3Dmol HTML visualizations.")
    return html_files


# # Example usage
# if __name__ == "__main__":
#     # This goes after code that creates the PDB files
#     html_files = create_contact_visualizations_for_clusters(
#         clusters=clusters,
#         mm_output=mm_output,
#         representative_pdbs_dir=representative_pdbs_dir
#     )
    
#     print("HTML visualization files created:")
#     for html_file in html_files:
#         print(f"  - {html_file}")


def get_py3dmol_paths_for_pair(pair, contact_clusters_dir):
    """
    For a given pair (e.g. ('SEC13','SEC31')), 
    find all representative_htmls/*.html matching that pair,
    extract their cluster indices, and return a dict mapping
    cluster_index -> absolute file path.
    """
    # Build the directory and search string
    html_dir     = os.path.join(contact_clusters_dir, 'representative_htmls')
    search_token = f"{pair[0]}__vs__{pair[1]}"

    # List all .html files matching the pair
    matching_files = [
        fname for fname in os.listdir(html_dir)
        if fname.endswith(".html") and search_token in fname
    ]

    # Extract cluster numbers and map to full paths
    cluster_paths = {}
    pattern = re.compile(r"-cluster_(\d+)\.html$")

    for fname in matching_files:
        match = pattern.search(fname)
        if not match:
            continue  # skip files that don’t conform
        cluster_id = int(match.group(1))
        full_path  = os.path.join(html_dir, fname)
        cluster_paths[cluster_id] = full_path

    return cluster_paths


def unify_pca_matrixes_and_py3dmol_for_pair(mm_output, pair, logger, left_panel_width=35):
    """
    Create a unified HTML visualization for a pair that combines:
    - Left panel: PCA and matrices visualization
    - Right panel: Selectable py3Dmol cluster visualizations
    
    Args:
        mm_output: Dictionary containing output configuration
        pair: Tuple of protein names (e.g., ('SEC13', 'SEC31'))
        left_panel_width: Percentage of window width for left panel (default: 35)
    """

    # Create a directory for saving plots
    contact_clusters_dir = os.path.join(mm_output['out_path'], 'contact_clusters')
    os.makedirs(contact_clusters_dir, exist_ok=True)

    # Out file
    unified_html_path = os.path.join(contact_clusters_dir, f"{pair[0]}__vs__{pair[1]}-interactive_plot.html")

    # PCA + matrixes file
    pca_and_matrixes_dir = os.path.join(contact_clusters_dir, 'pca_and_matrixes_html')
    pca_and_matrixes_path = os.path.join(pca_and_matrixes_dir, f'{pair[0]}__vs__{pair[1]}-pca_and_matrixes.html')

    # Get py3Dmol HTML visualizations of contacts (existing files)
    py3dmol_htmls_paths_dict = get_py3dmol_paths_for_pair(pair, contact_clusters_dir)

    # Check if required files exist
    if not os.path.exists(pca_and_matrixes_path):
        logger.error(f"   PCA and matrices file not found: {pca_and_matrixes_path}")
        return
    
    if not py3dmol_htmls_paths_dict:
        logger.error(f"   No py3Dmol HTML files found for pair {pair}")
        return
    
    # Sort cluster IDs for consistent ordering
    sorted_cluster_ids = sorted(py3dmol_htmls_paths_dict.keys())
    
    # Create relative paths from the unified HTML location to the existing files
    cluster_relative_paths = {}
    for cluster_id, full_path in py3dmol_htmls_paths_dict.items():
        # Get relative path from unified HTML to the existing cluster file
        cluster_relative_paths[cluster_id] = os.path.relpath(full_path, contact_clusters_dir)
    
    # Get relative path to PCA file
    pca_relative_path = os.path.relpath(pca_and_matrixes_path, contact_clusters_dir)
    
    # Create the unified HTML with lazy loading approach
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Visualization - {pair[0]} vs {pair[1]}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: Arial, sans-serif;
            height: 100vh;
            width: 100vw;
            overflow: hidden;
            background-color: #f0f0f0;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
            width: 100vw;
        }}
        
        .left-panel {{
            width: {left_panel_width}%;
            height: 100%;
            border-right: 2px solid #333;
            overflow: hidden;
        }}
        
        .right-panel {{
            width: {100 - left_panel_width}%;
            height: 100%;
            display: flex;
            flex-direction: column;
            background-color: white;
        }}
        
        .controls {{
            padding: 10px;
            background-color: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            min-height: 50px;
        }}
        
        .controls label {{
            font-weight: bold;
            color: #333;
        }}
        
        .cluster-btn {{
            padding: 8px 16px;
            margin: 2px;
            border: 2px solid #007bff;
            background-color: white;
            color: #007bff;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }}
        
        .cluster-btn:hover {{
            background-color: #e6f3ff;
            transform: translateY(-1px);
        }}
        
        .cluster-btn.active {{
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }}
        
        .py3dmol-container {{
            flex: 1;
            overflow: hidden;
            position: relative;
        }}
        
        .py3dmol-frame {{
            width: 100%;
            height: 100%;
            border: none;
            position: absolute;
            top: 0;
            left: 0;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.3s ease;
        }}
        
        .py3dmol-frame.active {{
            opacity: 1;
            visibility: visible;
        }}
        
        .pca-frame {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        
        .loading {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            color: #666;
            font-size: 18px;
        }}
        
        .error {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            color: #dc3545;
            font-size: 16px;
            text-align: center;
            padding: 20px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                flex-direction: column;
            }}
            
            .left-panel {{
                width: 100%;
                height: 50%;
                border-right: none;
                border-bottom: 2px solid #333;
            }}
            
            .right-panel {{
                width: 100%;
                height: 50%;
            }}
            
            .controls {{
                padding: 5px;
                min-height: 40px;
            }}
            
            .cluster-btn {{
                padding: 6px 12px;
                font-size: 12px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="left-panel">
            <iframe class="pca-frame" src="{pca_relative_path}" title="PCA and Matrices"></iframe>
        </div>
        
        <div class="right-panel">
            <div class="controls">
                <label>Protein Contact Visualization - Cluster:</label>
                {' '.join([f'<button class="cluster-btn{" active" if i == 0 else ""}" onclick="showCluster({cluster_id})" data-cluster="{cluster_id}">{cluster_id}</button>' for i, cluster_id in enumerate(sorted_cluster_ids)])}
            </div>
            
            <div class="py3dmol-container">
                {' '.join([f'<iframe class="py3dmol-frame{" active" if i == 0 else ""}" id="cluster-{cluster_id}" data-src="{cluster_relative_paths[cluster_id]}" title="Cluster {cluster_id}"></iframe>' for i, cluster_id in enumerate(sorted_cluster_ids)])}
            </div>
        </div>
    </div>
    
    <script>
        let loadedClusters = new Set();
        
        function loadClusterFrame(clusterId) {{
            const frame = document.getElementById(`cluster-${{clusterId}}`);
            if (!frame || loadedClusters.has(clusterId)) {{
                return;
            }}
            
            const src = frame.getAttribute('data-src');
            if (src) {{
                frame.src = src;
                loadedClusters.add(clusterId);
            }}
        }}
        
        function showCluster(clusterId) {{
            // Load the cluster frame if not already loaded
            loadClusterFrame(clusterId);
            
            // Hide all frames
            const frames = document.querySelectorAll('.py3dmol-frame');
            frames.forEach(frame => {{
                frame.classList.remove('active');
            }});
            
            // Show selected frame with a slight delay to ensure proper rendering
            setTimeout(() => {{
                const selectedFrame = document.getElementById(`cluster-${{clusterId}}`);
                if (selectedFrame) {{
                    selectedFrame.classList.add('active');
                    
                    // Trigger a resize event to help py3Dmol render properly
                    selectedFrame.onload = function() {{
                        try {{
                            selectedFrame.contentWindow.dispatchEvent(new Event('resize'));
                        }} catch (e) {{
                            // Cross-origin restrictions might prevent this, but that's OK
                        }}
                    }};
                }}
            }}, 100);
            
            // Update button states
            const buttons = document.querySelectorAll('.cluster-btn');
            buttons.forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Activate selected button
            const selectedBtn = document.querySelector(`[data-cluster="${{clusterId}}"]`);
            if (selectedBtn) {{
                selectedBtn.classList.add('active');
            }}
        }}
        
        // Handle window resize
        window.addEventListener('resize', function() {{
            // Trigger resize events on all loaded iframes
            const activeFrame = document.querySelector('.py3dmol-frame.active');
            if (activeFrame && activeFrame.contentWindow) {{
                try {{
                    activeFrame.contentWindow.dispatchEvent(new Event('resize'));
                }} catch (e) {{
                    // Cross-origin restrictions might prevent this, but that's OK
                }}
            }}
        }});
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            // Load and show first cluster by default
            const firstClusterId = {sorted_cluster_ids[0] if sorted_cluster_ids else 0};
            showCluster(firstClusterId);
            
            // Add keyboard navigation
            document.addEventListener('keydown', function(e) {{
                const clusterIds = {sorted_cluster_ids};
                const activeBtn = document.querySelector('.cluster-btn.active');
                if (!activeBtn) return;
                
                const currentCluster = parseInt(activeBtn.getAttribute('data-cluster'));
                const currentIndex = clusterIds.indexOf(currentCluster);
                
                if (e.key === 'ArrowLeft' && currentIndex > 0) {{
                    showCluster(clusterIds[currentIndex - 1]);
                }} else if (e.key === 'ArrowRight' && currentIndex < clusterIds.length - 1) {{
                    showCluster(clusterIds[currentIndex + 1]);
                }}
            }});
        }});
    </script>
</body>
</html>
"""
    
    # Write the unified HTML file
    with open(unified_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"   Unified HTML visualization created: {unified_html_path}")
    logger.info(f"      - Available clusters: {sorted_cluster_ids}")
    
    return unified_html_path




def unify_pca_matrixes_and_py3dmol(mm_output, pairs, logger):

    all_pair_matrices = mm_output['pairwise_contact_matrices']
    
    for pair in pairs:
        if pair not in all_pair_matrices:
            continue

        unify_pca_matrixes_and_py3dmol_for_pair(mm_output, pair, logger)

    



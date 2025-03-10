
import os
import glob
import re
import json
from pathlib import Path

def create_report(directory_path):
    """
    Creates a unified HTML report for the MultimerMapper output.
    
    Args:
        directory_path (str): Path to the directory containing the MultimerMapper output.
    """
    directory_path = os.path.abspath(directory_path)
    output_path = os.path.join(directory_path, "report.html")
    
    # Collect protein names
    protein_names = get_protein_names(directory_path)
    
    # Collect available contact clusters
    contact_clusters = get_contact_clusters(directory_path)
    
    # Collect available pLDDT clusters
    plddt_clusters = get_plddt_clusters(directory_path)
    
    # Collect available monomer trajectories
    monomer_trajectories = get_monomer_trajectories(directory_path)
    
    # Collect available fallback analysis images
    fallback_images = get_fallback_images(directory_path)
    
    # Collect available combinations suggestion files
    combinations_files = get_combinations_files(directory_path)
    
    # Generate the main HTML content
    html_content = generate_html(
        directory_path, 
        protein_names, 
        contact_clusters, 
        plddt_clusters, 
        monomer_trajectories,
        fallback_images,
        combinations_files
    )
    
    # Write the HTML content to the output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Report created at: {output_path}")

def get_protein_names(directory_path):
    """Extract unique protein names from the directory structure."""
    protein_names = set()
    
    # Extract from domains directory
    domain_files = glob.glob(os.path.join(directory_path, "domains", "*-domains_plot.html"))
    for file in domain_files:
        protein_name = os.path.basename(file).split("-domains_plot.html")[0]
        protein_names.add(protein_name)
    
    # Extract from plddt_clusters directory
    plddt_files = glob.glob(os.path.join(directory_path, "plddt_clusters", "*", "*-interactive_plddt_clusters.html"))
    for file in plddt_files:
        protein_name = os.path.basename(file).split("-interactive_plddt_clusters.html")[0]
        protein_names.add(protein_name)
    
    # Extract from monomer_trajectories directory
    monomer_dirs = glob.glob(os.path.join(directory_path, "monomer_trajectories", "*"))
    for dir_path in monomer_dirs:
        protein_name = os.path.basename(dir_path)
        protein_names.add(protein_name)
    
    return sorted(list(protein_names))

def get_contact_clusters(directory_path):
    """Extract contact cluster information."""
    contact_cluster_files = glob.glob(os.path.join(directory_path, "contact_clusters", "*-interactive_plot.html"))
    contact_clusters = []
    
    for file in contact_cluster_files:
        basename = os.path.basename(file)
        match = re.match(r"(.+)__vs__(.+)-interactive_plot\.html", basename)
        if match:
            protein1, protein2 = match.groups()
            protein2 = protein2.split('-interactive_plot')[0]
            contact_clusters.append({
                "protein1": protein1,
                "protein2": protein2,
                "path": os.path.relpath(file, directory_path)
            })
    
    return contact_clusters

def get_plddt_clusters(directory_path):
    """Extract pLDDT cluster information."""
    plddt_clusters = {}
    
    plddt_dirs = glob.glob(os.path.join(directory_path, "plddt_clusters", "*"))
    for dir_path in plddt_dirs:
        protein_name = os.path.basename(dir_path)
        plddt_file = os.path.join(dir_path, f"{protein_name}-interactive_plddt_clusters.html")
        
        if os.path.exists(plddt_file):
            plddt_clusters[protein_name] = os.path.relpath(plddt_file, directory_path)
    
    return plddt_clusters

def get_monomer_trajectories(directory_path):
    """Extract monomer trajectories information."""
    trajectories = {}
    
    protein_dirs = glob.glob(os.path.join(directory_path, "monomer_trajectories", "*"))
    for protein_dir in protein_dirs:
        protein_name = os.path.basename(protein_dir)
        trajectories[protein_name] = {"monomer": None, "domains": []}
        
        # Check for monomer data
        monomer_dir = os.path.join(protein_dir, f"{protein_name}_monomer")
        if os.path.exists(monomer_dir):
            interactive_file = os.path.join(monomer_dir, f"{protein_name}_interactive_trajectory.html")
            rmsd_file = os.path.join(monomer_dir, f"{protein_name}_monomer_RMSD_traj.html")
            
            if os.path.exists(interactive_file) and os.path.exists(rmsd_file):
                trajectories[protein_name]["monomer"] = {
                    "interactive": os.path.relpath(interactive_file, directory_path),
                    "rmsd": os.path.relpath(rmsd_file, directory_path)
                }
        
        # Check for domain data
        domain_dirs = glob.glob(os.path.join(protein_dir, f"{protein_name}_domain_*"))
        for domain_dir in domain_dirs:
            domain_name = os.path.basename(domain_dir)
            domain_num = domain_name.split("_domain_")[1]
            
            interactive_file = os.path.join(domain_dir, f"{protein_name}_interactive_trajectory.html")
            rmsd_file = os.path.join(domain_dir, f"{domain_name}_RMSD_traj.html")
            
            if os.path.exists(interactive_file) and os.path.exists(rmsd_file):
                trajectories[protein_name]["domains"].append({
                    "domain_num": domain_num,
                    "interactive": os.path.relpath(interactive_file, directory_path),
                    "rmsd": os.path.relpath(rmsd_file, directory_path)
                })
    
    # Remove proteins with no data
    trajectories = {k: v for k, v in trajectories.items() if v["monomer"] is not None or v["domains"]}
    
    return trajectories

def get_fallback_images(directory_path):
    """Get all PNG files in the fallback_analysis directory."""
    fallback_images = glob.glob(os.path.join(directory_path, "fallback_analysis", "*.png"))
    return [os.path.relpath(img, directory_path) for img in fallback_images]

def get_combinations_files(directory_path):
    """Get all text files in the combinations_suggestions directory."""
    files = glob.glob(os.path.join(directory_path, "combinations_suggestions", "*.txt"))
    files.extend(glob.glob(os.path.join(directory_path, "combinations_suggestions", "*.csv")))
    files.extend(glob.glob(os.path.join(directory_path, "combinations_suggestions", "*.fasta")))
    return [os.path.relpath(f, directory_path) for f in files]

def generate_html(directory_path, protein_names, contact_clusters, plddt_clusters, 
                  monomer_trajectories, fallback_images, combinations_files):
    """Generate the main HTML content."""
    
    # Convert data to JSON for JavaScript use
    contact_clusters_json = json.dumps(contact_clusters)
    plddt_clusters_json = json.dumps(plddt_clusters)
    monomer_trajectories_json = json.dumps(monomer_trajectories)
    protein_names_json = json.dumps(protein_names)
    fallback_images_json = json.dumps(fallback_images)
    combinations_files_json = json.dumps(combinations_files)
    
    # Main HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MultimerMapper Report</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {{
            --primary-color: #2a6496;
            --secondary-color: #4c88c2;
            --accent-color: #6ca2da;
            --light-color: #e8f0f9;
            --dark-color: #1a3c5a;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
            --gray-color: #6c757d;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Arial', sans-serif;
        }}
        
        body {{
            background-color: #f4f7fa;
            overflow-x: hidden;
        }}
        
        .container {{
            display: flex;
            min-height: 100vh;
        }}
        
        /* Sidebar/Menu Styles */
        .sidebar {{
            width: 280px;
            background-color: var(--dark-color);
            color: white;
            height: 100vh;
            position: fixed;
            overflow-y: auto;
            transition: all 0.3s;
            z-index: 100;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
        }}
        
        .sidebar-header {{
            padding: 20px;
            background-color: rgba(0, 0, 0, 0.2);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        
        .sidebar-header h3 {{
            font-size: 1.5rem;
            margin: 0;
            font-weight: 700;
        }}
        
        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .sidebar-logo i {{
            font-size: 1.8rem;
            color: var(--accent-color);
        }}
        
        .menu-item {{
            padding: 15px 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            cursor: pointer;
            display: flex;
            align-items: center;
            transition: background-color 0.3s;
        }}
        
        .menu-item:hover {{
            background-color: rgba(255, 255, 255, 0.05);
        }}
        
        .menu-item i {{
            margin-right: 10px;
            width: 20px;
            text-align: center;
            color: var(--accent-color);
        }}
        
        .menu-item.active {{
            background-color: var(--primary-color);
        }}
        
        .submenu {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.5s ease;
            background-color: rgba(0, 0, 0, 0.2);
        }}
        
        .submenu-item {{
            padding: 12px 20px 12px 50px;
            cursor: pointer;
            transition: background-color 0.3s;
            font-size: 0.9rem;
        }}
        
        .submenu-item:hover {{
            background-color: rgba(255, 255, 255, 0.05);
        }}
        
        /* Main Content Area */
        .main-content {{
            flex: 1;
            padding: 20px;
            margin-left: 280px;
            transition: all 0.3s;
        }}
        
        iframe {{
            width: 100%;
            height: calc(100vh - 40px);
            border: none;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            background-color: white;
        }}
        
        /* Contact Clusters Slide Panel */
        .slide-panel {{
            position: fixed;
            top: 0;
            right: -600px;
            width: 600px;
            height: 100vh;
            background-color: white;
            box-shadow: -2px 0 10px rgba(0, 0, 0, 0.1);
            transition: right 0.3s;
            z-index: 1000;
            padding: 20px;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
        }}
        
        .slide-panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 15px;
            border-bottom: 1px solid #e3e3e3;
            margin-bottom: 20px;
        }}
        
        .slide-panel-header h3 {{
            margin: 0;
            color: var(--primary-color);
        }}
        
        .close-panel {{
            background: none;
            border: none;
            font-size: 1.5rem;
            color: var(--gray-color);
            cursor: pointer;
        }}
        
        .close-panel:hover {{
            color: var(--danger-color);
        }}
        
        .protein-selector {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .protein-column {{
            flex: 1;
        }}
        
        .protein-column h4 {{
            margin-bottom: 10px;
            color: var(--dark-color);
        }}
        
        .protein-button {{
            display: block;
            width: 100%;
            padding: 10px;
            margin-bottom: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #f8f9fa;
            text-align: left;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .protein-button:hover {{
            background-color: var(--light-color);
        }}
        
        .protein-button.selected {{
            background-color: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }}
        
        .protein-button.available {{
            background-color: rgba(40, 167, 69, 0.1);
            border-color: var(--success-color);
        }}
        
        .protein-button.unavailable {{
            background-color: #f8f9fa;
            color: var(--gray-color);
            cursor: not-allowed;
        }}
        
        .view-button {{
            padding: 12px 20px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}
        
        .view-button:hover {{
            background-color: var(--secondary-color);
        }}
        
        .view-button:disabled {{
            background-color: var(--gray-color);
            cursor: not-allowed;
        }}
        
        /* Split View for RMSD Trajectory */
        .split-view {{
            display: flex;
            height: calc(100vh - 40px);
            gap: 20px;
        }}
        
        .split-panel {{
            flex: 1;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            background-color: white;
            overflow: hidden;
        }}
        
        .split-panel iframe {{
            height: 100%;
            box-shadow: none;
        }}
        
        /* Fallback Analysis Gallery */
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        
        .gallery-item {{
            overflow: hidden;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s;
        }}
        
        .gallery-item:hover {{
            transform: scale(1.02);
        }}
        
        .gallery-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        /* Combinations Suggestions */
        .file-list {{
            list-style: none;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        
        .file-item {{
            padding: 15px;
            margin-bottom: 10px;
            border: 1px solid #e3e3e3;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
            display: flex;
            align-items: center;
        }}
        
        .file-item:hover {{
            background-color: var(--light-color);
        }}
        
        .file-item i {{
            margin-right: 10px;
            color: var(--primary-color);
        }}
        
        .file-content {{
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            white-space: pre-wrap;
            font-family: monospace;
            overflow-x: auto;
        }}
        
        /* Log File View */
        .log-content {{
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            height: calc(100vh - 80px);
            overflow-y: auto;
            font-family: monospace;
            white-space: pre-wrap;
            color: #333;
            line-height: 1.5;
        }}
        
        /* About Page */
        .about-content {{
            padding: 30px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        
        .about-content h1 {{
            color: var(--primary-color);
            margin-bottom: 20px;
        }}
        
        /* Data Table for RMSD Trajectories */
        .data-table {{
            width: 100%;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        
        .protein-row {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #e3e3e3;
        }}
        
        .protein-row:last-child {{
            border-bottom: none;
        }}
        
        .protein-name {{
            font-weight: bold;
            color: var(--dark-color);
            margin-bottom: 10px;
            font-size: 1.1rem;
        }}
        
        .trajectory-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .trajectory-button {{
            padding: 8px 15px;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }}
        
        .trajectory-button:hover {{
            background-color: var(--secondary-color);
        }}
        
        .trajectory-button i {{
            margin-right: 5px;
        }}
        
        /* Utility Classes */
        .mt-20 {{
            margin-top: 20px;
        }}
        
        .mb-20 {{
            margin-bottom: 20px;
        }}
        
        /* Mobile Responsiveness */
        @media (max-width: 768px) {{
            .sidebar {{
                width: 0;
                position: fixed;
            }}
            
            .sidebar.active {{
                width: 280px;
            }}
            
            .main-content {{
                margin-left: 0;
            }}
            
            .slide-panel {{
                width: 100%;
                right: -100%;
            }}
            
            .split-view {{
                flex-direction: column;
            }}
        }}
        /* Add this to the existing slide-panel CSS */
        .slide-panel {{
            /* existing styles */
            right: -600px; /* Ensure this matches the initial position */
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Sidebar/Menu -->
        <div class="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-logo">
                    <i class="fas fa-dna"></i>
                    <h3>MultimerMapper</h3>
                </div>
            </div>
            
            <div class="menu-item" id="home-button">
                <i class="fas fa-home"></i>
                <span>Home</span>
            </div>
            
            <div class="menu-item" id="graph-2d-button">
                <i class="fas fa-project-diagram"></i>
                <span>2D Graph</span>
            </div>
            
            <div class="menu-item" id="graph-py3dmol-button">
                <i class="fas fa-cube"></i>
                <span>py3Dmol Graph</span>
            </div>
            
            <div class="menu-item" id="graph-plotly3d-button">
                <i class="fas fa-cubes"></i>
                <span>plotly3D Graph</span>
            </div>
            
            <div class="menu-item menu-dropdown" id="domains-dropdown">
                <i class="fas fa-puzzle-piece"></i>
                <span>Domains</span>
            </div>
            <div class="submenu" id="domains-submenu">
                <!-- Will be populated dynamically -->
            </div>
            
            <div class="menu-item" id="contact-clusters-button">
                <i class="fas fa-network-wired"></i>
                <span>Contact Clusters</span>
            </div>
            
            <div class="menu-item menu-dropdown" id="plddt-clusters-dropdown">
                <i class="fas fa-chart-bar"></i>
                <span>pLDDT Clusters</span>
            </div>
            <div class="submenu" id="plddt-clusters-submenu">
                <!-- Will be populated dynamically -->
            </div>
            
            <div class="menu-item" id="rmsd-trajectories-button">
                <i class="fas fa-chart-line"></i>
                <span>RMSD Trajectories</span>
            </div>
            
            <div class="menu-item" id="fallback-analysis-button">
                <i class="fas fa-images"></i>
                <span>Fallback Analysis</span>
            </div>
            
            <div class="menu-item" id="combinations-suggestions-button">
                <i class="fas fa-lightbulb"></i>
                <span>Combinations Suggestions</span>
            </div>
            
            <div class="menu-item" id="run-loggings-button">
                <i class="fas fa-file-alt"></i>
                <span>Run Loggings</span>
            </div>
            
            <div class="menu-item" id="about-button">
                <i class="fas fa-info-circle"></i>
                <span>About MultimerMapper</span>
            </div>
        </div>
        
        <!-- Main Content Area -->
        <div class="main-content" id="main-content">
            <!-- Default content (will be replaced dynamically) -->
            <iframe id="main-frame" src="graphs/2D_graph.html"></iframe>
        </div>
        
        <!-- Contact Clusters Slide Panel -->
        <div class="slide-panel" id="contact-clusters-panel">
            <div class="slide-panel-header">
                <h3>Contact Clusters</h3>
                <button class="close-panel" id="close-contact-panel">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="protein-selector">
                <div class="protein-column">
                    <h4>Select First Protein</h4>
                    <div id="protein1-buttons">
                        <!-- Will be populated dynamically -->
                    </div>
                </div>
                <div class="protein-column">
                    <h4>Select Second Protein</h4>
                    <div id="protein2-buttons">
                        <!-- Will be populated dynamically -->
                    </div>
                </div>
            </div>
            <button class="view-button" id="view-contact-cluster" disabled>
                <i class="fas fa-eye"></i> View Contact Cluster
            </button>
        </div>

        <!-- Slide panel for domains -->
        <div class="slide-panel" id="domains-panel">
            <div class="slide-panel-header">
                <h3>Domains Selection</h3>
                <button class="close-panel" id="close-domains-panel">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="protein-selector">
                <div class="protein-column">
                    <h4>Select Protein</h4>
                    <div id="domains-protein-buttons"></div>
                </div>
            </div>
        </div>

        <!-- Slide panel for pLDDT Clusters -->
        <div class="slide-panel" id="plddt-clusters-panel">
            <div class="slide-panel-header">
                <h3>pLDDT Clusters Selection</h3>
                <button class="close-panel" id="close-plddt-panel">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="protein-selector">
                <div class="protein-column">
                    <h4>Select Protein</h4>
                    <div id="plddt-protein-buttons"></div>
                </div>
            </div>
        </div>

    </div>
    
    <script>
        // Data for dynamic content
        const proteinNames = {protein_names_json};
        const contactClusters = {contact_clusters_json};
        const plddtClusters = {plddt_clusters_json};
        const monomerTrajectories = {monomer_trajectories_json};
        const fallbackImages = {fallback_images_json};
        const combinationsFiles = {combinations_files_json};
        
        // DOM Elements
        const mainContent = document.getElementById('main-content');
        const mainFrame = document.getElementById('main-frame');
        const sidebar = document.querySelector('.sidebar');
        const menuItems = document.querySelectorAll('.menu-item');
        const contactClustersPanel = document.getElementById('contact-clusters-panel');
        
        // Function to load content into the main frame
        function loadInFrame(path) {{
            mainFrame.src = path;
        }}
        
        // Function to replace main content with custom HTML
        function setMainContent(html) {{
            mainContent.innerHTML = html;
        }}
        
        // Initialize menu
        function initMenu() {{
            // Home button
            document.getElementById('home-button').addEventListener('click', () => {{
                setMainContent('<iframe id="main-frame" src="graphs/2D_graph.html"></iframe>');
            }});
            
            // 2D Graph button
            document.getElementById('graph-2d-button').addEventListener('click', () => {{
                setMainContent('<iframe id="main-frame" src="graphs/2D_graph.html"></iframe>');
            }});
            
            // py3Dmol Graph button
            document.getElementById('graph-py3dmol-button').addEventListener('click', () => {{
                setMainContent('<iframe id="main-frame" src="graphs/3D_graph_py3Dmol.html"></iframe>');
            }});
            
            // plotly3D Graph button
            document.getElementById('graph-plotly3d-button').addEventListener('click', () => {{
                setMainContent('<iframe id="main-frame" src="graphs/3D_graph_plotly.html"></iframe>');
            }});
            
            // Initialize domains dropdown
            initDomainsDropdown();
            
            // Contact Clusters button
            document.getElementById('contact-clusters-button').addEventListener('click', openContactClustersPanel);
            
            // Initialize pLDDT Clusters dropdown
            initPLDDTClustersDropdown();
            
            // RMSD Trajectories button
            document.getElementById('rmsd-trajectories-button').addEventListener('click', showRMSDTrajectories);
            
            // Fallback Analysis button
            document.getElementById('fallback-analysis-button').addEventListener('click', showFallbackAnalysis);
            
            // Combinations Suggestions button
            document.getElementById('combinations-suggestions-button').addEventListener('click', showCombinationsSuggestions);
            
            // Run Loggings button
            document.getElementById('run-loggings-button').addEventListener('click', showRunLoggings);
            
            // About button
            document.getElementById('about-button').addEventListener('click', showAboutPage);
            
            // Menu dropdown functionality
            const menuDropdowns = document.querySelectorAll('.menu-dropdown');
            menuDropdowns.forEach(dropdown => {{
                dropdown.addEventListener('click', function() {{
                    const submenuId = this.id.replace('dropdown', 'submenu');
                    const submenu = document.getElementById(submenuId);
                    
                    // Toggle submenu visibility
                    if (submenu.style.maxHeight) {{
                        submenu.style.maxHeight = null;
                    }} else {{
                        submenu.style.maxHeight = submenu.scrollHeight + 'px';
                    }}
                }});
            }});
        }}
        
        // With this corrected code:
        // Keep the original menu initialization and add panel handlers
        function initDomainsDropdown() {{
            // Original dropdown functionality removed, just keep empty
        }}

        function initPLDDTClustersDropdown() {{
            // Original dropdown functionality removed, just keep empty
        }}

        // Add panel handlers for domains and pLDDT clusters
        document.getElementById('domains-dropdown').addEventListener('click', openDomainsPanel);
        document.getElementById('plddt-clusters-dropdown').addEventListener('click', openPLDDTPanel);

        // Domains button click handler
        document.getElementById('domains-dropdown').addEventListener('click', openDomainsPanel);
        document.getElementById('plddt-clusters-dropdown').addEventListener('click', openPLDDTPanel);

        function openDomainsPanel() {{
            const panel = document.getElementById('domains-panel');
            panel.style.right = '0';
            populateDomainButtons();
            
            document.getElementById('close-domains-panel').addEventListener('click', () => {{
                panel.style.right = '-600px';
            }});
        }}

        function openPLDDTPanel() {{
            const panel = document.getElementById('plddt-clusters-panel');
            panel.style.right = '0';
            populatePLDDTButtons();
            
            document.getElementById('close-plddt-panel').addEventListener('click', () => {{
                panel.style.right = '-600px';
            }});
        }}

        function populateDomainButtons() {{
            const container = document.getElementById('domains-protein-buttons');
            container.innerHTML = '';
            
            proteinNames.forEach(protein => {{
                const button = document.createElement('button');
                button.className = 'protein-button';
                button.textContent = protein;
                button.addEventListener('click', () => {{
                    loadInFrame(`domains/${{protein}}-domains_plot.html`);
                    document.getElementById('domains-panel').style.right = '-600px';
                }});
                container.appendChild(button);
            }});
        }}

        function populatePLDDTButtons() {{
            const container = document.getElementById('plddt-protein-buttons');
            container.innerHTML = '';
            
            Object.keys(plddtClusters).forEach(protein => {{
                const button = document.createElement('button');
                button.className = 'protein-button';
                button.textContent = protein;
                button.addEventListener('click', () => {{
                    loadInFrame(plddtClusters[protein]);
                    document.getElementById('plddt-clusters-panel').style.right = '-600px';
                }});
                container.appendChild(button);
            }});
        }}
        
        // Open Contact Clusters Panel
        function openContactClustersPanel() {{
            // Show the panel
            contactClustersPanel.style.right = '0';
            
            // Populate protein buttons
            populateContactProteinButtons();
            
            // Close button functionality
            document.getElementById('close-contact-panel').addEventListener('click', () => {{
                contactClustersPanel.style.right = '-600px';
            }});
        }}
        
        // Populate Contact Protein Buttons
        function populateContactProteinButtons() {{
            const protein1Buttons = document.getElementById('protein1-buttons');
            const protein2Buttons = document.getElementById('protein2-buttons');
            
            // Clear existing buttons
            protein1Buttons.innerHTML = '';
            protein2Buttons.innerHTML = '';
            
            // Get unique proteins from contact clusters
            const uniqueProteins = new Set();
            contactClusters.forEach(cluster => {{
                uniqueProteins.add(cluster.protein1);
                uniqueProteins.add(cluster.protein2);
            }});
            
            // Create buttons for protein1
            proteinNames.forEach(protein => {{
                const button = document.createElement('button');
                button.className = 'protein-button';
                button.textContent = protein;
                button.setAttribute('data-protein', protein);
                
                if (uniqueProteins.has(protein)) {{
                    button.classList.add('available');
                }} else {{
                    button.classList.add('unavailable');
                }}
                
                button.addEventListener('click', selectProtein1);
                protein1Buttons.appendChild(button);
            }});
            
            // View button functionality
            const viewButton = document.getElementById('view-contact-cluster');
            viewButton.addEventListener('click', viewContactCluster);
        }}
        
        // Select first protein
        function selectProtein1() {{
            // Remove selected class from all buttons
            document.querySelectorAll('#protein1-buttons .protein-button').forEach(btn => {{
                btn.classList.remove('selected');
            }});
            
            // Add selected class to this button
            this.classList.add('selected');
            
            // Get the selected protein
            const selectedProtein = this.getAttribute('data-protein');
            
            // Update protein2 buttons
            updateProtein2Buttons(selectedProtein);
        }}
        
        // Update protein2 buttons based on selected protein1
        function updateProtein2Buttons(protein1) {{
            const protein2Buttons = document.getElementById('protein2-buttons');
            protein2Buttons.innerHTML = '';
            
            // Get available protein2 options for selected protein1
            const availableProtein2 = new Set();
            contactClusters.forEach(cluster => {{
                if (cluster.protein1 === protein1) {{
                    availableProtein2.add(cluster.protein2);
                }} else if (cluster.protein2 === protein1) {{
                    availableProtein2.add(cluster.protein1);
                }}
            }});
            
            // Create buttons for protein2
            proteinNames.forEach(protein => {{
                const button = document.createElement('button');
                button.className = 'protein-button';
                button.textContent = protein;
                button.setAttribute('data-protein', protein);
                
                if (availableProtein2.has(protein)) {{
                    button.classList.add('available');
                }} else {{
                    button.classList.add('unavailable');
                    button.disabled = true;
                }}
                
                button.addEventListener('click', selectProtein2);
                protein2Buttons.appendChild(button);
            }});
        }}
        
        // Select second protein
        function selectProtein2() {{
            // Remove selected class from all buttons
            document.querySelectorAll('#protein2-buttons .protein-button').forEach(btn => {{
                btn.classList.remove('selected');
            }});
            
            // Add selected class to this button
            this.classList.add('selected');
            
            // Enable the view button
            document.getElementById('view-contact-cluster').disabled = false;
        }}
        
        // View contact cluster
        function viewContactCluster() {{
            const protein1 = document.querySelector('#protein1-buttons .selected').getAttribute('data-protein');
            const protein2 = document.querySelector('#protein2-buttons .selected').getAttribute('data-protein');
            
            // Find the corresponding contact cluster
            let clusterPath = '';
            contactClusters.forEach(cluster => {{
                if ((cluster.protein1 === protein1 && cluster.protein2 === protein2) || 
                    (cluster.protein1 === protein2 && cluster.protein2 === protein1)) {{
                    clusterPath = cluster.path;
                }}
            }});
            
            // Close the panel
            contactClustersPanel.style.right = '-600px';
            
            // Load the contact cluster
            if (clusterPath) {{
                setMainContent(`<iframe id="main-frame" src="${{clusterPath}}"></iframe>`);
            }}
        }}
        
        // Show RMSD Trajectories
        function showRMSDTrajectories() {{
            let html = `
                <div class="data-table">
                    <h2 class="mb-20">RMSD Trajectories</h2>
            `;
            
            Object.keys(monomerTrajectories).forEach(protein => {{
                const data = monomerTrajectories[protein];
                // Sort domains numerically
                const sortedDomains = data.domains.sort((a, b) => 
                    parseInt(a.domain_num) - parseInt(b.domain_num)
                );
                
                html += `<div class="protein-row">
                    <div class="protein-name">${{protein}}</div>
                    <div class="trajectory-buttons">`;
                
                if (data.monomer) {{
                    html += `
                        <button class="trajectory-button" onclick="showSplitView('${{data.monomer.interactive}}', '${{data.monomer.rmsd}}')">
                            <i class="fas fa-chart-line"></i> Monomer Trajectory
                        </button>
                    `;
                }}
                
                sortedDomains.forEach(domain => {{
                    html += `
                        <button class="trajectory-button" onclick="showSplitView('${{domain.interactive}}', '${{domain.rmsd}}')">
                            <i class="fas fa-puzzle-piece"></i> Domain ${{domain.domain_num}}
                        </button>
                    `;
                }});
                
                html += `</div></div>`;
            }});
            
            html += `</div>`;
            setMainContent(html);
        }}
        
        // Show Split View for RMSD Trajectory
        function showSplitView(interactivePath, rmsdPath) {{
            const html = `
                <div class="split-view">
                    <div class="split-panel">
                        <iframe src="${{interactivePath}}"></iframe>
                    </div>
                    <div class="split-panel">
                        <iframe src="${{rmsdPath}}"></iframe>
                    </div>
                </div>
            `;
            setMainContent(html);
        }}
        
        // Show Fallback Analysis Gallery
        function showFallbackAnalysis() {{
            if (fallbackImages.length === 0) {{
                setMainContent(`
                    <div class="about-content">
                        <h2>No Fallback Analysis Images Available</h2>
                        <p>There are no fallback analysis images in the output directory.</p>
                    </div>
                `);
                return;
            }}
            
            let html = `<div class="gallery">`;
            
            fallbackImages.forEach(image => {{
                html += `
                    <div class="gallery-item">
                        <img src="${{image}}" alt="Fallback Analysis" onclick="showFullImage('${{image}}')">
                    </div>
                `;
            }});
            
            html += `</div>`;
            setMainContent(html);
        }}
        
        // Show Full Image
        function showFullImage(path) {{
            setMainContent(`
                <div style="padding: 20px; text-align: center;">
                    <button class="view-button mb-20" onclick="showFallbackAnalysis()">
                        <i class="fas fa-arrow-left"></i> Back to Gallery
                    </button>
                    <img src="${{path}}" alt="Fallback Analysis" style="max-width: 100%; height: auto;">
                </div>
            `);
        }}
        
        // Show Combinations Suggestions
        function showCombinationsSuggestions() {{
            if (combinationsFiles.length === 0) {{
                setMainContent(`
                    <div class="about-content">
                        <h2>No Combinations Suggestions Available</h2>
                        <p>There are no combinations suggestion files in the output directory.</p>
                    </div>
                `);
                return;
            }}
            
            let html = `
                <div class="about-content">
                    <h2 class="mb-20">Combinations Suggestions</h2>
                    <ul class="file-list">
            `;
            
            combinationsFiles.forEach(file => {{
                const fileName = file.split('/').pop();
                const fileExt = fileName.split('.').pop().toLowerCase();
                
                let icon = 'fa-file-alt';
                if (fileExt === 'csv') icon = 'fa-file-csv';
                if (fileExt === 'fasta') icon = 'fa-dna';
                
                html += `
                    <li class="file-item" onclick="showFileContent('${{file}}')">
                        <i class="fas ${{icon}}"></i> ${{fileName}}
                    </li>
                `;
            }});
            
            html += `
                    </ul>
                    <div id="file-content-container"></div>
                </div>
            `;
            
            setMainContent(html);
        }}
        
        // Show File Content
        function showFileContent(path) {{
            fetch(path)
                .then(response => response.text())
                .then(content => {{
                    const container = document.getElementById('file-content-container');
                    container.innerHTML = `
                        <div class="file-content">${{content}}</div>
                    `;
                }})
                .catch(error => {{
                    console.error('Error fetching file:', error);
                }});
        }}
        
        // Show Run Loggings
        function showRunLoggings() {{
            fetch('loggings.txt')
                .then(response => response.text())
                .then(content => {{
                    setMainContent(`
                        <div class="log-content">${{content}}</div>
                    `);
                }})
                .catch(error => {{
                    setMainContent(`
                        <div class="about-content">
                            <h2>Run Loggings Not Available</h2>
                            <p>The loggings.txt file was not found in the output directory.</p>
                        </div>
                    `);
                }});
        }}
        
        // Show About Page
        function showAboutPage() {{
            setMainContent(`
                <div class="about-content">
                    <h1>About MultimerMapper</h1>
                    <p>MultimerMapper is a comprehensive tool for analyzing protein structures and interactions. It provides various visualizations and analyses to help understand protein complexes.</p>
                    
                    <h2 class="mt-20">Key Features:</h2>
                    <ul>
                        <li>Interactive 2D and 3D graph visualizations of protein complexes</li>
                        <li>Domain analysis and visualization</li>
                        <li>Contact cluster analysis between protein pairs</li>
                        <li>pLDDT cluster analysis for model quality assessment</li>
                        <li>RMSD trajectory analysis for conformational changes</li>
                        <li>Fallback analysis for problematic structures</li>
                        <li>Suggestions for optimal protein combinations</li>
                    </ul>
                    
                    <h2 class="mt-20">How to Use This Report:</h2>
                    <p>Navigate using the sidebar menu to explore different aspects of your protein complex analysis. Each section provides specific insights:</p>
                    <ul>
                        <li><strong>2D/3D Graphs:</strong> Overall structure of the protein complex</li>
                        <li><strong>Domains:</strong> Structural domains within each protein</li>
                        <li><strong>Contact Clusters:</strong> Interaction interfaces between protein pairs</li>
                        <li><strong>pLDDT Clusters:</strong> Model quality assessment</li>
                        <li><strong>RMSD Trajectories:</strong> Conformational dynamics</li>
                        <li><strong>Fallback Analysis:</strong> Alternative analyses for problematic structures</li>
                        <li><strong>Combinations Suggestions:</strong> Recommended protein combinations</li>
                    </ul>
                </div>
            `);
        }}
        
        // Initialize the UI
        document.addEventListener('DOMContentLoaded', function() {{
            initMenu();
            
            // Add window functions to global scope
            window.showSplitView = showSplitView;
            window.showFullImage = showFullImage;
            window.showFileContent = showFileContent;
        }});
    </script>
</body>
</html>
    """
    
    return html

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python create_report.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    create_report(directory_path)
#!/usr/bin/env python3
"""
Protein Domain Visualization Generator

This script creates an interactive HTML visualization showing:
- Protein sequence as horizontal rectangles
- Per-residue pLDDT heatmap (AlphaFold colors)
- Coefficient of variation heatmap
- InterPro domain annotations
- Domain segments from TSV file

Usage:
python protein_viz.py <interpro_json> <protein_id> <plddts_json> <domains_tsv> [output_html]
"""

import json
import pandas as pd
import argparse
import sys
from pathlib import Path

def alphafold_color(plddt):
    """Convert pLDDT score to AlphaFold color scheme"""
    if plddt >= 90:
        return "#0053D6"  # Very high confidence (dark blue)
    elif plddt >= 70:
        return "#65CBF3"  # Confident (light blue)
    elif plddt >= 50:
        return "#FFDB13"  # Low confidence (yellow)
    else:
        return "#FF7D45"  # Very low confidence (orange)

def cv_color(cv, min_cv, max_cv):
    """Convert CV to color scale (red=low CV, green=high CV)"""
    if max_cv == min_cv:
        return "#FFFF00"  # Yellow for uniform CV
    
    # Normalize CV to 0-1 range
    normalized = (cv - min_cv) / (max_cv - min_cv)
    
    # Interpolate between red (0) and green (1)
    red = int(255 * (1 - normalized))
    green = int(255 * normalized)
    blue = 0
    
    return f"#{red:02x}{green:02x}{blue:02x}"

def generate_domain_colors():
    """Generate distinct colors for domains"""
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8C471", "#82E0AA", "#F1948A", "#85C1E9", "#F4D03F",
        "#D7DBDD", "#AED6F1", "#A9DFBF", "#F9E79F", "#D2B4DE"
    ]
    return colors

def create_interpro_html_visualization(interpro_data, protein_id, plddts_data, domains_df, output_file, logger = None):
    """Create the HTML visualization"""
    
    # # Debug: Print data structure info
    # print(f"Debug: interpro_data keys: {interpro_data.keys() if interpro_data else 'None'}")
    # if interpro_data and 'results' in interpro_data:
    #     print(f"Debug: Number of results: {len(interpro_data['results'])}")
    #     if interpro_data['results']:
    #         print(f"Debug: First result keys: {interpro_data['results'][0].keys()}")
    
    # Check if interpro_data is properly structured
    if not interpro_data and logger is not None:
        logger.error("InterPro data is None or empty")
    
    if 'results' not in interpro_data and logger is not None:
        logger.error("InterPro data missing 'results' key")
    
    if not interpro_data['results'] and logger is not None:
        logger.error("InterPro results list is empty")
    
    if 'sequence' not in interpro_data['results'][0] and logger is not None:
        logger.error("InterPro results missing 'sequence' key")
    
    # Get protein sequence length
    sequence = interpro_data['results'][0]['sequence']
    if not sequence and logger is not None:
        logger.error("Protein sequence is empty")
    
    # seq_length = len(sequence)
    # print(f"Debug: Sequence length: {seq_length}")
    
    # # Get pLDDT data for the protein
    # print(f"Debug: Available protein IDs in pLDDT data: {list(plddts_data.keys()) if plddts_data else 'None'}")
    # print(f"Debug: Looking for protein ID: '{protein_id}'")
    
    if not plddts_data and logger is not None:
        logger.error("pLDDT data is None or empty")
    
    if protein_id not in plddts_data and logger is not None:
        logger.error(f"Protein ID '{protein_id}' not found in pLDDT data. Available IDs: {list(plddts_data.keys())}")
    
    if 'per_res_plddts_mean' not in plddts_data[protein_id] and logger is not None:
        logger.error(f"Missing 'per_res_plddts_mean' for protein {protein_id}")
    
    if 'per_res_plddts_cv' not in plddts_data[protein_id] and logger is not None:
        logger.error(f"Missing 'per_res_plddts_cv' for protein {protein_id}")
    
    mean_plddts = plddts_data[protein_id]['per_res_plddts_mean']
    cv_plddts = plddts_data[protein_id]['per_res_plddts_cv']
    
    if not mean_plddts or not cv_plddts and logger is not None:
        logger.error(f"Empty pLDDT data for protein {protein_id}")
    
    # print(f"Debug: pLDDT mean length: {len(mean_plddts)}, CV length: {len(cv_plddts)}")
    
    # # Ensure data length matches sequence length
    # if len(mean_plddts) != seq_length or len(cv_plddts) != seq_length:
    #     print(f"Warning: Data length mismatch. Sequence: {seq_length}, pLDDT mean: {len(mean_plddts)}, pLDDT CV: {len(cv_plddts)}")
    
    # Get domain segments for this protein
    protein_domains = domains_df[domains_df['Protein_ID'] == protein_id] if not domains_df.empty else pd.DataFrame()
    
    # Get InterPro matches
    if 'matches' not in interpro_data['results'][0] and logger is not None:
        logger.error("InterPro results missing 'matches' key")
    
    interpro_matches = interpro_data['results'][0]['matches']
    if interpro_matches is None:
        interpro_matches = []
    #     print("Warning: InterPro matches is None, using empty list")
    
    # print(f"Debug: Number of InterPro matches: {len(interpro_matches)}")
    
    # # Debug: Print match structure
    # if interpro_matches:
    #     print(f"Debug: First match keys: {interpro_matches[0].keys() if interpro_matches[0] else 'First match is None'}")
    #     if interpro_matches[0] and 'signature' in interpro_matches[0]:
    #         print(f"Debug: First match signature keys: {interpro_matches[0]['signature'].keys() if interpro_matches[0]['signature'] else 'Signature is None'}")

    
    # Calculate CV range for color scaling
    min_cv = min(cv_plddts)
    max_cv = max(cv_plddts)
    
    # Generate domain colors
    domain_colors = generate_domain_colors()

    seq_length = len(sequence)
    total_width = seq_length * 8  # 8px per residue
    
    # Start building HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Protein {protein_id} Domain Visualization</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin: 0 auto;
                overflow-x: auto;
            }}
            .protein-viz {{
                margin: 0px 0;
                position: relative;
            }}
            .row {{
                display: flex;
                margin: 2px 0;
                position: relative;
                flex-shrink: 0;
                width: fit-content;
            }}
            .residue {{
                width: 8px;
                height: 20px;
                margin: 0;
                cursor: crosshair;
                position: relative;
            }}
            .domain-row {{
                height: 20px;
                position: relative;
                margin: 5px 0;
                flex-shrink: 0;
                width: fit-content;
            }}
            .domain-segments-row {{
                height: 35px;
                position: relative;
                margin: 8px 0;
                flex-shrink: 0;
                width: fit-content;
            }}
            .domain-segment-box {{
                position: absolute;
                height: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: bold;
                color: white;
                text-shadow: 1px 1px 1px rgba(0,0,0,0.7);
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                border: 2px solid #000;
            }}
            .domain-segment {{
                position: absolute;
                height: 30px;
                border: 1px solid #333;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 10px;
                font-weight: bold;
                color: white;
                text-shadow: 1px 1px 1px rgba(0,0,0,0.7);
                overflow: hidden;
                white-space: nowrap;
            }}
            .separator-line {{
                position: absolute;
                width: 2px;
                background-color: black;
                top: calc( (35px / 2) );  /* 17.5px from the row’s top */
                height: calc(100% - 17.5px)
            }}
            .row-label {{
                position: absolute;
                left: -100px;
                width: 90px;
                text-align: right;
                font-size: 12px;
                font-weight: bold;
                top: 50%;
                transform: translateY(-50%);
            }}
            .tooltip {{
                position: absolute;
                background-color: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 8px;
                border-radius: 5px;
                font-size: 12px;
                z-index: 1000;
                max-width: 300px;
                display: none;
                pointer-events: none;
            }}
            .legend {{
            	margin-left: 110px;
                margin-top: 20px;
                display: flex;
                justify-content: left;
                align-items: center;
                gap: 100px;
            }}
            .legend-section {{
                display: flex;
                align-items: center;
                gap: 20px;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 12px;
            }}
            .legend-color {{
                width: 20px;
                height: 15px;
                border: 1px solid #333;
            }}
            .cv-scale {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .cv-gradient {{
                width: 150px;
                height: 15px;
                background: linear-gradient(to right, #ff0000, #00ff00);
                border: 1px solid #333;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="protein-viz" style="position: relative; margin-left: 120px; width: {{total_width}}px; min-width: {{total_width}}px;">
    """
    
    # Add domain segments row (first row)
    html_content += """
                <div class="row domain-segments-row">
                    <div class="row-label">PAE domains</div>
    """
    if not protein_domains.empty:
        for _, domain in protein_domains.iterrows():
            start = domain['Start'] - 1  # Convert to 0-based
            end = domain['End'] - 1
            width = (end - start + 0.75) * 8
            left = start * 8
            domain_num = domain['Domain']
            
            tooltip_text = f"Segment: {domain_num}<br>Position: {start+1}-{end+1}<br>Mean pLDDT: {domain.get('Mean_pLDDT', 'N/A')}<br>InterPro: {domain.get('InterPro', 'N/A')}"
            
            html_content += f"""
                    <div class="domain-segment-box" 
                         style="width: {width}px; left: {left}px;"
                         data-tooltip="{tooltip_text}">
                        {domain_num}
                    </div>
            """
    html_content += "</div>\n"
    
    # Add pLDDT mean row
    html_content += """
                <div class="row">
                    <div class="row-label">pLDDT Mean</div>
    """
    for i, plddt in enumerate(mean_plddts):
        color = alphafold_color(plddt)
        html_content += f"""
                    <div class="residue" style="background-color: {color};" 
                         data-tooltip="Position: {i+1}<br>pLDDT: {plddt:.2f}<br>Residue: {sequence[i] if i < len(sequence) else 'N/A'}"></div>
        """
    html_content += "</div>\n"
    
    # Add CV row
    html_content += """
                <div class="row">
                    <div class="row-label">pLDDT CV</div>
    """
    for i, cv in enumerate(cv_plddts):
        color = cv_color(cv, min_cv, max_cv)
        html_content += f"""
                    <div class="residue" style="background-color: {color};" 
                         data-tooltip="Position: {i+1}<br>CV: {cv:.4f}<br>Residue: {sequence[i] if i < len(sequence) else 'N/A'}"></div>
        """
    html_content += "</div>\n"
    
    # Add InterPro domain rows
    for idx, match in enumerate(interpro_matches):
        # Safely get domain information
        signature = match.get('signature', {})
        if not signature:
            continue
            
        domain_name = signature.get('name') or signature.get('accession', f'Domain_{idx+1}')
        description = signature.get('description', 'No description available')
        
        # Handle None values
        if domain_name is None:
            domain_name = f'Domain_{idx+1}'
        if description is None:
            description = 'No description available'
            
        color = domain_colors[idx % len(domain_colors)]
        
        # print(f"Debug: Processing domain {idx+1}: {domain_name}")
        
        html_content += f"""
                <div class="row domain-row">
                    <div class="row-label">{domain_name[:15]}...</div>
        """
        
        # Safely get locations
        locations = match.get('locations', [])
        if locations is None:
            locations = []
            
        for location in locations:
            if location is None:
                continue
                
            start = location.get('start', 1) - 1  # Convert to 0-based
            end = location.get('end', 1) - 1
            
            if start is None or end is None:
                continue
                
            width = (end - start + 1) * 8
            left = start * 8
            
            evalue = match.get('evalue', 'N/A')
            if evalue is None:
                evalue = 'N/A'
            
            tooltip_text = f"Domain: {domain_name}<br>Description: {description}<br>Position: {start+1}-{end+1}<br>E-value: {evalue}"
            
            html_content += f"""
                    <div class="domain-segment" 
                         style="background-color: {color}; width: {width}px; left: {left}px; height: 18px;"
                         data-tooltip="{tooltip_text}">
                        {domain_name[:20]}
                    </div>
            """
        
        html_content += "</div>\n"
    
    # Add separator lines for domain segments
    if not protein_domains.empty:
        for _, domain in protein_domains.iterrows():
            # Add line at start
            start_pos = (domain['Start'] - 1) * 8
            html_content += f'<div class="separator-line" style="left: {start_pos}px;"></div>\n'
            
            # Add line at end
            end_pos = domain['End'] * 8
            html_content += f'<div class="separator-line" style="left: {end_pos}px;"></div>\n'
    
    html_content += """
            </div>
            
            <div class="legend">
                <div class="legend-section">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #0053D6;"></div>
                        <span>pLDDT ≥90 (Very High)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #65CBF3;"></div>
                        <span>pLDDT 70-90 (Confident)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FFDB13;"></div>
                        <span>pLDDT 50-70 (Low)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FF7D45;"></div>
                        <span>pLDDT <50 (Very Low)</span>
                    </div>
                </div>
                
                <div class="cv-scale">
                    <span style="font-size: 11px;">Low CV</span>
                    <div class="cv-gradient"></div>
                    <span style="font-size: 11px;">High CV</span>
                </div>
            </div>
            
            <div class="tooltip" id="tooltip"></div>
        </div>
        
        <script>
            // Tooltip functionality
            const tooltip = document.getElementById('tooltip');
            const elements = document.querySelectorAll('[data-tooltip]');
            
            elements.forEach(element => {
                element.addEventListener('mouseenter', (e) => {
                    const tooltipText = e.target.getAttribute('data-tooltip');
                    tooltip.innerHTML = tooltipText;
                    tooltip.style.display = 'block';
                });
                
                element.addEventListener('mousemove', (e) => {
                    tooltip.style.left = e.pageX + 10 + 'px';
                    tooltip.style.top = e.pageY - 10 + 'px';
                });
                
                element.addEventListener('mouseleave', () => {
                    tooltip.style.display = 'none';
                });
            });
        </script>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # print(f"Visualization saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate protein domain visualization')
    parser.add_argument('interpro_json', help='InterPro results JSON file')
    parser.add_argument('protein_id', help='Protein ID to visualize')
    parser.add_argument('plddts_json', help='Per-residue pLDDT JSON file')
    parser.add_argument('domains_tsv', help='Domains TSV file')
    parser.add_argument('output_html', nargs='?', help='Output HTML file (optional)')
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if not args.output_html:
        args.output_html = f"{args.protein_id}_visualization.html"
    
    try:
        # Load InterPro data
        with open(args.interpro_json, 'r') as f:
            interpro_data = json.load(f)
        
        # Load pLDDT data
        with open(args.plddts_json, 'r') as f:
            plddts_data = json.load(f)
        
        # Load domains data
        try:
            domains_df = pd.read_csv(args.domains_tsv, sep='\t')
        except FileNotFoundError:
            print("Warning: Domains TSV file not found. Continuing without domain segments.")
            domains_df = pd.DataFrame()
        except Exception as e:
            print(f"Warning: Could not read domains TSV file: {e}")
            domains_df = pd.DataFrame()
        
        # Generate visualization
        create_interpro_html_visualization(interpro_data, args.protein_id, plddts_data, domains_df, args.output_html)
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in data - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

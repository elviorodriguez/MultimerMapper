import xml.etree.ElementTree as ET
from collections import defaultdict
import random

def parse_tsv(tsv_file, threshold=40):
    """Parse TSV file and return a dictionary of proteins and their domains."""
    data = defaultdict(list)
    protein_colors = {}  # Store a single color per protein
    backbone_lengths = {}  # Store backbone lengths separately
    
    with open(tsv_file, 'r') as f:
        header = next(f).strip().split('\t')
        has_name = 'Name' in header
        has_color = 'Color' in header
        
        for line in f:
            parts = line.strip().split('\t')
            while len(parts) < 7:
                parts.append('')
            protein_id, domain_num, start, end, mean_plddt, name, color = parts[:7]
            
            try:
                start = int(start)
                end = int(end)
                mean_plddt = float(mean_plddt)
            except ValueError:
                continue
            
            if protein_id not in protein_colors:
                protein_colors[protein_id] = f'#{random.randint(0, 0xFFFFFF):06x}'
            
            # Store backbone length regardless of filtering
            backbone_lengths[protein_id] = max(backbone_lengths.get(protein_id, 0), end)
            
            if mean_plddt >= threshold:
                if has_name and has_color:
                    data[protein_id].append({
                        'start': start,
                        'end': end,
                        'name': name.strip(),
                        'color': color.strip()
                    })
                else:
                    data[protein_id].append({
                        'start': start,
                        'end': end,
                        'name': f'Domain {domain_num}',
                        'color': protein_colors[protein_id]
                    })
    
    return data, backbone_lengths

def create_svg(domains_dict, backbone_lengths, output_file, keep_disorder=False):
    """Create SVG visualization from the parsed data."""
    scale = 1.5  # Scale for backbone and domain lengths
    name_offset = 70  # Space between names and backbones
    row_height = 40  # Space between rows for proteins
    backbone_thickness = 12  # Thickness of the backbone
    
    svg_width = int(100 + name_offset + (max(backbone_lengths.values(), default=0) * scale) + 50)  # Dynamic width
    svg_height = 80 + (len(backbone_lengths) * row_height)  # Dynamic height
    
    svg_root = ET.Element('svg', xmlns='http://www.w3.org/2000/svg')
    svg_root.set('width', f'{svg_width}px')
    svg_root.set('height', f'{svg_height}px')

    # Add title
    ET.SubElement(svg_root, 'text', {
        'x': '50',
        'y': '40',
        'font_size': '24',
        'font_family': 'Arial',
        'fill': '#333333'
    }).text = 'Protein Domain Visualizations'
    
    y_pos = 80  # Starting position for the first protein

    for protein_id, max_length in backbone_lengths.items():
        # Create group for this protein
        g = ET.SubElement(svg_root, 'g', transform=f'translate(50, {y_pos})')

        # Add protein name (left-aligned)
        ET.SubElement(g, 'text', {
            'x': '0',
            'y': '30',
            'font_size': '14',
            'font_family': 'Arial',
            'fill': '#000000',
            'text-anchor': 'start'
        }).text = protein_id

        # Draw backbone (gray rectangle)
        backbone_start = name_offset
        backbone_width = max_length * scale
        ET.SubElement(g, 'rect', {
            'x': str(backbone_start),
            'y': '20',
            'width': str(backbone_width),
            'height': str(backbone_thickness),
            'rx': '3',
            'fill': '#AFABAB',
            'stroke': '#000000',
            'stroke-width': '1'
        })

        # Draw domains if present
        for domain in domains_dict.get(protein_id, []):
            if keep_disorder or domain['name']:
                start = domain['start']
                end = domain['end']
                name = domain['name']
                color = domain['color']

                x_pos = backbone_start + (start - 1) * scale
                width = (end - start + 1) * scale

                # Draw domain as a rounded rectangle
                ET.SubElement(g, 'rect', {
                    'x': str(x_pos),
                    'y': '14',
                    'width': str(width),
                    'height': '24',
                    'rx': '3',
                    'fill': color,
                    'stroke': '#000000',
                    'stroke-width': '1'
                })

                # Add domain name text (centered)
                text_x = x_pos + width / 2
                ET.SubElement(g, 'text', {
                    'x': str(text_x),
                    'y': '28',
                    'font_size': '10',
                    'font_family': 'Arial',
                    'fill': '#000000',
                    'text-anchor': 'middle',
                    'dominant-baseline': 'middle'
                }).text = name

        # Move to the next position for the next protein
        y_pos += row_height

    # Save SVG
    tree = ET.ElementTree(svg_root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate SVG visualization of protein domains from TSV file.')
    parser.add_argument('tsv_file', help='Input TSV file')
    parser.add_argument('-o', '--output', default='protein_domains.svg',
                        help='Output SVG file (default: protein_domains.svg)')
    parser.add_argument('--keep_disorder', action='store_true',
                        help='Keep the drawing of disordered domains (those without Name or Color)')
    parser.add_argument('--threshold', type=float, default=40, help='Threshold for mean_plddt (default: 40)')
    
    args = parser.parse_args()

    data, backbone_lengths = parse_tsv(args.tsv_file, args.threshold)
    create_svg(data, backbone_lengths, args.output, keep_disorder=args.keep_disorder)

if __name__ == '__main__':
    main()

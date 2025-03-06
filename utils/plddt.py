
def interpolate_color(value, start_color, end_color):
    """Interpolate a color between start_color and end_color based on value (0 to 1 scale)."""
    start_rgb = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
    interpolated_rgb = tuple(
        round(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * value) for i in range(3)
    )
    return f"#{interpolated_rgb[0]:02X}{interpolated_rgb[1]:02X}{interpolated_rgb[2]:02X}"

def get_plddt_color(plddt: int|float, use_hex=True, use_discrete=False):
    """
    Get the color corresponding to the pLDDT value.
    :param plddt: The pLDDT value (0-100).
    :param use_hex: Whether to use hex color codes or color names.
    :param use_discrete: Whether to use discrete color mapping or gradient interpolation.
    """
    pLDDT_colors = ["darkblue", "lightblue", "yellow", "orange", "red"]
    if use_hex:
        pLDDT_colors = ["#00008B", "#ADD8E6", "#FFFF00", "#FFA500", "#FF0000"]

    if use_discrete:
        # Original discrete mapping
        if plddt >= 90:
            return pLDDT_colors[0]
        elif plddt >= 70:
            return pLDDT_colors[1]
        elif plddt >= 50:
            return pLDDT_colors[2]
        elif plddt >= 40:
            return pLDDT_colors[3]
        else:
            return pLDDT_colors[4]
    else:
        # Interpolated gradient mapping
        if plddt >= 90:
            return pLDDT_colors[0]
        elif plddt < 40:
            return pLDDT_colors[-1]
        else:
            # Determine the range
            ranges = [(90, 70), (70, 50), (50, 40)]
            for i, (upper, lower) in enumerate(ranges):
                if upper > plddt >= lower:
                    # Scale the pLDDT value within this range (0 to 1)
                    scale_value = (plddt - lower) / (upper - lower)
                    return interpolate_color(scale_value, pLDDT_colors[i+1], pLDDT_colors[i])

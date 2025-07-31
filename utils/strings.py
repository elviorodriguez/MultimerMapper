import re

# To match JSON files with PDB files
from difflib import SequenceMatcher

# To select pdbs that match the .json filename      
def find_most_similar(query_string, string_list):
    '''
    Find the most similar string in a list based on token-based similarity.

    Parameters:
    - query_string (str): The query string to find a match for.
    - string_list (list of str): A list of strings to compare against the query string.

    Returns:
    str: The most similar string in the provided list.

    Example:
    >>> file_list = [
    ...    'YNG2__vs__YNG2L_relaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_002_alphafold2_multimer_v3_model_3_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_003_alphafold2_multimer_v3_model_5_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_004_alphafold2_multimer_v3_model_2_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_005_alphafold2_multimer_v3_model_4_seed_000.pdb'
    ... ]
    >>> query_file = 'YNG2__vs__YNG2L_scores_rank_005_alphafold2_multimer_v3_model_4_seed_000.json'
    >>> most_similar = find_most_similar(query_file, file_list)
    >>> print(f"The most similar file to '{query_file}' is: {most_similar}")
    
    NOTE: It always will prefer relaxed PDBs over unrelaxed (smaller difference).
    '''
    similarities = [SequenceMatcher(None, query_string, s).ratio() for s in string_list]
    most_similar_index = similarities.index(max(similarities))
    return string_list[most_similar_index]


########################## Helper functions ######################################

def find_all_indexes(string_list, target_string):
    """
    Returns a list of indexes of all occurrences of the target string in the list of strings.
    
    Parameters:
        string_list (list of str): The list of strings to search within.
        target_string (str): The string to find the indexes of.
        
    Returns:
        list of int: The list of indexes where the target string occurs.
    """
    return [index for index, value in enumerate(string_list) if value == target_string]



def longest_between(text: str, start: str, end: str) -> str:
    """
    Return the longest substring in `text` found between `start` and `end`.
    If no match is found, returns an empty string.
    """
    # build regex: escape the delimiters, capture anything (including newlines) in between
    pattern = re.escape(start) + r'(.*?)' + re.escape(end)
    # find all non-overlapping matches
    matches = re.findall(pattern, text, flags=re.DOTALL)
    # return the longest match (or '' if none)
    return max(matches, key=len) if matches else ''


def padded_flag(label: str, width: int, pad_char: str = '-') -> str:
    """
    Center the `label` inside a field of total length `width`,
    padded on both sides with `pad_char`. If width is smaller
    than the label length, returns the label unchanged.
    """
    label_length = len(label)
    if label_length >= width:
        return label
    total_pad = width - label_length
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return pad_char * left_pad + label + pad_char * right_pad


import re
from typing import List

def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text, leaving only the visible content.
    
    Args:
        text: String that may contain HTML tags
        
    Returns:
        String with HTML tags removed
    """
    # Remove HTML tags using regex
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text

def get_visual_line_lengths(text: str) -> List[int]:
    """
    Get the visual lengths of all lines in HTML text, ignoring HTML tags.
    Lines are separated by <br> tags or string boundaries.
    
    Args:
        text: HTML text string
        
    Returns:
        List of integers representing visual length of each line
    """
    # Split by <br> tags (case insensitive)
    lines = re.split(r'<br\s*/?>', text, flags=re.IGNORECASE)
    
    # Calculate visual length for each line (after stripping HTML tags)
    visual_lengths = []
    for line in lines:
        clean_line = strip_html_tags(line)
        visual_lengths.append(len(clean_line))
    
    return visual_lengths

def get_max_visual_line_length(text: str) -> int:
    """
    Get the maximum visual line length in HTML text, ignoring HTML tags.
    
    Args:
        text: HTML text string
        
    Returns:
        Integer representing the maximum visual line length
    """
    lengths = get_visual_line_lengths(text)
    return max(lengths) if lengths else 0

def get_longest_visual_line(text: str) -> str:
    """
    Get the visually longest line from HTML text (after stripping HTML tags).
    
    Args:
        text: HTML text string
        
    Returns:
        The longest line with HTML tags stripped
    """
    # Split by <br> tags (case insensitive)
    lines = re.split(r'<br\s*/?>', text, flags=re.IGNORECASE)
    
    if not lines:
        return ""
    
    # Find the line with maximum visual length
    max_line = ""
    max_length = 0
    
    for line in lines:
        clean_line = strip_html_tags(line)
        if len(clean_line) > max_length:
            max_length = len(clean_line)
            max_line = clean_line
    
    return max_line

def padded_flag_html_aware(label: str, width: int, pad_char: str = '-') -> str:
    """
    Center the `label` inside a field of total length `width`,
    padded on both sides with `pad_char`. Width is calculated
    based on visual length (ignoring HTML tags in width calculation).
    
    Args:
        label: The label text (may contain HTML tags)
        width: Target visual width
        pad_char: Character to use for padding
        
    Returns:
        Padded string
    """
    # Calculate visual length of label (without HTML tags)
    visual_label_length = len(strip_html_tags(label))
    
    if visual_label_length >= width:
        return label
    
    total_pad = width - visual_label_length
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    
    return pad_char * left_pad + label + pad_char * right_pad

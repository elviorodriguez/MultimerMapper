
# Function to print progress
def print_progress_bar(current, total, text = "", progress_length = 40):
    '''
    Prints a progress bar:
        
        Progress{text}: [===--------------------------] 10.00%
        Progress{text}: [=============================] 100.00%
        Progress{text}: [=========--------------------] 33.00%
    
    Parameters:
    - current (float):
    - total (float): 
    - progress_length (int): length of the full progress bar.
    - text (str):
        
    Returns:
    - None    
    '''
    try:
        percent = current / total * 100
        progress = int(progress_length * current / total)
    except ZeroDivisionError:
        percent = 100
        progress = int(progress_length)
    except Exception as e:
        print("ERROR: ")
        return f"Progress{text}: UNKNOWN ERROR ({e})"
    progress_bar_template = "[{:<" + str(progress_length - 1) + "}]"
    
    if current >= total:
        progress_bar = progress_bar_template.format("=" * (progress_length - 1))
    else:
        progress_bar = progress_bar_template.format("=" * progress + "-" * (progress_length - progress - 1))
        
    return f"Progress{text}: {progress_bar} {percent:.2f}%"

if __name__ == "__main__":
    print_progress_bar(current = [], total = [], text = "", progress_length = 40)
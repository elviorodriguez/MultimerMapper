
import tempfile
import shutil
import atexit
import os

# To remove setup and remove temporal directories
def setup_temp_dir(logger):
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    logger.debug(f'Temporary directory created at: {temp_dir}')
    
    # Ensure the directory is cleaned up when the program exits
    def cleanup_temp_dir():
        shutil.rmtree(temp_dir)
        logger.debug(f'Temporary directory {temp_dir} deleted')
    
    # Register the cleanup function to be called on program exit
    atexit.register(cleanup_temp_dir)
    
    return temp_dir

# To remove temporal files
def setup_temp_file(logger, file_path='temp-plot.html'):
    # Ensure the file is cleaned up when the program exits
    def cleanup_temp_file():
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f'Temporary file {file_path} deleted')
    
    # Register the cleanup function to be called on program exit
    atexit.register(cleanup_temp_file)
    
    return file_path
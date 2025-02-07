import gdown
import zipfile
import os


def download_zip_from_drive(file_id, output_path):   
    url = f'https://drive.google.com/uc?export=download&id={file_id}'    
    gdown.download(url, output_path, quiet=False)


def unzip_file(zip_path, output_folder):
    # Check if the zip file exists
    if not os.path.exists(zip_path):
        print(f"The file '{zip_path}' does not exist.")
        return
    
    # Create the output folder (data) if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")
    
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
        print(f"File extracted to '{output_folder}'.")


if __name__ == "__main__":  

    #for training data
    file_id = '1uskxuu1Y49NeCTSTw68XWW9gaGRHJbmj'  
    zip_output_path = 'train.zip' 
    extract_to_folder = 'data/train'  

    download_zip_from_drive(file_id, zip_output_path)    
    unzip_file(zip_output_path, extract_to_folder)

     #for testing data
    file_id = '12YmYWjw3W48AM1QWZFiGTP8JQPsymgdt'  
    zip_output_path = 'test.zip' 
    extract_to_folder = 'data/test' 
    
    download_zip_from_drive(file_id, zip_output_path)    
    unzip_file(zip_output_path, extract_to_folder)

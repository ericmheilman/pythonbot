#!/bin/sh
import os
import requests
import zipfile

# Function to download ChromeDriver
def download_chromedriver():
    url = "https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip"  # Adjust for your OS (linux64 for Linux)
    driver_zip = "chromedriver.zip"
    
    # Download ChromeDriver
    print(f"Downloading ChromeDriver from {url}...")
    response = requests.get(url)
    
    # Save it to a file
    with open(driver_zip, "wb") as file:
        file.write(response.content)
    
    # Extract the ZIP file
    with zipfile.ZipFile(driver_zip, 'r') as zip_ref:
        zip_ref.extractall("./")  # Extract to the current directory

    # Set permissions to make the driver executable
    os.chmod("chromedriver", 0o755)
    
    # Clean up the ZIP file
    os.remove(driver_zip)

    print("ChromeDriver downloaded and extracted successfully.")

# Run the function
if __name__ == "__main__":
    download_chromedriver()


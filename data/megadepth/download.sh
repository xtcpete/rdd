#!/bin/bash

# Base URL for the files
url_base="https://cvg-data.inf.ethz.ch/megadepth/"

# Define the tar files and their corresponding destination directories
declare -A file_mappings=(
    ["Undistorted_SfM.tar.gz"]="Undistorted_SfM/"
    ["depth_undistorted.tar.gz"]="depth_undistorted/"
    ["scene_info.tar.gz"]="scene_info/"
)

# Download, extract, and move files
for tar_name in "${!file_mappings[@]}"; do
    echo "downloading ${tar_name}"

    out_name="${file_mappings[$tar_name]}"
    
    # Full path of the tar.gz file
    tar_path="./${tar_name}"
    
    # Download the file
    wget "${url_base}${tar_name}" -O "$tar_path"
    
    # Check if download was successful
    if [ $? -ne 0 ]; then
        echo "Failed to download $tar_name"
        exit 1
    fi
    
    # Extract the tar.gz file
    tar -xzf "$tar_path" -C "./"
    
    # Remove the tar.gz file after extraction
    rm "$tar_path"
    
    # Move the extracted folder to the desired location
    extracted_folder="${tar_name%%.*}"  # Get the folder name (remove .tar.gz)
    mv "$tmp_dir/$extracted_folder" "$tmp_dir/$out_name"
done
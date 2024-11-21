import os
import pandas as pd

# Function to find all comparison_results.xlsx files in subdirectories with "SVM" in the folder name
def find_comparison_files(root_dir):
    files = []
    for subdir, dirs, files_in_dir in os.walk(root_dir):
        for file in files_in_dir:
            if file == "comparison_results.xlsx":
                # Check if "SVM" is in the folder name
                folder_name = os.path.basename(subdir)
                if "SVM" in folder_name:
                    files.append(os.path.join(subdir, file))
    return files

# Function to merge the Excel files
def merge_comparison_files(files):
    # Create an empty DataFrame to hold the merged data
    merged_df = pd.DataFrame()
    
    for file in files:
        # Read the file
        df = pd.read_excel(file)

        # Get the folder name
        folder_name = os.path.basename(os.path.dirname(file))

        # Add the folder name as the first column for each row of the data
        df.insert(0, 'Folder Name', folder_name)

        # Select the first 5 columns of data (Model, Accuracy, Recall, Precision, F1 Score)
        df = df.iloc[:, :6]  # Includes Folder Name and the 5 data columns

        # Merge into the final DataFrame
        merged_df = pd.concat([merged_df, df], axis=0, ignore_index=True)

    return merged_df

# Function to save the merged data to a new Excel file
def save_merged_file(merged_df, output_path):
    merged_df.to_excel(output_path, index=False)

# Path to the root directory where the folders are located
root_dir = r"D:\University\Semester 3\Applied Machine Learning\Practice2\ML-Project-Breast-Cancer\results"  # Replace with the correct path

# Find all comparison_results.xlsx files in the folders with "SVM" in their name
comparison_files = find_comparison_files(root_dir)

# Merge the files
merged_data = merge_comparison_files(comparison_files)

# Save the merged data to a new Excel file
output_file = "merged_comparison_results.xlsx"  # Output file path
save_merged_file(merged_data, output_file)

print(f"Merged file saved as {output_file}")
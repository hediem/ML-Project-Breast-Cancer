import os

def count_and_append_rules_in_txt_files(root_dir):
    # Walk through all subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)

                # Read the file content
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Count all rules, regardless of depth
                rule_count = sum(1 for line in lines if "|---" in line)

                # Append the rule count at the end
                with open(file_path, 'a') as f:
                    f.write(f"\nNumber of rules: {rule_count}\n")

                print(f"Processed file: {file_path} - Rules counted: {rule_count}")

# Specify the root directory containing the folders with .txt files
root_directory = r"D:\University\Semester 3\Applied Machine Learning\Practice2"  # Replace with the correct path

# Run the function
count_and_append_rules_in_txt_files(root_directory)

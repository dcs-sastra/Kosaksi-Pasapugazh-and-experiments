import os
import pandas as pd
import argparse

def parse_args():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert Excel files to CSV files while maintaining directory structure.')
    parser.add_argument('-i', '--input-directory', type=str, required=True, help='The input directory containing Excel files.')
    parser.add_argument('-o', '--output-directory', type=str, required=True, help='The output directory for CSV files.')

    # Parse arguments
    return parser.parse_args()

def convert_excel_to_csv(input_directory, output_directory):
    # Traverse the directory structure
    for subdir, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(('.xlsx', '.xls')):
                # Form full file paths
                file_path = os.path.join(subdir, file)
                
                # Read the Excel file
                if file.endswith('xlsx'):
                    df = pd.read_excel(file_path, engine='openpyxl')
                elif file.endswith('xls'):
                    df = pd.read_html(file_path)[0]
                
                # Create the corresponding output path
                relative_path = os.path.relpath(subdir, input_directory)
                output_subdir = os.path.join(output_directory, relative_path)
                
                # Ensure the output subdirectory exists
                os.makedirs(output_subdir, exist_ok=True)
                
                # Form the output file path
                csv_file = os.path.splitext(file)[0] + '.csv'
                output_file_path = os.path.join(output_subdir, csv_file)
                
                # Write the CSV file
                df.to_csv(output_file_path, index=False)

def main():
    args = parse_args()
    # Convert all Excel files to CSV
    convert_excel_to_csv(
        args.input_directory, 
        args.output_directory
    )

if __name__ == '__main__':
    main()

import re

# Function to extract training information and epoch metrics from log content
def parse_log_file(log_file_path, output_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Regular expressions to find specific training status and epoch completion statistics
    training_status_pattern = re.compile(r"Training BERT \+ Doc2Vec features with Dimension-\d+")
    epoch_completion_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} Epoch \d+/\d+, (Train|Val) (Loss|Accuracy): [\d.]+, (Train|Val) (Loss|Accuracy): [\d.]+)")

    # Extracting information
    training_status_info = training_status_pattern.findall(log_content)
    epoch_completion_info = epoch_completion_pattern.findall(log_content)

    # Write results to output file
    with open(output_file_path, 'w') as output_file:
        for info in training_status_info:
            output_file.write(info + '\n')

        for data in epoch_completion_info:
            output_file.write(data[0] + '\n')

# Example usage
log_file_path = '/root/yyx/Multiple_Detection/logs/train_BERT_Doc2vec.log'
output_file_path = '/root/yyx/Multiple_Detection/logs/BERT_doc2vec.txt'
parse_log_file(log_file_path, output_file_path)

import random

def create_random_unlabelled_test_file(original_file, output_file, sample_size=100000):
    """
    Randomly samples log messages from the original file
    and writes them as unlabelled logs into a new test file.
    """
    with open(original_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Randomly sample without replacement
    sampled_lines = random.sample(lines, min(sample_size, len(lines)))

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in sampled_lines:
            parts = line.strip().split(" ", 1)
            if len(parts) > 1:
                log_message = parts[1]  # exclude label
                outfile.write(log_message + "\n")

    print(f" Created random unlabelled test file: {output_file} with {len(sampled_lines)} lines.")
create_random_unlabelled_test_file(
    original_file="/content/drive/MyDrive/BGL.log",
    output_file="/content/drive/MyDrive/New_BGL.log",
    sample_size=100000
)

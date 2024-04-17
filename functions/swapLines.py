def swapLines(output_file_path, reference_file_path):
    # Read the lines from the reference file
    with open(reference_file_path, 'r', encoding='utf-8') as ref_file:
        reference_lines = ref_file.readlines()

    # Read the lines from the output file, and fix them if needed
    with open(output_file_path, 'r', encoding='utf-8') as output_file:
        output_lines = output_file.readlines()
    count = 0
    # Iterate through each line in the output file
    for i in range(len(output_lines)):
        line = output_lines[i]
        # Check if the line contains any English characters
        if any(((char >='a' and char<='z') or (char>='A' and char<='Z')) for char in line):
            count=count+1
            print(line)
            # Replace the line with the corresponding line from the reference file
            if i < len(reference_lines):
                output_lines[i] = reference_lines[i]

    # Write the fixed lines back to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.writelines(output_lines)

    print(count)
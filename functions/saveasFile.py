import os

def save(textArray, location, fileName):
    file_path = f"{location}/{fileName}.txt"

    # Check if the file already exists
    counter = 1
    while os.path.exists(file_path):
        file_path = f"{location}/{fileName}_{counter}.txt"
        counter += 1

    os.makedirs(location, exist_ok=True)
    
    # Open file in append mode
    with open(file_path, "a", encoding="utf-8") as file:
        for line in textArray:
            file.write(line + "\n")
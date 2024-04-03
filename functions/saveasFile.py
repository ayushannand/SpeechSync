import os

# If we want to save the file as a single string, we can pass as string as true
def save(textArray, location, fileName, asString=False):
    file_path = f"{location}/{fileName}.txt"
    counter = 1
    while os.path.exists(file_path):
        file_path = f"{location}/{fileName}_{counter}.txt"
        counter += 1
    os.makedirs(location, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as file:
        for line in textArray:
            if asString:
                file.write(line)
            else:
                file.write(line + "\n")
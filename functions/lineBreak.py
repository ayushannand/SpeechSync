# This is a helper function which will split the paragraph into lines, and return it.
import re 
def getLines(paragraph, delimiter):
    lines = paragraph.split(delimiter)

    # Remove leading and trailing spaces and periods
    cleanLines = []
    for line in lines:
        cleanLine = re.sub(r'^[ .]*', '', line)
        cleanLine = re.sub(r'[ .]*$', '', cleanLine)
        if cleanLine!='':
            cleanLines.append(cleanLine)
        
    return cleanLines
def postpHindi(lines):
    res = []
    for line in lines:
        # Remove trailing spaces
        line = line.rstrip()
        # Check if line ends with a period or pipe
        if line.endswith('.') or line.endswith('।') or line.endswith('?'):
            res.append(line)
        else:
            # If line does not end with period or pipe, add pipe at the end
            res.append(line + '।')
    return res

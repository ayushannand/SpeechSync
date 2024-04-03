import subprocess

def getScore(ref, out, resultName, comments = ""):
    command = f"sacrebleu {ref} -i {out} -m bleu -b -w 4"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    bleu_score = result.stdout.strip()
    with open('testing/Score/scores.txt', "a") as file:
        file.write(resultName + " - " + bleu_score +  ";\t" + comments +  "\n")

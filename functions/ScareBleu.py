import subprocess
import tempfile

def getScore(ref, out, resultName, stringfiy=False,comments=""):
    if stringfiy:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_ref_file, \
                tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_out_file:
            with open(ref, 'r') as ref_file:
                ref_content = ref_file.read().replace('\n', '')
                temp_ref_file.write(ref_content)
            with open(out, 'r') as out_file:
                out_content = out_file.read().replace('\n', '')
                temp_out_file.write(out_content)
        command = f"sacrebleu {temp_ref_file.name} -i {temp_out_file.name} -m bleu -b -w 4"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        bleu_score = result.stdout.strip()
        with open('testing/Score/scores.txt', "a") as file:
            file.write(resultName + " - " + bleu_score + ";\t" + comments + "\n")
        temp_ref_file.close()
        temp_out_file.close()
        return bleu_score
    else:
        command = f"sacrebleu {ref} -i {out} -m bleu -b -w 4"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        bleu_score = result.stdout.strip()
        with open('testing/Score/scores.txt', "a") as file:
            file.write(resultName + " - " + bleu_score +  ";\t" + comments +  "\n")
        return bleu_score

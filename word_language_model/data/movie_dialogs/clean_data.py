import spacy
import re

nlp = spacy.load('en_core_web_md')

BREAK = "<BRK>"
ori_file = "movie_lines.txt"
train_file = "train_file.txt"
valid_file = "valid_file.txt"
test_file = "test_file.txt"

"""TODO:
	1. add "b" to regex
	2. trim to say, 20 words
"""

break_signal = r"!|\?|\,|\.|\:|\;|\-\-"
# remove_signal = r"""'|\"|\.\.\.|\~|\(|\)|\%|\$|\#|\@|\&|\*|\+|\=|\^|\<|\>"""
remove_signal = r"""'|\"|\~|\(|\)|\%|\$|\#|\@|\&|\*|\+|\=|\^|\<|\>"""

def tokenize(line):
    line = re.sub(remove_signal, "", line)
    doc = nlp(line)
    annotate = []
    for token in doc:
        brk = True
        for c in token.text:
            if c.isalnum():
                brk = False
        if not brk:
            annotate.append(token.text)
        else:
            if annotate[len(annotate)-1] != BREAK:
                annotate.append(BREAK)
    return " ".join([t for t in annotate])

def get_sent(line):
    # find the last occurance of "+++$+++"
    return line[line.rfind("+++$+++")+8:].lower()

def annotate(line):
    line = re.sub(remove_signal, "", line)
    # not append BREAK to the end of sentence because we will append a EOS signal
    try:
        if line[-3] in break_signal or line[-3] == "-":#the last two elements are "\n"
            line = line[:-3] + "\n"
    except:
        pass
    line = re.sub(break_signal, BREAK, line)
    line = re.sub("-", "", line)#take "--" as a break signal, so to take care of cases like "---"
    #print(line)
    return line

def main():
    #num_lines = !wc -l ori_file
    #num_lines = int(num_lines[0].split()[0])
    num_lines = 304713
    training = open(train_file, "w")
    validing = open(valid_file, "w")
    testing = open(test_file, "w")
    writing_to_file = training
    with open(ori_file, "r", encoding='ISO-8859-1') as f:
        i = 0
        for line in f:
            line = get_sent(line)
            line = tokenize(line)
            # line = annotate(line)
            if i <= 0.8 * num_lines:
                training.write(line)
            elif i > 0.8 * num_lines and i <= 0.9 * num_lines:
                validing.write(line)
            else:
                testing.write(line)
            i += 1
            if i % 5000 == 0:
                print(i)
    training.close()
    validing.close()
    testing.close()

main()

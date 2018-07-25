import spacy
import re

nlp = spacy.load('en_core_web_md')

BREAK = "<BRK>"
ori_file = "movie_lines.txt"
train_file = "train_file"
valid_file = "valid_file"
test_file = "test_file"

break_signal = r"!|\?|\,|\.|\:|\;"
remove_signal = r"""'|\"|\-|\...|\~|\(|\)|\%|\$|\#|\@|\&|\*|\+|\=|\^|\<|\>"""

def tokenize(line):
	doc = nlp(line)
	return " ".join([token.text for token in doc])

def get_sent(line):
	# find the last occurance of "+++$+++"
	return line[line.rfind("+++$+++")+8:]

def annotate(line):
	line = re.sub(remove_signal, "", line)
	# not append BREAK to the end of sentence because we will append a EOS signal
	if line[-1] in break_signal:
		line = line[:-1]
	line = re.sub(break_signal, BREAK, line)
	return line

def main():
	# num_lines = !wc -l ori_file
	# num_lines = int(num_lines[0].split()[0])
	num_lines = 304713
	training = open("train_file", "w")
	validing = open("valid_file", "w")
	testing = open("test_file", "w")
	writing_to_file = training
	with open(ori_file, "r") as f:
		i = 0
		for line in f:
			line = get_sent(line)
			line = tokenize(line)
			line = annotate(line)
			if i <= 0.8 * num_lines:
				training.write(line)
			elif i > 0.8 * num_lines and i <= 0.9 * num_lines:
				validing.write(line)
			else:
				testing.write(line)
			i += 1
	training.close()
	valiing.close()
	testing.close()





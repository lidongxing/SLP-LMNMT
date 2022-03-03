import json
import csv
import codecs

src_lines,tgt_lines = [],[]
es2en_file_path = 'test2010.ro' #4098080


en1_file_path = 'test2010.en'

with open(es2en_file_path) as f1:
	# src_lines.append('es')
	for line in f1:
		line = line.strip()
		# if line.startswith('"'):continue
		src_lines.append(line)


# tgt_file_path = 'corpus.en1'
with open(en1_file_path) as f5:
	# tgt_lines.append('en')
	for line in f5:
		line = line.strip()
		tgt_lines.append(line)

print(len(src_lines),len(tgt_lines))
print(src_lines[:5])
sources = []
targets = []
for src_line,tgt_line in zip(src_lines,tgt_lines):
	src_line = src_line.strip()
	tgt_line = tgt_line.strip()
	sources.append(src_line)
	targets.append(tgt_line)
print(len(sources),len(targets))

with open('test2010_roen.tsv', 'w') as f:
	tsv_w = csv.writer(f, delimiter='\t', lineterminator='\n')
	# tsv_w.writerow(['it','en'])
	for num in range(len(src_lines)):
		tsv_w.writerow([sources[num],targets[num]])

# 1900875 it
# 399253 ro
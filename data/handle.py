import os


aa = ''
bb = []
with open('val.txt', 'r') as f:
	lines = f.readlines()
	for idx,line in enumerate(lines):
		splited = line.split()
		if len(splited) == 1:
			if line[-4:-1] == 'jpg':
				bb.append(aa)
				aa = 'data/val/'
			aa += line[:-1] + ' '
		if len(splited) == 10:
			aa += splited[0] + ' '
			aa += splited[1] + ' '
			aa += splited[2] + ' '
			aa += splited[3] + ' '
			aa += '1 '

with open('val_rewrite.txt', 'w') as f:
	for x in bb:
		f.write(x)
		f.write('\n')

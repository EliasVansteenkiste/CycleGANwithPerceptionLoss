# written by Elias Vansteenkiste, August 29, 2017
import sys
import subprocess


original_path = sys.argv[1]
destination_path = sys.argv[2]

p_find = subprocess.Popen('ls -la '+str(original_path), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

filenames = []

for idx, filename in enumerate(p_find.stdout.readlines()):
	words = filename.rsplit()
	if idx == 0 or len(words)<3:
		continue
	if words[-1] not in ['.', '..']:
		filenames.append(words[-1])

filenames_sorted = sorted(filenames)
for idx, filename in enumerate(filenames_sorted):
	if idx%10 == 0:
		cmd = 'cp '+str(original_path)+'/'+filename+' '+str(destination_path)+'/'+filename
		print cmd
		p_cpy = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
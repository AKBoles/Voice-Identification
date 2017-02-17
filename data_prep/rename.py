import os
import sys

# change directory to first argument
original_dir = os.getcwd()
data_dir = sys.argv[1]
filetype = sys.argv[2]
os.chdir(data_dir)
print('Input data directory: %s' %data_dir)

# change each Speaker's flac file to flac#.flac
for f1 in os.listdir(os.getcwd()):
	os.chdir(data_dir + f1)
	c = 0
	for f3 in os.listdir(os.getcwd()):
		print(f3)
		if '.wav' in f3:
			os.rename(f3, f1 + '_' + str(c) + '.' + filetype)
			c = c + 1
			#if c < 10:
			#	os.rename(f3, '0' + str(c) + '.' + filetype)
			#	c = c + 1
			#else:
			#	os.rename(f3, str(c) + '.' + filetype)
			#	c = c + 1
	os.chdir(data_dir + f1)

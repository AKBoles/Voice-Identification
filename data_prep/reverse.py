import pydub
import os
import sys

# reverse all files in a directory, save them to new directory
def reverse_audio(orig, dest):
  os.chdir(orig)
  files = os.listdir(orig)
  waves = []
  for f in files:
    waves.append(pydub.AudioSegment.from_wav(f))
  os.chdir(dest)
  for f, w in zip(files, waves):
    rev = w.reverse()
    rev.export(f[:-4] + '_rev' + '.wav', 'wav')

if __name__ == "__main__":
  # if this script is being called by itself, need to specify the arguments to input into function
  orig = sys.argv[1]
  dest = sys.argv[2]
  reverse_audio(orig=orig, dest=dest)

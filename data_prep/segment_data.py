import os
import sys
import time
import speech_data
from pydub import AudioSegment as audio

# segment data function
def segment(data, seg_location, length):
  os.chdir(data)
  files = os.listdir(data)
  speakers = speech_data.get_speakers(data)
  waves = []
  num = {}
  for s in speakers:
    num[s] = 0
  c = 0
  for f in files: # grab all wave files in list
    waves.append(audio.from_wav(f))
    c = c + 1
  os.chdir(seg_location)
  for f,w in zip(files,waves): # need to segment the data into one second intervals
    begin = 0
    end = 1
    while (end*length) < int(w.duration_seconds):
      segment = w[begin*1000*length:end*1000*length]
      segment.export(speech_data.speaker(f) + '_' +  str(num[speech_data.speaker(f)]) + '.wav', 'wav')
      begin = begin + length
      end = end + length
      num[speech_data.speaker(f)] = num[speech_data.speaker(f)] + 1

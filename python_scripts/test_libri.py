import os
import sys
import time

# need to test 10 - person data set from LibriSpeech on all different types of networks
# change the following: layer size, dropout for each layer size
# currently keeping the file length constant at 2 seconds
# perform the above on different depth of networks (have 1, 2, 3 depth)

# python scripts with networks defined
libri = ['libri_1layer.py', 'libri_2layer.py', 'libri_3layer.py']

layer_size = 2
while layer_size < 130:
    dropout = 0.3
    while dropout < 1:
        for l in libri:
            os.system('python ' + l + ' ' + str(layer_size) + ' ' + str(dropout))
        dropout = dropout + 0.1
    layer_size = layer_size + 2

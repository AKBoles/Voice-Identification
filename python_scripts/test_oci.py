import os
import sys
import time

# need to test 10 - person data set from LibriSpeech on all different types of networks
# change the following: layer size, dropout for each layer size
# currently keeping the file length constant at 2 seconds
# perform the above on different depth of networks (have 1, 2, 3 depth)

# python scripts with networks defined
oci = ['oci_1layer.py', 'oci_2layer.py', 'oci_3layer.py']

layer_size = 2
while layer_size < 130:
    dropout = 0.3
    while dropout < 1:
        for o in oci:
            os.system('python ' + o + ' ' + str(layer_size) + ' ' + str(dropout))
        dropout = dropout + 0.1
    layer_size = layer_size + 2

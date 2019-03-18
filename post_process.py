"""
post_process.py
This is a tool to collect:
(2) CV and instantaneous bias values from log file
(3) FES info from the bias.npy file

Usage
-----
python post_process.py simulation.log bias.npy
"""
import sys
import os
import numpy as np
import csv

# set working directory and certain paths
work_dir = '5ns_96.0_narrow'
cv_file = os.path.join(work_dir,'20ns_96.0_5UG9.log')
bias_file = os.path.join(work_dir,'biases/bias_50918095053337_5001.npy')
output1 = os.path.join(work_dir,'cv_bias.dat')
output2 = os.path.join(work_dir,'fes_0.dat')

# collect cv and bias data
log = cv_file
with open(log) as results:
    lines = results.readlines()[57:]

position = []
added = []
for line in lines:
    line = line.strip('\n').split(':')
    if '(' in line[0]:
        line[0] = line[0].strip('(').strip(',)')
        position.append(float(line[0]))
    if 'bias added' == line[0]:
        added.append(float(line[1]))

# output the cv_bias trajectory
f = open(output1, 'w')
writer = csv.writer(f, delimiter='\t')
writer.writerows(zip(position,added))
f.close()

# collect free energy values
bias = bias_file
data = -1*np.load(bias)
min_v = -20
max_v = 20
x = []
for i in range(len(data)):
    x.append(round(min_v+i*(max_v-min_v)/len(data),7))

# output the FES file
f = open(output2, 'w')
writer = csv.writer(f, delimiter='\t')
writer.writerows(zip(x,data))
f.close()

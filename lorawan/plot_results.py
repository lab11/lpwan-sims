#! /usr/bin/env python3

# --- Processing ---
import numpy as np
import scipy.stats as st
import sys

# get result dict filename
if len(sys.argv) != 2:
    print("Invalid input. Expected:\n\t$ ./result_plot.py results.dict")
    sys.exit(1)

# read in dict
results = None
with open(sys.argv[1], 'r') as infile:
    results = eval(infile.read())

# print results to console
for node_count in sorted(results.keys()):
    average_prrs = []
    for basestation_count in sorted(results[node_count].keys()):
        result_array = results[node_count][basestation_count]
        prrs = [data[1] for data in result_array]
        average_prr = np.mean(prrs)
        ci = st.t.interval(0.95, len(prrs)-1, loc=np.mean(prrs), scale=st.sem(prrs))
        ci_lower = ci[0]
        ci_upper = ci[1]
        print("{:4} {:2} {:7.3f}% {:.3f}:{:.3f}".format(node_count, basestation_count-1, average_prr, ci_lower, ci_upper))

    print("")


# --- Plot ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

## Tableau Color Blind 10
#t10_blind = [(0.0, 0.41960, 0.64313), (1.0, 0.50196, 0.05490), (0.67058, 0.67058, 0.670588), 
#            (0.34901, 0.34901, 0.34901), (0.37254, 0.619607, 0.819607), (0.784313, 0.32156, 0.0), 
#            (0.53725, 0.53725, 0.537254), (0.63921, 0.78431, 0.925490), (1.0, 0.73725, 0.474509), 
#            (0.811764, 0.811764, 0.8117647)]
#
##Here are the default color index for 10 - they apply to some of the colorblind version
#blue = 0
#orange = 1
#green = 2
#red = 3
#purple = 4
#brown = 5
#pink = 6
#grey = 7
#yellow = 8
#cyan = 9

# create a plot of some specific results
w, h = matplotlib.figure.figaspect(0.4)
fig, ax = plt.subplots(dpi=300, figsize=(w,h))
plt.grid(True, which='major', ls='-.', alpha=0.5)

plt.xlim(1,10)
plt.xlabel('Number of Gateways', fontsize=24)
plt.xticks(fontsize=24)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

plt.ylim(0,1)
plt.ylabel('Packet Reception Rate', fontsize=24)
plt.yticks(fontsize=24)

# get data
for node_count in sorted(results.keys()):
    average_prrs = []
    for basestation_count in sorted(results[node_count].keys()):
        result_array = results[node_count][basestation_count]
        prrs = [data[1] for data in result_array]
        average_prr = np.mean(prrs)
        #ci = st.t.interval(0.95, len(prrs)-1, loc=np.mean(prrs), scale=st.sem(prrs))
        #ci_lower = ci[0]
        #ci_upper = ci[1]

        average_prrs.append(average_prr)

    data = np.array(average_prrs)
    plt.plot(np.arange(1, data.size+1), data, label='{} nodes'.format(node_count*2), linestyle='-', marker='.')

# place legend outside on right
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1.02,0.5), fontsize=20, ncol=1)
#lgd = plt.legend(loc=0, fontsize=24, ncol=1)

# save plot
outfilename = sys.argv[1].split('.')[0] + '.pdf'
plt.savefig(outfilename, bbox_inches='tight', format='pdf')


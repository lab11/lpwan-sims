#!/usr/bin/env python3
"""
 LoRaSim 0.2.1: simulate collisions in LoRa - directional nodes
 Copyright © 2016-2017 Thiemo Voigt <thiemo@sics.se> and Martin Bor <m.bor@lancaster.ac.uk>

 This work is licensed under the Creative Commons Attribution 4.0
 International License. To view a copy of this license,
 visit http://creativecommons.org/licenses/by/4.0/.

 Do LoRa Low-Power Wide-Area Networks Scale? Martin Bor, Utz Roedig, Thiemo Voigt
 and Juan Alonso, MSWiM '16, http://dx.doi.org/10.1145/2988287.2989163

 Mitigating Inter-Network Interference in LoRa Low-Power Wide-Area Networks,
 Thiemo Voigt, Martin Bor, Utz Roedig, and Juan Alonso, EWSN '17

 $Date: 2017-05-12 19:16:16 +0100 (Fri, 12 May 2017) $
 $Revision: 334 $

 This file is further modified as noted in https://github.com/lab11/lpwan-sims
"""

import simpy
import random
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import os


#
# check for collisions at base station
# Note: called before a packet (or rather node) is inserted into the list
def checkcollision(packet):
    col = 0 # flag needed since there might be several collisions for packet
    # lost packets don't collide
    if packet.lost:
       return 0
    if packetsAtBS[packet.bs]:
        for other in packetsAtBS[packet.bs]:
            if other.id != packet.nodeid:
               # simple collision
               if frequencyCollision(packet, other.packet[packet.bs]) \
                   and sfCollision(packet, other.packet[packet.bs]):
                   if full_collision:
                       if timingCollision(packet, other.packet[packet.bs]):
                           # check who collides in the power domain
                           c = powerCollision(packet, other.packet[packet.bs])
                           # mark all the collided packets
                           # either this one, the other one, or both
                           for p in c:
                               p.collided = 1
                               if p == packet:
                                   col = 1
                       else:
                           # no timing collision, all fine
                           pass
                   else:
                       packet.collided = 1
                       other.packet[packet.bs].collided = 1  # other also got lost, if it wasn't lost already
                       col = 1
        return col
    return 0

#
# frequencyCollision, conditions
#
#        |f1-f2| <= 120 kHz if f1 or f2 has bw 500 
#        |f1-f2| <= 60 kHz if f1 or f2 has bw 250 
#        |f1-f2| <= 30 kHz if f1 or f2 has bw 125
def frequencyCollision(p1,p2):
    if (abs(p1.freq-p2.freq)<=120 and (p1.bw==500 or p2.bw==500)):
        return True
    elif (abs(p1.freq-p2.freq)<=60 and (p1.bw==250 or p2.bw==250)):
        return True
    else:
        if (abs(p1.freq-p2.freq)<=30):
            return True
    return False

def sfCollision(p1, p2):
    if p1.sf == p2.sf:
        # p2 may have been lost too, will be marked by other checks
        return True
    return False

def powerCollision(p1, p2):
    powerThreshold = 6 # dB
    if abs(p1.rssi - p2.rssi) < powerThreshold:
        # packets are too close to each other, both collide
        # return both packets as casualties 
        return (p1, p2)
    elif p1.rssi - p2.rssi < powerThreshold:
        # p2 overpowered p1, return p1 as casualty
        return (p1,)
    # p2 was the weaker packet, return it as a casualty  
    return (p2,)

def timingCollision(p1, p2):
    # assuming p1 is the freshly arrived packet and this is the last check
    # we've already determined that p1 is a weak packet, so the only
    # way we can win is by being late enough (only the first n - 5 preamble symbols overlap)
    
    # assuming 8 preamble symbols
    Npream = 8
    
    # we can lose at most (Npream - 5) * Tsym of our preamble
    Tpreamb = 2**p1.sf/(1.0*p1.bw) * (Npream - 5)
    
    # check whether p2 ends in p1's critical section
    p2_end = p2.addTime + p2.rectime
    p1_cs = env.now + Tpreamb
    if p1_cs < p2_end:
        # p1 collided with p2 and lost
        return True
    return False

# this function computes the airtime of a packet
# according to LoraDesignGuide_STD.pdf
#
def airtime(sf,cr,pl,bw):
    H = 0        # implicit header disabled (H=0) or not (H=1)
    DE = 0       # low data rate optimization enabled (=1) or not (=0)
    Npream = 8   # number of preamble symbol (12.25  from Utz paper)

    if bw == 125 and sf in [11, 12]:
        # low data rate optimization mandated for BW125 with SF11 and SF12
        DE = 1
    if sf == 6:
        # can only have implicit header with SF6
        H = 1

    Tsym = (2.0**sf)/bw
    Tpream = (Npream + 4.25)*Tsym
    #print "sf", sf, " cr", cr, "pl", pl, "bw", bw
    payloadSymbNB = 8 + max(math.ceil((8.0*pl-4.0*sf+28+16-20*H)/(4.0*(sf-2*DE)))*(cr+4),0)
    Tpayload = payloadSymbNB * Tsym
    return Tpream + Tpayload

# Determine the maximum distance a packet can travel
# Uses the Hata Model for urban, medium-sized cities
# tx_power in dB, rx_sensitivity in dB (should be negative)
# returns distance in kilometers
def rangeEstimate(tx_power, rx_sensitivity):
    max_path_loss = tx_power - rx_sensitivity
    freq_mhz = 915
    distance_m = 0.0484418*math.exp(0.0724083*max_path_loss)/(freq_mhz**0.837107)*1000
    return distance_m

# Determine the power of a packet given the distance it has traveled
# Uses the Hata Model for urban, medium-sized cities
# tx_power in dB, distance in meters
def powerEstimate(tx_power, distance_m):
    freq_mhz = 915
    rx_power = 69.55 + 26.16*math.log10(freq_mhz) - 13.82*math.log10(100) + (44.9 - 6.55*math.log10(100))*math.log10(distance_m/1000) - (0.8 + (1.1*math.log10(freq_mhz)-0.7)*(1) - 1.56*math.log10(freq_mhz))
    return rx_power


#
# this function creates a BS
#
class myBS():
    def __init__(self, id):
        self.id = id
        self.x = 0
        self.y = 0

        # This is a hack for now
        global nrBS
        global maxDist
        global baseDist

        if (nrBS == 1 and self.id == 0):
            self.x = maxDist
            self.y = maxDist
        else:
            print("Too many base stations!!")
            sys.exit(1)

#        if (nrBS == 2 and self.id == 0):
#            self.x = maxDist
#            self.y = maxY
#
#        if (nrBS == 2 and self.id == 1):
#            self.x = maxDist + baseDist
#            self.y = maxY
#
#        if (nrBS == 3 and self.id == 0):
#            self.x = maxDist + baseDist
#            self.y = maxY
#
#        if (nrBS == 3 and self.id == 1):
#            self.x = maxDist 
#            self.y = maxY
#
#        if (nrBS == 3 and self.id == 2): 
#            self.x = maxDist + 2*baseDist
#            self.y = maxY
#
#        if (nrBS == 4 and self.id == 0): 
#            self.x = maxDist + baseDist
#            self.y = maxY 
#
#        if (nrBS == 4 and self.id == 1):
#            self.x = maxDist 
#            self.y = maxY
#
#        if (nrBS == 4 and self.id == 2): 
#            self.x = maxDist + 2*baseDist
#            self.y = maxY
#
#        if (nrBS == 4 and self.id == 3): 
#            self.x = maxDist + baseDist
#            self.y = maxY + baseDist 
#
#        if (nrBS == 5 and self.id == 0): 
#            self.x = maxDist + baseDist
#            self.y = maxY + baseDist 
#
#        if (nrBS == 5 and self.id == 1): 
#            self.x = maxDist 
#            self.y = maxY + baseDist 
#
#        if (nrBS == 5 and self.id == 2): 
#            self.x = maxDist + 2*baseDist
#            self.y = maxY + baseDist 
#
#        if (nrBS == 5 and self.id == 3): 
#            self.x = maxDist + baseDist
#            self.y = maxY 
#
#        if (nrBS == 5 and self.id == 4): 
#            self.x = maxDist + baseDist
#            self.y = maxY + 2*baseDist 
#
#
#        if (nrBS == 6): 
#            if (self.id < 3):
#                self.x = (self.id+1)*maxX/4.0
#                self.y = maxY/3.0
#            else:
#                self.x = (self.id+1-3)*maxX/4.0
#                self.y = 2*maxY/3.0
#
#        if (nrBS == 8): 
#            if (self.id < 4):
#                self.x = (self.id+1)*maxX/5.0
#                self.y = maxY/3.0
#            else:
#                self.x = (self.id+1-4)*maxX/5.0
#                self.y = 2*maxY/3.0
#
#        if (nrBS == 24): 
#            if (self.id < 8):
#                self.x = (self.id+1)*maxX/9.0
#                self.y = maxY/4.0
#            elif (self.id < 16):
#                self.x = (self.id+1-8)*maxX/9.0
#                self.y = 2*maxY/4.0
#            else:
#                self.x = (self.id+1-16)*maxX/9.0
#                self.y = 3*maxY/4.0
#
#        if (nrBS == 96): 
#            if (self.id < 24):
#                self.x = (self.id+1)*maxX/25.0
#                self.y = maxY/5.0
#            elif (self.id < 48):
#                self.x = (self.id+1-24)*maxX/25.0
#                self.y = 2*maxY/5.0
#            elif (self.id < 72):
#                self.x = (self.id+1-48)*maxX/25.0
#                self.y = 3*maxY/5.0
#            else:
#                self.x = (self.id+1-72)*maxX/25.0
#                self.y = 4*maxY/5.0

        
        print("BSx:", self.x, "BSy:", self.y)

        global graphics
        global pointsize
        if graphics:
            global ax
            if (self.id == 0):
                ax.add_artist(plt.Circle((self.x, self.y), pointsize*2, fill=True, color='cyan'))
                ax.add_artist(plt.Circle((self.x, self.y), maxDist, fill=False, color='black'))
           # if (self.id == 1):
           #     ax.add_artist(plt.Circle((self.x, self.y), 4, fill=True, color='red'))
           #     ax.add_artist(plt.Circle((self.x, self.y), maxDist, fill=False, color='red'))
           # if (self.id == 2):
           #     ax.add_artist(plt.Circle((self.x, self.y), 4, fill=True, color='green'))
           #     ax.add_artist(plt.Circle((self.x, self.y), maxDist, fill=False, color='green'))
           # if (self.id == 3):
           #     ax.add_artist(plt.Circle((self.x, self.y), 4, fill=True, color='brown'))
           #     ax.add_artist(plt.Circle((self.x, self.y), maxDist, fill=False, color='brown'))
           # if (self.id == 4):
           #     ax.add_artist(plt.Circle((self.x, self.y), 4, fill=True, color='orange'))
           #     ax.add_artist(plt.Circle((self.x, self.y), maxDist, fill=False, color='orange'))


# this function creates a node
#
class myNode():
    def __init__(self, id, period, dataRate, packetlen, myBS):
        global bs

        self.bs = myBS
        self.id = id
        self.period = period

        self.x = 0
        self.y = 0
        self.packet = []
        self.dist = []

        # this is very complex prodecure for placing nodes
        # and ensure minimum distance between each pair of nodes
        found = 0
        rounds = 0
        global nodes
        while (found == 0 and rounds < 100):
            a = random.random()
            b = random.random()
            if b<a:
                a,b = b,a
            posx = b*maxDist*math.cos(2*math.pi*a/b)+self.bs.x
            posy = b*maxDist*math.sin(2*math.pi*a/b)+self.bs.y
            if len(nodes) > 0:
                for index, n in enumerate(nodes):
                    dist = np.sqrt(((abs(n.x-posx))**2)+((abs(n.y-posy))**2)) 
                    # we set this so nodes can be placed everywhere
                    # otherwise there is a risk that little nodes are placed
                    # between the base stations where it would be more crowded
                    if dist >= 0: 
                        found = 1
                        self.x = posx
                        self.y = posy
                    else:
                        rounds = rounds + 1
                        if rounds == 100:
                            print("could not place new node, giving up")
                            exit(-2) 
            else:
                self.x = posx
                self.y = posy
                found = 1


        # create "virtual" packet for each BS
        global nrBS
        for i in range(0,nrBS):
            d = np.sqrt((self.x-bs[i].x)*(self.x-bs[i].x)+(self.y-bs[i].y)*(self.y-bs[i].y)) 
            self.dist.append(d)
            self.packet.append(myPacket(self.id, dataRate, packetlen, self.dist[i], i))
        print('node %d' %id, "x", self.x, "y", self.y, "dist: ", self.dist, "my BS:", self.bs.id)

        self.sent = 0

        # graphics for node
        global graphics
        global pointsize
        if graphics:
            global ax
            if self.bs.id == 0:
                    ax.add_artist(plt.Circle((self.x, self.y), pointsize, fill=True, color='blue'))
            #if self.bs.id == 1:
            #        ax.add_artist(plt.Circle((self.x, self.y), 2, fill=True, color='red'))
            #if self.bs.id == 2:
            #        ax.add_artist(plt.Circle((self.x, self.y), 2, fill=True, color='green'))
            #if self.bs.id == 3:
            #        ax.add_artist(plt.Circle((self.x, self.y), 2, fill=True, color='brown'))
            #if self.bs.id == 4:
            #        ax.add_artist(plt.Circle((self.x, self.y), 2, fill=True, color='orange'))


#
# this function creates a packet (associated with a node)
# it also sets all parameters, currently random
#
class myPacket():
    def __init__(self, nodeid, data_rate, plen, distance_m, bs):
        global nodes
        global dr_configurations
        global tx_power
        global rx_sensitivity

        # new: base station ID
        self.bs = bs
        self.nodeid = nodeid

        # pick configuration values based on data rate
        self.sf = dr_configurations[data_rate][0]
        self.bw = dr_configurations[data_rate][1]/1000
        self.cr = 1 # this means 4/5 coding rate

        #XXX: may want to add multiple channels in the future
        self.freq = 915000000 # defaults to 915 MHz

        # calculate packet strength at receiver
        rx_power = powerEstimate(tx_power, distance_m)
        self.rssi = rx_power

        # packet on-air time
        self.pl = plen
        self.symTime = (2.0**self.sf)/self.bw
        self.rectime = airtime(self.sf,self.cr,self.pl,self.bw)
        print("Airtime is: {}".format(self.rectime))

        # denote if packet is collided
        self.collided = 0
        self.processed = 0

        # mark the packet as lost when it's rssi is below the sensitivity
        self.lost = self.rssi < rx_sensitivity
        print("node {} bs {} lost {}".format(self.nodeid, self.bs, self.lost))


#
# main discrete event loop, runs for each node
# a global list of packet being processed at the gateway
# is maintained
#       
def transmit(env,node):
    while True:
        # time before sending anything (include prop delay)
        # send up to 2 seconds earlier or later
        yield env.timeout(random.expovariate(1.0/float(node.period)))

        # time sending and receiving
        # packet arrives -> add to base station

        node.sent = node.sent + 1

        global packetSeq
        packetSeq = packetSeq + 1

        global nrBS
        for bs in range(0, nrBS):
           if node in packetsAtBS[bs]:
                print("ERROR: packet already in")
           else:
                # adding packet if no collision
                if checkcollision(node.packet[bs]) == 1:
                    node.packet[bs].collided = 1
                else:
                    node.packet[bs].collided = 0
                packetsAtBS[bs].append(node)
                node.packet[bs].addTime = env.now
                node.packet[bs].seqNr = packetSeq
 
        # take first packet rectime        
        yield env.timeout(node.packet[0].rectime)

        # if packet did not collide, add it in list of received packets
        # unless it is already in
        for bs in range(0, nrBS):
            if node.packet[bs].lost:
                lostPackets.append(node.packet[bs].seqNr)
            else:
                if node.packet[bs].collided == 0:
                    if nrNetworks == 1:
                        packetsRecBS[bs].append(node.packet[bs].seqNr)
                    else:
                        # now need to check for right BS
                        #TODO: shouldn't this be adjusted for multiple base stations on a single network?
                        if node.bs.id == bs:
                            packetsRecBS[bs].append(node.packet[bs].seqNr)
                    # recPackets is a global list of received packets
                    # not updated for multiple networks        
                    if recPackets:
                        if recPackets[-1] != node.packet[bs].seqNr:
                            recPackets.append(node.packet[bs].seqNr)
                    else:
                        recPackets.append(node.packet[bs].seqNr)
                else:
                    # XXX only for debugging
                    collidedPackets.append(node.packet[bs].seqNr)

        # complete packet has been received by base station
        # can remove it

        for bs in range(0, nrBS):                    
            if node in packetsAtBS[bs]:
                packetsAtBS[bs].remove(node)
                # reset the packet
                node.packet[bs].collided = 0
                node.packet[bs].processed = 0


#
# "main" program
#

# Local Configurations
nrNodes = 1000
avgSendTime = 600*1000
simtime = 100*60*60*1000
packetLen = 21

# global configurations
nrBS = 1
full_collision = True
nrNetworks = 1
graphics = True
print("Nodes per base station:", nrNodes) 
print("AvgSendTime (exp. distributed):",avgSendTime)
print("Simtime: ", simtime)
print("nrBS: ", nrBS)
print("Full Collision: ", full_collision)
print("nrNetworks: ", nrNetworks)

# global stuff
nodes = []
packetsAtBS = []
env = simpy.Environment()

nrCollisions = 0
nrReceived = 0
nrProcessed = 0

# global value of packet sequence numbers
packetSeq = 0

# list of received packets
recPackets=[]
collidedPackets=[]
lostPackets = []

# set of [Spreading Factor, Bandwidth, Sensitivity] for each US data rate
dr_configurations = {
    0: [10, 125000, -132],
    1: [9,  125000, -129],
    2: [8,  125000, -126],
    3: [7,  125000, -123],
    4: [8,  500000, -119],
}

# figure out the minimal sensitivity for the given experiment
dataRate = 3
rx_sensitivity = dr_configurations[dataRate][2]
tx_power = 14

# uses Hata Model for maximum distance
maxDist = rangeEstimate(tx_power, rx_sensitivity)
print("Max distance: {} meters".format(maxDist))

# set a point size for graphics displays based on distance
pointsize = 0.01*maxDist

#XXX: for now we're running with a single channel. Should fix this maybe
# maximum number of packets the BS can receive at the same time
maxBSReceives = 1

# prepare graphics and add sink
if graphics:
    plt.ion()
    plt.figure()
    ax = plt.gcf().gca()

# list of base stations
bs = []

# list of packets at each base station, init with 0 packets
packetsAtBS = []
packetsRecBS = []
for i in range(0,nrBS):
    b = myBS(i)
    bs.append(b)
    packetsAtBS.append([])
    packetsRecBS.append([])

#TODO: adjust this setup
for i in range(0,nrNodes):
    # myNode takes period (in ms), base station id packetlen (in Bytes)
    # 1000000 = 16 min
    for j in range(0,nrBS):
        # create nrNodes for each base station
        node = myNode(i*nrBS+j, avgSendTime, dataRate, packetLen, bs[j])
        nodes.append(node)

        env.process(transmit(env,node))

#prepare show
if graphics:
    plt.xlim([0-0.1*maxDist, 2*maxDist+0.1*maxDist])
    plt.ylim([0-0.1*maxDist, 2*maxDist+0.1*maxDist])
    plt.draw()
    plt.show()  

# store nodes and basestation locations
with open('nodes.txt', 'w') as nfile:
    for node in nodes:
        nfile.write('{x} {y} {id}\n'.format(**vars(node)))

with open('basestation.txt', 'w') as bfile:
    for basestation in bs:
        bfile.write('{x} {y} {id}\n'.format(**vars(basestation)))

# start simulation
env.run(until=simtime)

#TODO: need to update these output statistics

# print stats and save into file
print("nr received packets (independent of right base station)", len(recPackets))
print("nr collided packets", len(collidedPackets))
print("nr lost packets (not correct)", len(lostPackets))

sum = 0
for i in range(0,nrBS):
    print("packets at BS",i, ":", len(packetsRecBS[i]))
    sum = sum + len(packetsRecBS[i])
print("sent packets: ", packetSeq)
print("overall received at right BS: ", sum)

sumSent = 0
sent = []
for i in range(0, nrBS):
    sent.append(0)
for i in range(0,nrNodes*nrBS):
    sumSent = sumSent + nodes[i].sent
    print("id for node ", nodes[i].id, "BS:", nodes[i].bs.id, " sent: ", nodes[i].sent)
    sent[nodes[i].bs.id] = sent[nodes[i].bs.id] + nodes[i].sent
for i in range(0, nrBS):
    print("send to BS[",i,"]:", sent[i])

print("sumSent: ", sumSent)

der = []
# data extraction rate
derALL = len(recPackets)/float(sumSent)
sumder = 0
for i in range(0, nrBS):
    der.append(len(packetsRecBS[i])/float(sent[i]))
    print("DER BS[",i,"]:", der[i])
    sumder = sumder + der[i]
avgDER = (sumder)/nrBS
print("avg DER: ", avgDER)
print("DER with 1 network:", derALL)

# this can be done to keep graphics visible
if graphics:
    input('Press Enter to continue ...')

# save experiment data into a dat file that can be read by e.g. gnuplot
fname = "lorasim_{}node_{}basestation_{}network_results.dat".format(nrNodes, nrBs, nrNetworks)
print(fname)
if os.path.isfile(fname):
    res = "\n" + str(nrNodes) + " " + str(der[0]) 
else:
    res = "# nrNodes DER0 AVG-DER\n" + str(nrNodes) + " " + str(der[0]) + " " + str(avgDER) 
with open(fname, "a") as myfile:
    myfile.write(res)
myfile.close()

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
    captured = False
    # lost packets don't collide
    if packet.lost:
       return (0, captured)
    if packetsAtBS[packet.bs]:
        for other in packetsAtBS[packet.bs]:
            if other.id != packet.nodeid:
               # simple collision
               if frequencyCollision(packet, other.packet[packet.bs]) and sfCollision(packet, other.packet[packet.bs]):
                   # more complicated collisions
                   if timingCollision(packet, other.packet[packet.bs]):
                       # check who collides in the power domain
                       c = powerCollision(packet, other.packet[packet.bs])
                       # mark all the collided packets
                       # either this one, the other one, or both
                       for p in c:
                           p.collided = 1
                           if p == packet:
                               col = 1

                       # if the packet didn't collide, it only survived due to capture effect
                       if packet not in c:
                           captured = True

                   else:
                       # no timing collision, all fine
                       pass
        return (col, captured)
    return (0, captured)

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
    if p1.rssi < p2.rssi:
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
    else:
        return True

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
    def __init__(self, id, networkID):
        self.id = id
        self.networkID = networkID
        self.x = 0
        self.y = 0

        # This is a hack for now
        global nrBS
        global maxDist
        global baseDist

        # remove the "planned" setup for now and replace with random distribution
        if False:
            if self.id == 0 or self.id == 1:
                self.x = 0
                self.y = 0
            elif self.id == 2:
                self.x = maxDist/2
                self.y = 0
            elif self.id == 3:
                self.x = -maxDist/2
                self.y = 0
            elif self.id == 4:
                self.x = 0
                self.y = maxDist/2
            elif self.id == 5:
                self.x = 0
                self.y = -maxDist/2
            elif self.id == 6:
                self.x = maxDist/2*math.sin(math.pi/4)
                self.y = maxDist/2*math.cos(math.pi/4)
            elif self.id == 7:
                self.x = -maxDist/2*math.sin(math.pi/4)
                self.y = -maxDist/2*math.cos(math.pi/4)
            elif self.id == 8:
                self.x = -maxDist/2*math.sin(math.pi/4)
                self.y = maxDist/2*math.cos(math.pi/4)
            elif self.id == 9:
                self.x = maxDist/2*math.sin(math.pi/4)
                self.y = -maxDist/2*math.cos(math.pi/4)
            else:
                print("Too many base stations!!")
                sys.exit(1)

        # this is very complex prodecure for placing base stations
        # and ensure minimum distance between each pair of base stations
        found = 0
        rounds = 0
        global bs
        while (found == 0 and rounds < 100):
            a = random.random()
            b = random.random()
            if b<a:
                a,b = b,a
            posx = b*maxDist*math.cos(2*math.pi*a/b)
            posy = b*maxDist*math.sin(2*math.pi*a/b)
            if len(bs) > 0:
                for index, n in enumerate(bs):
                    dist = np.sqrt(((abs(n.x-posx))**2)+((abs(n.y-posy))**2)) 
                    if dist >= 0: 
                        found = 1
                        self.x = posx
                        self.y = posy
                    else:
                        rounds = rounds + 1
                        if rounds == 100:
                            print("could not place new base station, giving up")
                            exit(-2) 
            else:
                self.x = posx
                self.y = posy
                found = 1

        #print("BSx:", self.x, "BSy:", self.y)

        global graphics
        global pointsize
        if graphics:
            global ax
            if (self.networkID == 0):
                ax.add_artist(plt.Circle((self.x, self.y), pointsize*2, fill=True, color='cyan'))
            if (self.networkID == 1):
                ax.add_artist(plt.Circle((self.x, self.y), pointsize*2, fill=True, color='orange'))
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
    def __init__(self, id, period, dataRate, packetlen, networkID):

        self.id = id
        self.networkID = networkID
        self.x = 0
        self.y = 0
        self.period = period
        self.packet = []
        self.dist = []
        self.sent = 0

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
            posx = b*maxDist*math.cos(2*math.pi*a/b)
            posy = b*maxDist*math.sin(2*math.pi*a/b)
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
        global bs
        global nrBS
        for i in range(0,nrBS):
            d = np.sqrt((self.x-bs[i].x)*(self.x-bs[i].x)+(self.y-bs[i].y)*(self.y-bs[i].y)) 
            self.dist.append(d)
            self.packet.append(myPacket(self.id, dataRate, packetlen, self.dist[i], i))
        #print('node %d' %id, "x", self.x, "y", self.y, "dist: ", self.dist, "myNetwork:", self.networkID)

        # graphics for node
        global graphics
        global pointsize
        if graphics:
            global ax
            if self.networkID == 0:
                    ax.add_artist(plt.Circle((self.x, self.y), pointsize, fill=True, color='blue'))
            if self.networkID == 1:
                    ax.add_artist(plt.Circle((self.x, self.y), pointsize, fill=True, color='red'))
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

        # denote if packet is collided
        self.collided = 0
        self.captured = False

        # mark the packet as lost when it's rssi is below the sensitivity
        self.lost = self.rssi < rx_sensitivity
        #print("node {} bs {} lost {}".format(self.nodeid, self.bs, self.lost))
        if self.lost:
            print("*****Packet was totally lost. This shouldn't happen******")


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
        mySeqNo = packetSeq

        global nrBS
        for i in range(0, nrBS):
           if node in packetsAtBS[i]:
                print("ERROR: packet already in")
           else:
                # adding packet if no collision
                (collided, captured) = checkcollision(node.packet[i])
                if collided == 1:
                    node.packet[i].collided = 1
                    node.packet[i].captured = False # can't have survived due to capture effect if it didn't survive...
                else:
                    node.packet[i].collided = 0
                    node.packet[i].captured = captured
                packetsAtBS[i].append(node)
                node.packet[i].addTime = env.now
                node.packet[i].seqNr = packetSeq
 
        # take first packet rectime        
        yield env.timeout(node.packet[0].rectime)

        # if packet did not collide, add it in list of received packets
        # unless it is already in
        global bs
        only_captured = True
        received = False
        for i in range(0, nrBS):
            if node.packet[i].lost:
                lostPackets.append(node.packet[i].seqNr)
            else:
                if node.packet[i].collided == 0:
                    if nrNetworks == 1:
                        received = True
                        packetsRecBS[i].append(node.packet[i].seqNr)
                    else:
                        # now need to check for right network
                        if node.networkID == bs[i].networkID:
                            received = True
                            packetsRecBS[i].append(node.packet[i].seqNr)
                            if not node.packet[i].captured:
                                only_captured = False
                    # recPackets is a global list of received packets
                    # not updated for multiple networks        
                    if recPackets:
                        if recPackets[-1] != node.packet[i].seqNr:
                            recPackets.append(node.packet[i].seqNr)
                    else:
                        recPackets.append(node.packet[i].seqNr)
                else:
                    # XXX only for debugging
                    collidedPackets.append(node.packet[i].seqNr)

        if received and only_captured:
            packetsRecNetworkCaptureEffect[node.networkID].append(mySeqNo)

        # complete packet has been received by base station
        # can remove it

        for i in range(0, nrBS):                    
            if node in packetsAtBS[i]:
                packetsAtBS[i].remove(node)
                # reset the packet
                node.packet[i].collided = 0


#
# "main" program
#

# Local Configurations
maxNodes = 2000         # maximum number of nodes to increment to
maxBS = 20              # maximum number of base stations to increment to
repeatCount = 10        # number of times to try each base station amount
avgSendTime = 60*1000   # once per minute
simtime = 24*60*60*1000 # one day
packetLen = 20          # 20 byte packets

# global configurations
graphics = False
print("AvgSendTime (exp. distributed):",avgSendTime)
print("Simtime: ", simtime)

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
tx_power = 20

# uses Hata Model for maximum distance
maxDist = rangeEstimate(tx_power, rx_sensitivity)/2
print("Max distance: {} meters".format(maxDist))
print("")

# keep a dict of all results
results = {}

# number of nodes each network contains
for nodeCount in range(100, maxNodes+100, 100):
    results[nodeCount] = {}

    # total across all networks, 1 for first network, N-1 for other network
    for baseStationCount in range(2,maxBS+1+1):
        results[nodeCount][baseStationCount] = []

        for repetition in range(repeatCount):
            results[nodeCount][baseStationCount].append({})

            # configurations
            nrNodes = nodeCount
            nrBS = baseStationCount
            nrNetworks = 2 # 1 or 2 networks
            print("Run configuration:")
            print("\tNodes per base station:", nrNodes) 
            print("\tnrBS: ", nrBS)
            print("\tnrNetworks: ", nrNetworks)

            # global stuff
            nodes = []
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
                ax.add_artist(plt.Circle((0, 0), maxDist, fill=False, color='black'))

            # list of base stations
            bs = []

            # list of packets at each base station, init with 0 packets
            packetsAtBS = []
            packetsRecBS = []
            for i in range(0,nrBS):
                b = None

                # Put the first base station on its own network, and the rest on network 2
                if nrNetworks == 2 and i != 0:
                    b = myBS(i, 1)
                else:
                    b = myBS(i, 0)

                bs.append(b)
                packetsAtBS.append([])
                packetsRecBS.append([])

            # keep track of packets only received due to capture effect
            packetsRecNetworkCaptureEffect = []
            for i in range(0, nrNetworks):
                packetsRecNetworkCaptureEffect.append([])

            # create N nodes on irrelevant network, 100 on multi-gateway network
            for i in range(nrNodes):
                node = myNode(i, avgSendTime, dataRate, packetLen, 0)
                nodes.append(node)
                env.process(transmit(env,node))
            for i in range(100):
                node = myNode(nrNodes+i, avgSendTime, dataRate, packetLen, 1)
                nodes.append(node)
                env.process(transmit(env,node))

            ##XXX: remove symmetric node creation for now
            #for i in range(0, nrNodes):
            #    for j in range(0, nrNetworks):
            #        # create nrNodes for each base station
            #        node = myNode(i*nrNetworks+j, avgSendTime, dataRate, packetLen, j)
            #        nodes.append(node)
            #        env.process(transmit(env,node))

            #prepare show
            if graphics:
                plt.xlim([-1.1*maxDist, 1.1*maxDist])
                plt.ylim([-1.1*maxDist, 1.1*maxDist])
                plt.draw()
                plt.show()  

            # start simulation
            env.run(until=simtime)

            # determine total number of unique packets received by the network
            network_transmissions = [0]*nrNetworks
            network_receptions = [0]*nrNetworks
            for i in range(0, nrNetworks):
                for j in range(len(nodes)):
                    if nodes[j].networkID == i:
                        network_transmissions[i] += nodes[j].sent

                all_received_packets = []
                all_captured_packets = []
                for j in range(0, nrBS):
                    if bs[j].networkID == i:
                        all_received_packets += packetsRecBS[j]
                network_receptions[i] = len(set(all_received_packets))

            # general stats on performance
            print("General stats")
            print("\ttransmitted packets", sum(network_transmissions))
            if packetSeq != sum(network_transmissions):
                print("Transmission count failure. {} != {}".format(packetSeq, sum(network_transmissions)))
                sys.exit(1)
            print("\treceived packets", len(recPackets))
            print("\treceived packets (deduplicated) {}".format(sum(network_receptions)))
            print("\tcollided packets", len(collidedPackets))
            print("\tlost packets (should be zero)", len(lostPackets))

            # network-specific statistics
            for i in range(0, nrNetworks):
                print("Network stats {}".format(i))
                print("\tnumber of base stations {}".format(len([station for station in bs if station.networkID == i])))
                print("\tnumber of nodes {}".format(len([node for node in nodes if node.networkID == i])))
                print("\ttransmitted packets {}".format(network_transmissions[i]))
                print("\treceived packets {} (deduplicated)".format(network_receptions[i]))
                print("\treceived packets due to capture effect {}".format(len(packetsRecNetworkCaptureEffect[i])))
                print("\tpacket reception rate {}".format(network_receptions[i]/network_transmissions[i]))

                # save results
                results[nodeCount][baseStationCount][repetition][i] = network_receptions[i]/network_transmissions[i]

            # do not need to record node or basestation location
            if False:
                # store nodes and basestation locations to file
                with open('nodes.txt', 'w') as nfile:
                    nfile.write("#NodeID NetworkID NodeX NodeY\n")
                    for node in nodes:
                        nfile.write('{} {} {} {}\n'.format(node.id, node.networkID, node.x, node.y))

                with open('basestation.txt', 'w') as bfile:
                    bfile.write("#BSID NetworkID BSX BSI\n")
                    for basestation in bs:
                        bfile.write('{} {} {} {}\n'.format(basestation.id, basestation.networkID, basestation.x, basestation.y))

            if graphics:
                #input('Testing...')
                plt.pause(1)
            print("")


# record results to file
with open('results_{}nodes_{}basestations_{}repeats_100setup.dict'.format(maxNodes, maxBS, repeatCount), 'w') as outfile:
    outfile.write('{}\n'.format(results))


# keep graphics visible
if graphics:
    input('Press Enter to continue ...')


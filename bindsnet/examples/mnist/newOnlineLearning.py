# This file used : https://bindsnet-docs.readthedocs.io/guide/guide_part_i.html#adding-network-components as a starting point

#NOTE this is the version from NOV 1st

import sys
import time

sys.path.insert(0, '/home/lj/Documents/prefetch_project/bindsnet/')
import os
import torch

from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet import encoding

from itertools import islice
import os
import numpy as np
import csv


def computeDeltas(offsetlist):
    if len(offsetlist) < 2:
        print("error list too short")
    result = []

    for i in range(1, len(offsetlist)):
        result.append(offsetlist[i] - offsetlist[i - 1])
    return result


def minPrediction(listof):
    currentmin = listof[0].confidence
    mintuple = listof[0]
    index = -1

    for i in range(0, len(listof)):
        if currentmin > listof[i].confidence:
            currentmin = listof[i].confidence
            mintuple = listof[i]
            index = i

    if index == -1:
        print("error min tuple index")
    return (mintuple, index)


class createNetwork:

    def __init__(self, patternLength, confidenceThreshold, minConfidence):

        # mappings for
        self.offset_neuronlabel_dict = {}  # map neuron to (offset_delta, confidence)
        # offset_neuronlabel_dict[int(str(pc) + str(page))][neuron] = (offset_delta, 1)

        self.neuronmap = dict()
        self.confidenceThreshold = confidenceThreshold
        self.minConfidence = minConfidence
        self.patternLength = patternLength
        self.network = DiehlAndCook2015(
            n_inpt=128 * patternLength,
            n_neurons=50,
            exc=22.5,
            inh=17.5,
            dt=1,
            norm=38.4,
            theta_plus=.05,
            inpt_shape=(1, 128 * patternLength),
        )

        # Simulation time.
        time = 100
        dt = 1
        device = 'cpu'

        # set up the spike monitors
        self.spikes = {}
        for layer in set(self.network.layers):
            self.spikes[layer] = Monitor(
                self.network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
            )
            self.network.add_monitor(self.spikes[layer], name="%s_spikes" % layer)

        self.output_monitor = self.spikes['Ae']
        self.table = {}

    # inputs into the network the values
    # from the linked list and then
    # returns the predicted result
    def inputIntoNetwork(self, inputs_arr):

        current = torch.full([128 * self.patternLength], 0)
        for i in range(0, self.patternLength):
            # why to adjust this?
            current[(i * 128) + (inputs_arr[i] + 64)] = 512

        # generate the spiking inputs for the network
        input_data_new = encoding.encodings.poisson(current, 100, 1, device='cpu')
        test_inputs = {"X": input_data_new}

        # pass the spike trains as input into the network
        self.network.run(inputs=test_inputs, time=100)

        # array for keeping track of firing neurons
        spikecount = np.zeros(50)
        outputNeurons = {}

        for time in range(0, 100):
            for i in range(0, 50):
                if self.output_monitor.get('s')[time][0][i]:
                    # print("neuron fire", i)
                    spikecount[i] += 1
                    outputNeurons[i] = 1

        # resets the spike monitors
        self.network.reset_state_variables()  # Reset state variables.
        return outputNeurons

    # makes a given number of predictions on the given inputs
    # returns a prediction list (prediction, confidence), and a list of unlabeled Neurons
    #   def init(self, delta, offset, neuron, pc, page, usage, confidence):
    def makePrediction(self, inputs_arr, numPredictions, pc, page, offset):
        # inputs into the network to get the output neurons
        offset_outputNeurons = self.inputIntoNetwork(
            inputs_arr)  # hotNetworksOffset[pc].inputIntoNetwork(offset_delta_arr).keys()

        predictions = []
        unlabeledNeurons = []
        if offset_outputNeurons:
            for neuron in offset_outputNeurons:
                if neuron in self.offset_neuronlabel_dict:

                    offset_delta_prediction, confidence = self.offset_neuronlabel_dict.get(neuron)

                    # if the confidence is large enough
                    if confidence >= confidenceThreshold:

                        if len(predictions) < numPredictions:

                            #predictions.append((offset_delta_prediction, confidence, neuron))
                             temp = Prediction(offset_delta_prediction, offset, neuron, pc, page, 0, confidence)
                             predictions.append(temp)

                        else:
                            replace, index = minPrediction(predictions, 1)
                            if replace.confidence < confidence:
                                predictions[index] = Prediction(offset_delta_prediction, offset, neuron, pc, page, 0, confidence)
                                #predictions[index] = (offset_delta_prediction, confidence, neuron)


                else:
                    # record the neurons are not in the neuron label dict, then will label them in next iteration with pc/page
                    #unlabeledNeurons.append((neuron, pcpage, offset))
                    unlabeledNeurons.append(Prediction(0, offset, neuron, pc, page, 0, 0))
                    # to assign a label to a neuron, that can be done outside of the network
                    # alongside of decreasing the confidence

        return (predictions, unlabeledNeurons)

    def assignLabel(self, neuron, label, confidence):

        # TODO if confidence goes below minimum, then remove it
        print(confidence)
        print(minConfidence)
        print("------")
        if confidence < minConfidence:
            del self.offset_neuronlabel_dict[neuron]
            return #TODO this is new
            # because it used to remove it, then re add it
        self.offset_neuronlabel_dict[neuron] = (label, confidence)

    def getLabel(self, neuron):
        return self.offset_neuronlabel_dict.get(neuron)

    # trying to remove from an empty queue for some reason starts an infinite loop(?)


# ... so don't do that


#NOTE change to a ranked queue in the future (ranked on usage)
class PredictionBuffer:
    def __init__(self, maxSize):
        self.maxSize = maxSize
        self.buffer = []  # prediction, usage

    # adds the prediction to the buffer
    # if the buffer is too big it returns
    # the prediction that was pushed out
    # otherwise returns none
    def addPrediction(self, prediction):
        #TODO this will probably need to change
        self.buffer.append(prediction)
        if len(self.buffer) >= self.maxSize:
            return self.buffer.pop(0)
        return None

    # this checks to see if the request is in the buffer
    # if it is, then return true and increase the usage
    # amount for that entry in the buffer
    def getPrediction(self, request):

        for i in range(0, len(self.buffer)):
            if request == self.buffer[i][0]:
                self.buffer[i] = (self.buffer[i][0], self.buffer[i][1] + 1)
                return i
        return -1

class Prediction:
   def __init__(self, delta, offset, neuron, pc, page, usage, confidence):
      self.delta = delta
      self.offset = offset
      self.neuron = neuron
      self.pc = pc
      self.page = page
      self.usage = usage
      self.confidence = confidence


#takes in a pcpage string, offset, hotpcpage dictionary, and list of unlabeledNeurons
#returns the remaining list of neurons after creating maps for the unlabeled Neurons
#assigns the labels to the hotpcpage keys for the current pcpage
def labelNeurons(pcpage, offset, hotpcpage, unlabeledNeurons):
    if pcpage in hotpcpage:
        prevNetwork = hotpcpage[pcpage]
        removeIndicies = []

        # modify labels
        for i in range(0, len(unlabeledNeurons)):
            unlabeledPrediction = unlabeledNeurons[i]
            if unlabeledPrediction.pc + unlabeledPrediction.page == pcpage: 
                removeIndicies.append(i)
                prevNetwork.assignLabel(unlabeledPrediction.neuron, offset - unlabeledPrediction.offset, 1)

        #TODO fix this as it will be slow
        temp = []
        for i in range(0, len(unlabeledNeurons)):
            if i not in removeIndicies:
               temp.append(unlabeledNeurons[i]) 
        unlabeledNeurons = temp
        return unlabeledNeurons
    return unlabeledNeurons

#checks to see if there was a hit in the prefetch buffer
def checkHit(predictionBuffer, page, offset, hotpcpage):

   hit = 0
   for i in range(0, len(predictionBuffer.buffer)):
      current = predictionBuffer.buffer[i]
      if current.offset + current.delta == offset and current.page == str(page):
         hit = 1
         current.usage += 1
         currentNetwork = hotpcpage[current.pc + current.page]
         label = currentNetwork.getLabel(current.neuron)
         if label != None:
            currentNetwork.assignLabel(current.neuron, label[0], label[1] + 1)
         else:
            print("error label")
   return hit


# given the removed prediction, decrease its
# confidence in its network based on the usage 
def decreaseConfidence(removedPrediction, hotpcpage):
   if removedPrediction.usage == 0:

      predictionNetwork = hotpcpage[removedPrediction.pc + removedPrediction.page]

      # check to see what the label is
      label = predictionNetwork.getLabel(removedPrediction.neuron)
      if label != None:
         neuron = removedPrediction.neuron
         confidence = label[1] 
         predictionNetwork.assignLabel(neuron, label[0], confidence - 1)
      return 0
   return 1


# creates a prediction buffer with size 100
predictionBuffer = PredictionBuffer(100)

# make a dictionary to keep track of hot pcpages
# to store the networks
hotpcpage = dict()

# pattern storage
pcpagepattern = dict()
# key: pcpage (string)
# value: list of offsets

# constants:
patternlength = 3
numberOfPredictions = 2
minConfidence = -1
confidenceThreshold = 1

unlabeledNeurons = []
prevPcPage = ""

start = 0
end = 100
index = 0

correctPredictions = 0
totalPredictions = 0

print("hello world")
# this just tests the code and doesn't do anything else
usetest = input("just run tests? y/n")


if usetest == "y":

   #test 1:
   #tests that basic labeling works 
   if 1 > 0:
      predictionBuffer = PredictionBuffer(100)
      hotpcpage = dict()
      pcpage = "101"
      hotpcpage[pcpage] = createNetwork(patternlength, confidenceThreshold,minConfidence)
      currentNetwork = hotpcpage["101"]
      predictions = []
      newunlabeledNeurons = []
      index = 0

      while len(newunlabeledNeurons) < 1 or index < 100: 
         print(index)
         index += 1
         predictions, newunlabeledNeurons = currentNetwork.makePrediction([0, 0, 0, 0], numberOfPredictions, "1", "01", 10)


      stored = newunlabeledNeurons[0].neuron
      labelNeurons("101", 10, hotpcpage, [newunlabeledNeurons[0]])
      predictions, newunlabeledNeurons = currentNetwork.makePrediction([0, 0, 0, 0], numberOfPredictions, "1", "01", 10)
      
      if len(predictions) != 1:
         print("test 1 error labeling")

      if predictions[0].neuron != stored:
         print("test 1 error with neuron") 

      for i in newunlabeledNeurons:
         if i.neuron == stored:
            print("test 1 error unlabeledneuron")

      if stored not in currentNetwork.offset_neuronlabel_dict:
         print("test 1 not labeled")


      print(currentNetwork.offset_neuronlabel_dict[stored])

      #test 2 making sure that confidence decreasing actually removes stuff 
      for i in range(0, 100):
         decreaseConfidence(Prediction(10, 10, stored, "1", "01", 0, 1), hotpcpage)

      if stored in currentNetwork.offset_neuronlabel_dict:
         print("test 1 still labeled")
         print(currentNetwork.offset_neuronlabel_dict[stored])

      predictions, newunlabeledNeurons = currentNetwork.makePrediction([0, 0, 0, 0], numberOfPredictions, "1", "01", 10)
     
      #test 3 reassigning a label
      labelNeurons("101", 11, hotpcpage, [newunlabeledNeurons[0]])
      label, confidence = currentNetwork.offset_neuronlabel_dict[stored]
      if label != 1:
         print("wrong label " + str(label))
      if confidence != 1:
         print("wrong confidence " + str(confidence))


   #making sure that the prediction buffer actually works 
   #makes sure that basic prediction adding works
   if 2 > 0:
      predictionBuffer = PredictionBuffer(10)
      test = Prediction(10, 50, 0, "pc", "page", 0, 1)
      test2 = Prediction(20, 50, 0, "pc", "page", 0, 1)

      result = predictionBuffer.addPrediction(test)
      if len(predictionBuffer.buffer) != 1 or result != None:
         print("wrong prediction buffer length")
      
      result = predictionBuffer.addPrediction(test)
      if len(predictionBuffer.buffer) != 2 or result != None:
         print("wrong prediction buffer length")
      for i in range(0, 8):
        result = predictionBuffer.addPrediction(test2)
      
      if result.delta != 10:
         print("wrong prediction removed 1")

      result = predictionBuffer.addPrediction(test)
      if result.delta != 10:
         print("wrong prediction removed 2")
      result = predictionBuffer.addPrediction(test)
      if result.delta != 20:
         print("wrong prediction removed 3")
       

   #checking to see if the usage counter actually increments
   #checks prediction buffer, check hit, addPrediction
   if 3 > 0:
      predictionBuffer = PredictionBuffer(2)
      pcpage = "1page"
      currentnetwork = createNetwork(patternlength, confidenceThreshold,minConfidence)
      hotpcpage[pcpage] = currentnetwork 
      currentnetwork.assignLabel(0, 10, 1)
      currentnetwork.assignLabel(1, 20, 1)

      #check to make sure that there isn't a hit
      if checkHit(predictionBuffer, "page", 10, hotpcpage) != 0:
         print("hit with empty buffer")

      test = Prediction(10, 50, 0, "1", "page", 0, 1)
      test2 = Prediction(20, 50, 1, "1", "page", 0, 1)

      #now try adding something to the buffer
      result = predictionBuffer.addPrediction(test)
      if result != None:
         print("result from add prediction on empty buffer") 

      if checkHit(predictionBuffer, "page", 60, hotpcpage) == 0:
         print("unexpected miss")

      #add the second item
      result = predictionBuffer.addPrediction(test2)
      if result == None:
         print("result is empty") 
      if result.delta != 10:
         print("wrong prediction pushed out")
      if result.usage != 1:
         print("wrong usage amount")

      result = predictionBuffer.addPrediction(test2)
      if result == None:
         print("result is empty 2") 
      if result.delta != 20:
         print("wrong prediction pushed out 2")
      if result.usage != 0:
         print("wrong usage amount 2")

      if checkHit(predictionBuffer, "page", 70, hotpcpage) == 0:
         print("unexpected miss")
      if checkHit(predictionBuffer, "page", 70, hotpcpage) == 0:
         print("unexpected miss")

      result = predictionBuffer.addPrediction(test2)
      if result == None:
         print("result is empty 3") 
      if result.delta != 20:
         print("wrong prediction pushed out 3")
      if result.usage != 2:
         print("wrong usage amount 3")


   if 4 > 0:
      print("test 4")








input("done with tests")
file = open(sys.argv[1], mode='r')

for line in file:
    print("index:", index)

    # TODO this is lazy--------------------
    if index < start:
        index += 1
        continue

    if index > end:
        break

    index += 1
    elements = line.split(",")
    pc, page, offset = int(elements[3], 16), int(elements[2], 16) >> 12, ((int(elements[2], 16) >> 6) & 0x3f)
    print("current offset", offset)

   
    # call labelNeurons function here
    unlabeledNeurons = labelNeurons(str(pc) + str(page), offset, hotpcpage, unlabeledNeurons)

   
    # check prefetch buffer for hit
    if checkHit(predictionBuffer, page, offset, hotpcpage):
        correctPredictions += 1
        print("correct Prediction ")

    if (str(pc) + str(page)) in pcpagepattern:

        pcpagepattern[str(pc) + str(page)].append(offset)
        offsetlist = pcpagepattern[str(pc) + str(page)]

        if len(offsetlist) > patternlength:
            deltalist = computeDeltas(offsetlist[-4:])

            if str(pc) + str(page) in hotpcpage:
                currentNetwork = hotpcpage[str(pc) + str(page)]
                predictions, newunlabeledNeurons = currentNetwork.makePrediction(deltalist, numberOfPredictions, str(pc), str(page), offset)
                unlabeledNeurons += newunlabeledNeurons


                #-------------------------------------------------- 
                # need predictions, hotpcpage
                # add prediction to prefetch buffer
                for prediction in predictions:
                    totalPredictions += 1 
                    tempresult = predictionBuffer.addPrediction(prediction)
                    if tempresult != None:
                        decreaseConfidence(tempresult, hotpcpage)                    
                                       #-------------------------------------------------- 


            else:
                hotpcpage[str(pc) + str(page)] = createNetwork(patternlength, confidenceThreshold,minConfidence)
                currentNetwork = hotpcpage[str(pc) + str(page)]
                predictions, unlabeledNeurons = currentNetwork.makePrediction(deltalist, numberOfPredictions, str(pc), str(page), offset)
    else:
        pcpagepattern[str(pc) + str(page)] = []
        pcpagepattern[str(pc) + str(page)].append(offset)




    #print("prediction buffer: ", len(predictionBuffer.buffer))

    #print("iteration end__________________________________________________")







print("correct predictions")
print(correctPredictions)

print("total predictions")
print(totalPredictions)

print("average predictions")
print(totalPredictions / (end - start))


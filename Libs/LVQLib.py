
# coding: utf-8

# In[1]:

import os, csv
from random import shuffle


# In[2]:

class LVQData:
    def __init__(self):
        """
        LVQData Constructor:
            LVQData is a list of input vectors paired with their correct 
            instrument classification.
            LVQNet will iterate over this structure until it converges.
        """
        
        self.data          = [] # list of tuples 
        self.instrumentMap = {} # map integers with instruments (labels)
        self.instrumentNum = 0  # current integer instrument (to neuron)
        
    def shuffleData(self, n):
        """
        Shuffle the training data n times (in place)
            - n must be a valid integer
        """
        
        for i in range(n):
            shuffle(self.data)
        return self.data
    
    def getVectorData(self, index):
        """
        Get the STFT of the Frame (1025 bins) at the specified index
            index must be an valid integer
        """
        
        return self.data[index][0]
    
    def getVectorLabel(self, index):
        """
        Get the label of the Frame at the specified index
            - index must be an valid integer
        """
        
        return self.data[index][1]
    
    def getVector(self, index):
        """
        Get the STFT data and label of the Frame at the specified index
            - index must be an valid integer
        """
        
        return self.data[index][0], self.data[index][1]
    
    def lookupInstrument(self, index):
        """
        Lookup the instument specified by the number of the neuron
            - index must be a valid integer
        """
        return self.instrumentMap[index]

    def inverseLookupInstrument(self, name):
        """
        Lookup the instrument by name. The neuron number will be returned
            - name must be a valid string for one of the instruments
            
        (TODO) Use the third method here for small speedups later on.
        http://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
        """
        
        for key, instrument in self.instrumentMap.items():
            if (instrument == name):
                return key
        
    def loadCSV(self, filepath, label):
        """
        Add a STFT datapoint and label to the LVQData list
            - filepath must be valid string from current working directory 
            (os.getcwd() to check)
            - label must be a string describing the instrument of the csv 
            datapoints for entire CSV
        """
        
        with open(filepath, 'r') as f:
            read = csv.reader(f, delimiter=',')
            read.next()
            
            for row in read:
                data_struct = (arrayParser(row[1]), label)
                self.data.append(data_struct)

        if not(label in self.instrumentMap.values()):
            self.instrumentMap[self.instrumentNum] = label
            self.instrumentNum += 1
        else:
            print "%s is already in instrument map." % (label)
            
        return self.data


# In[3]:

class LVQNeuron:
    def __init__(self, name):
        self.name = name
        self.weights = []
    
    def setWeights(self, weights):
        """
        Import a vector of information (the first entry) for 
        initializing next unique neuron.
            - weights is a string from the import of a CSV
        """
        
        weights = weights.replace('[', '').replace(']', '').split(',')
        
        for weight in weights:
            self.weights.append(float(weight))
        
    def __len__(self):
        return len(self.weights)


# In[4]:

class LVQNet:
    def __init__(self, inCount, outCount):
        self.inputs   = inCount
        self.outputs  = outCount
        self.alpha    = 0.005
        self.csvCount = 0  # Limit the CSVs input to number of output neurons
        self.iter     = 0  # Number of times the vectors have been used in training
        self.neurons  = {} # Numbered index map (to outputs)
        
        for n in range(outCount):
            curr_neuron = LVQNeuron(n)
            self.neurons[n] = curr_neuron
            
    def __len__(self):
        return len(self.neurons)
    
    def getWeights(self, neuronNo):
        """
        Get the current weights of the neuron specified
            - neuronNo must be a valid integer
        """
        
        return self.neurons[neuronNo].weights
    
    def setWeights(self, neuronNo, newWeights):
        """
        Change the neuron weights with new Weight vector
            - neuronNo must be a valid integer
            - newWeights must be a list of numbers the same length 
            as the previous weights.
        """
        
        prev = self.neurons[neuronNo]
        
        if len(prev) == len(newWeights):
            prev.weights = newWeights
    
    def saveLVQNet(self, filepath):
        """
        Incomplete! Do not use!
        """
        with open(filepath, 'wb') as f:
            cursor = csv.writer(f, delimiter=',')
            cursor.writerow([self.inputs, self.outputs, self.alpha, 
                             self.csvCount, self.iter])
            
            for key, val in self.neurons.items():
                cursor.writerow([key, val])
        
    # STEP 0
    def enterCSV(self, filepath): 
        """
        Initialize neurons with first row of data in CSV 
        (each CSV should represent a neuron).
            - filepath must be valid string from current working
            directory (os.getcwd() to check)
        """
        
        if self.csvCount >= self.outputs:
            print "Reached limit of neurons"
            return
            
        with open(filepath, 'r') as f:
            read = csv.reader(f, delimiter=',')
            row = read.next()
            curr_neuron = self.neurons[self.csvCount]
            curr_neuron.setWeights(row[1])
            self.csvCount += 1
            
            print "Successfully added neuron from CSV", filepath
            return

    # STEP 3.1
    def edist(self, inputs, weights):
        """
        Helper function to get the Euclidean distance between input 
        vector and neuron weights.
            - inputs must be list of numbers with same number 
            of items as weights
            - weights must be a valid list of neuron weights 
        """
        
        euclideanDistance = 0
        
        if len(inputs) != len(weights):
            print len(inputs), "is different length than", len(weights)
            return
        
        for i in range(len(inputs)):
            nth = inputs[i] - weights[i]
            nth = nth ** 2
            euclideanDistance += nth
             
        return euclideanDistance ** (0.5)
    
    # STEP 3.2 
    def minDist(self, inputVector):
        """
        Given an input vector, calculate the closest (guess) neuron to classify
            - inputVector must be list of numbers with the same number of items 
            as the neuron weights
        """
        
        scores = [] # Euclidean Distances
        
        for neuron in self.neurons:
            wunit = self.neurons[neuron].weights
            scores.append(self.edist(inputVector, wunit))
        
        minNeuronIndex = scores.index(min(scores))
        return minNeuronIndex
    
    def minDistGradient(self, inputVector):
        """
        Given an input vector, calculate the list of neurons in order
            - inputVector must be a list of numbers with the same number of 
            items as the neuron weights
        """
        
        scores = {} # Euclidean Distances
        iterator = 0
        
        for neuron in self.neurons:
            wunit = self.neurons[neuron].weights
            scores[iterator] = self.edist(inputVector, wunit)
            iterator += 1

        return scores
    
    # STEP 4.1
    def calibrate(self, neuronNo, guessNo):
        """
        Given a neuron number and a guess neuron number, 
        first check to see if the guess was correct 
        or not and then calibrate the neurons accordingly 
        (increase if correct, decrease if incorrect).
            - neuronNo must be a valid integer 
            (between 0 and total number of neurons - 1)
            - guessNo  must be a valid integer 
            (between 0 and total number of neurons - 1)
        """
        
        neuron      = dataset.getVectorLabel(neuronNo) 
        inputVector = dataset.getVectorData(neuronNo)
        guess       = dataset.lookupInstrument(guessNo) 
        weights     = self.getWeights(guessNo)     
        
        addfunc = lambda oldWeight, vec: oldWeight + self.alpha * (vec - oldWeight)
        subfunc = lambda oldWeight, vec: oldWeight - self.alpha * (vec - oldWeight)

        if neuron == guess: 
            newWeights = map(addfunc, weights, inputVector) 
            self.setWeights(guessNo, newWeights)
            return newWeights
        else:
            newWeights = map(subfunc, weights, inputVector) 
            self.setWeights(guessNo, newWeights)
            return newWeights

    # STEP 5
    def reduceAlpha(self, beta):
        """
        Helper function to reduce alpha value
            - beta must be valid float between 1 and 100 (percent decrease)
        """
        
        self.alpha = self.alpha * (beta / 100.00)
        return self.alpha
    
    def iteration(self, dataset): # Helper function for iterating through during run function
        """
        Given a dataset, iteration goes through each vector and calibrates the neurons 
        using the calibrate function
            - dataset must be a list of tuples including vectors with correct dimensions 
            and classification labels
        """
        
        length = range(len(dataset.data))
        for v in length:
            # v is the dataset.data index while vector is the input vector
            vector = dataset.data[v]
            guess = self.minDist(vector[0])
            newWeights = self.calibrate(v, guess)
            
    
    def guess(self, stft):
        return self.minDist(stft)
            
    @staticmethod        
    def meanSquaredError(oldList, newList):
        """
        MeanSquaredError will return a averages from the difference 
        between the old neuron weight and new neuron weight squared.
            - oldList must be a list of weights that has the same 
            dimensions as newList
            - newList must be a list of weights that has the same 
            dimensions as the neuron weights
        """
        
        diffSquared = lambda a, b : (a - b) ** 2
        average     = lambda arr  :  float(sum(arr) / len(arr))
        
        diffs = map(diffSquared, oldList, newList)
        return average(diffs)
        
    def run(self, ds):
        """
        Run the algorithm given a LVQData object that has been 
        initialized with all the CSVs.
            - ds must have same number of CSVs initialized as neurons training 
            (or classifications)
        """
        
        if (len(ds.data) == 0):
            print "Need to initialize data in order to run the neural net."
            return
        else:
            self.iter += 1
            print "Running the algorithm with %d vectors. Iteration #%d." % (len(ds.data), self.iter)
        
        oldW = [] # Debug Before 
        for neuron in range(len(self.neurons)):
            w = self.getWeights(neuron)
            oldW.append(w)
            
        self.iteration(ds)

        newW = [] # Debug After
        for neuron in range(len(self.neurons)):
            w = self.getWeights(neuron)
            newW.append(w)
        
        means   = []
        average = lambda arr  :  float(sum(arr) / len(arr))
        for n in range(len(oldW)):
            means.append(LVQNet.meanSquaredError(oldW[n], newW[n]))
            
        return average(means)


# In[5]:

def arrayParser(arr):
    """
    ArrayParser takes an array created from reading a CSV into memory and 
    turns the lists-strings and casts them to a valid python list
        - arr must be a string that has the same format as a python list
    """
    # CSV usage: cast string list to python list
    smooth_stage_1 = arr.replace('[', '').replace(']', '').split(',')
    smooth_stage_2 = map(lambda unit: float(unit), smooth_stage_1)
    return smooth_stage_2


# In[6]:

def test(network, testdata, dataset):
    """
    General purpose testing (guess and check) for supervised dataset.
        - network must be a trained LVQ Network object
        - testdata must be a LVQ Data object
        - dataset must be a LVQ Data object (the one used to train the network)
    """
    numWrong = 0
    frameNum = 0
    frameLen = len(testdata.data)
    
    print "%d samples in testset. Commencing trials:" % (frameLen)
    print "Classifications:", dataset.instrumentMap
    
    for frame in testset.data:
        frameNum     += 1
        guess         = koho.guess(frame[0])
        guessGradient = koho.minDistGradient(frame[0])
        correctNum    = dataset.inverseLookupInstrument(frame[1])
        correctName   = frame[1]
        
        # Debug testing the neurons
        if (dataset.lookupInstrument(guess) != frame[1]):
            print "\n%d) Incorrect Guess : %s (%d)"  % (frameNum, dataset.lookupInstrument(guess), guess),
            print "[correct answer is: %s (%d)]" % (correctName, correctNum)
            print [ "%d: %.2f" % (key, guessGradient[key]) for key in guessGradient ]
            numWrong += 1
        else:
            print "Correct Guess: %s (%d)" % (dataset.lookupInstrument(guess), guess),
            print [ "%d: %.2f" % (key, guessGradient[key]) for key in guessGradient ]
            
    print "\nIncorrect %d / %d times" % (numWrong, frameLen)
    return True


# In[33]:

### Driver: Outline of the API / Algorithm in use    
if __name__ == '__main__':
    # Create Network with in and out neuron parameters
    koho = LVQNet(1025, 2)
    
    # Enter data (1-1 CSV to Output Neurons) 
    # Initializes the neurons with first onset from each unique CSV
    koho.enterCSV('./Data/Snare/snareFrames.csv')
    koho.enterCSV('./Data/Kick/kickDrumFrames4.csv')
    
    # Instantiate LVQ Training Data Structure and load rest of CSVs with labels
    dataset = LVQData()
    
    dataset.loadCSV('./Data/Snare/snareFrames.csv',    'snare-drum') # 0
    dataset.loadCSV('./Data/Snare/snareFrames3.csv',   'snare-drum') # 0
    dataset.loadCSV('./Data/Kick/kickDrumFrames4.csv', 'kick-drum')  # 1
    dataset.loadCSV('./Data/Kick/kickDrumFrames.csv',  'kick-drum')  # 1
    
    sigma = 1 # Sigma tracks the meanSquaredError between each calibration iteration in the LVQ training
    while (koho.iter < 500):
        dataset.shuffleData(3)
        sigma  = koho.run(dataset)
        alphie = koho.reduceAlpha(99)
        print sigma


# In[34]:

if __name__ == '__main__':
    testset = LVQData()
    
    testset.loadCSV('./Data/Test-Onsets/kickDrumTestFrames.csv', 'kick-drum')
    testset.loadCSV('./Data/Test-Onsets/snareTestFrames.csv',    'snare-drum')
    testset.loadCSV('./Data/Snare/snareFrames2.csv',             'snare-drum')

    test(koho, testset, dataset)


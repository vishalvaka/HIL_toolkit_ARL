import numpy as np
from numpy import unravel_index
import pandas as pd
import math
import neurokit2 as nk



# Function to find the pair of symbols with maximum frequency of occurrence

def FindPair(SymSeq):
    Pair = []
    Alphabet = np.unique(SymSeq)
    M = max(Alphabet)
    L = len(SymSeq)

    # This is an M x M 2D array which maps the pairs of possible symbols to 2D array indices. 
    # Eg: the pair '12' will correspond to the array indices (1, 2)-> (row, column)
    Count_Array = np.zeros((M,M))

    indx=0
    # indx variable starts from 1st element, shifts one symbol to the right of the symbol sequence with every iteration and stops before n-1th element
    while (indx<L-1): 
        a=SymSeq[indx]
        b=SymSeq[indx+1]
    
        # The pairs are transformed to indices of an array with value representing the frequency of occurrence.
        # In other words, the number of times a specific pair occurs in the sequence is stored as 
        # a count at the 2D array location with indices having the same numerical value as the pair
        # E.g. the number of times '12' occurs is stored at the location (row 1, column 2), with index (0, 1)
 
        Count_Array[a-1][b-1]=Count_Array[a-1][b-1]+1;
    
        # This block of code prevents you from counting consecutive 3 symbols with equal values 
        # E.g. if the sequence contains '1 1 1', then values at indx 1 and indx 2 are equal. 
        # The code then checks if value at indx 3 is also 1. 
        # If it is true, then indx is incremented so indx 2 is skipped and the
        # next loop starts at indx 3. 
        if a==b:
            if indx<L-2:
                if (SymSeq[indx+2]==a):
                    indx=indx+1
        indx=indx+1

    # Here, max_indx is another variable that gives the location of the pair with the maximum count
    max_indx=Count_Array.argmax()

    # max_indx variable is based on a 1D array index that runs through the matrix row-wise.
    # The 1D array index needs to be converted into 2D array indices that give the row and column information 
    # Row represents first element of the pair (Pair1)
    # Column represents the second element of the pair (Pair2)

    max_indx_2D = np.unravel_index(max_indx, Count_Array.shape)
    Pair1= max_indx_2D[0] + 1
    Pair2= max_indx_2D[1] + 1
    Pair = [Pair1, Pair2]
    
    return Pair


# Pair substitution step of NSRPS

def Substitute(SymSeq,Pair):

    SymSeqNew = []
    L = len(SymSeq)
    Alphabet = np.unique(SymSeq)
    M = max(Alphabet)

    RepSym = M+1 # New Symbol (to substitute in place of the most occurring pair)

    indx = 0
    # The indx variable shifts one symbol to the right of the sequence with every iteration
    while (indx<L-1):
        a=SymSeq[indx]
        b=SymSeq[indx+1]
    
        # If the values 'a, b' (at indx and indx+1) are equal to the value of the
        # most occurring pair (say '5, 6'), replace them with a new symbol.
        # And, slide index one symbol to the right (since there is one more slide after the if block, 
        # there are two slides, which makes the next iteration start at indx+2 after pair substitution)

        if (a==Pair[0]) and (b==Pair[1]):
            SymSeqNew.append(RepSym)
            indx=indx+1
        else:
        # Else, retain the original symbol
            SymSeqNew.append(a)
    
        # Slide index one symbol to the right 
        indx=indx+1
    
        # If the index is at the last element of the sequence, append the last
        # value of the sequence to the new sequence
        # This block works only if the last two symbols do NOT match the pair
        if indx==L-1:
            SymSeqNew.append(SymSeq[-1])

    return SymSeqNew


# Function to compute Shannon Entropy of a discrete sequence

def ShannonEntropy(sequence):

    N = len(sequence)

    # get counts and probabilities
    u = np.unique(sequence)  # u is an array that stores unique values of the sequence
    probs = np.zeros((len(u),1)) # initialize array to hold probabilities of those unique values

    # loop over unique values and calculate probabilities for that value
    for ui in range (0, len(u)):
        counts = [sequence[i]==u[ui] for i in range(0, len(sequence))] # "counts" is a logical array of same length as "sequence" with 1's wherever this condition is satisfied, and 0's where not. 
        probs[ui] = sum(counts) / N 

    log2_probs = [math.log2(probs[i]) for i in range(0,len(probs))]

    # Reshape probs array for element-wise multiplication
    probs = probs.reshape(1,-1)
    p_log2p = np.multiply(probs, log2_probs)
    p_log2p=p_log2p.reshape(-1,1) # reshape the output back for summation

    # compute entropy
    H = -sum(p_log2p)
    H[0]
    
    return H[0]

# Function to convert a time series into a symbolic sequence using the percentile method

def symbolic_sequence_percentile(InputTS):

    # Coverts an input time series into a symbolic sequence based on percentile
    # values (25th, 50th, 75th percentiles)

    # This approach is more robust against outliers and data with skewed distributions 
    # compared to previously used symbolizations.

    x=InputTS 
    SymSeq = np.zeros((len(x),1))

    # Calculate percentile values of the input time series
    edge0 = min(x)
    edge1 = np.percentile(x,25)
    edge2 = np.percentile(x,50)
    edge3 = np.percentile(x,75)
    edge4 = max(x)

    for i in range(0, len(x)):
        if (x[i]<=edge1) and (x[i]>=edge0): 
            SymSeq[i] = 1
        
        if (x[i]<=edge2) and (x[i]>edge1):
            SymSeq[i] = 2

        if (x[i]<=edge3) and (x[i]>edge2):
            SymSeq[i] = 3

        if (x[i]<=edge4) and (x[i]>edge3):
            SymSeq[i] = 4    

    # reshape SymSeq
    SymSeq = SymSeq.reshape(1,len(x))
    
    return SymSeq[0]

# Function to convert a time series into a symbolic sequence using the successive differences method

def symbolic_sequence_difference(RR_peaks):

    # Obtain successive differences

    RR_peaks_diff = np.ediff1d(RR_peaks)

    # Convert to symbolic sequence - accelerations represented by 1's and
    # decelerations represented by 0's. 

    seq_accl = np.zeros((len(RR_peaks_diff),1))

    for i in range(0, len(RR_peaks_diff)):
        if (RR_peaks_diff[i]<0): # shortening of the RR interval indicates acceleration
            seq_accl[i] = 1
        else: 
            seq_accl[i] = 0 # deceleration

    
    # Reshape the final sequence
    seq_accl = seq_accl.reshape(1, len(RR_peaks_diff))
    
    return seq_accl[0]


# Function to convert a time series into a symbolic sequence based on threshold

def symbolic_sequence_threshold(RR_peaks):

    # Obtain successive differences
    RR_peaks_diff = np.ediff1d(RR_peaks)

    # Convert to symbolic sequence - significant changes represented by 1's and
    # insignificant changes by 0's. 

    # threshold for significance (in millisecond, for RR intervals)
    tau = 50

    seq = np.zeros((len(RR_peaks_diff),1))

    for i in range(0, len(RR_peaks_diff)):
        if (abs(RR_peaks_diff[i])<tau): # 0 indicates no significant change in subsequent RR interval (absolute value)
            seq[i] = 0
        else: 
            seq[i] = 1 # 1 indicates change in subsequent RR interval by 50 ms or more (absolute value)

    # reshape the final sequence
    seq = seq.reshape(1,len(RR_peaks_diff))
    
    return seq[0]

# main

# ETC calculation function

def ETC(InputTS, symbolize):

    N = -1 # ETC measure using NSRPS is initialized to -1

    if symbolize=="None":
        SymSeq = InputTS  # input is already a symbolic sequence
    if symbolize=="percentile":
        SymSeq = symbolic_sequence_percentile(InputTS) # input is transformed to a symbolic sequence based on percentile values (quartile)
    if symbolize=="difference":
        SymSeq = symbolic_sequence_difference(InputTS) # input is transformed to a binary symbolic sequence using successive differences
    if symbolize=="threshold":
        SymSeq = symbolic_sequence_threshold(InputTS) # input is transformed to a binary symbolic sequence using a threshold
    
    # Convert all elements of the symbolic sequence to integer type (to prevent errors in the FindPair function)
    SymSeq = [int(SymSeq[i]) for i in range(0,len(SymSeq))]
    
    # Get rid of zeros in the symbolic sequence (should be in the main function of ETC)
    # Zeros need to be removed because in the FindPair function,
    # the array indices of Count_Array cannot be negative. 
    
    SymSeq = np.array(SymSeq)
    if len(SymSeq)<1:
        print("Symbol sequence is empty")
    minY = min(SymSeq)
    y = SymSeq - minY
    y = y+1
    SymSeq = y

    # The main loop for NSRPS iteration
    N = 0 # ETC measure 'N'
    Hnew = ShannonEntropy(SymSeq) # Shannon entropy of the symbolic sequence 
    L = len(SymSeq)


    while (Hnew > 1e-6) and (L > 1): 
        Pair = FindPair(SymSeq)   # find the pair of symbols with maximum frequency
        SymSeqNew = Substitute(SymSeq, Pair)  # substitute the pair with a new symbol
        Hnew = ShannonEntropy(SymSeqNew) # determine Shannon entropy of the new sequence 
        L = len(SymSeqNew) # calculate length of new sequence 
        N = N + 1 # ETC measure incremented by 1
        SymSeq = SymSeqNew
        SymSeqNew = None # clear the value of SymSeqNew for the next iteration

    # normalised ETC
    N_norm = N/(len(y)-1)   # need to use y because the length of the variable 'InputTS' may decrease by 1 if difference or threshold based symbolization is done.  
    
    return N


# Function to convert RR interval times series into a symbolic sequence based on variation in three consecutive symbols

def symbolic_dynamics(RR_intervals, symbolize):

    # Convert to symbolic sequence
    
    if symbolize=="None":
        RR_series = RR_intervals  # input is already a symbolic sequence
    if symbolize=="percentile":
        RR_series = symbolic_sequence_percentile(RR_intervals) # input is transformed to a symbolic sequence based on percentile values (quartile)
    if symbolize=="difference":
        RR_series = symbolic_sequence_difference(RR_intervals) # input is transformed to a binary symbolic sequence using successive differences
    if symbolize=="threshold":
        RR_series = symbolic_sequence_threshold(RR_intervals) # input is transformed to a binary symbolic sequence using a threshold
    

    # Initialize series representing groups of 3 consecutive symbols 
    var_series = np.zeros((len(RR_series)-2, 1))

    for i in range (0, len(RR_series)-2):
    
        # Extract sequence of 3 symbols
        sequence = RR_series[i:i+3]
   
        s1 = sequence[0]
        s2 = sequence[1]
        s3 = sequence[2]

        # Check equality of consecutive symbols
        e1 = (s1==s2)
        e2 = (s2==s3)

        # Classify the sequence as '0V', '1V' or '2V' based on amount of variation

        if ( (e1==1) and (e2==1) ):
            var_series[i] = 0

        if ( ((e1==1) and (e2==0)) or ((e1==0) and (e2==1)) ):
            var_series[i] = 1
    
        if ( (e1==0) and (e2==0) ):
            var_series[i] = 2
    
    var_series = var_series.reshape(1, len(RR_series)-2)

    # convert var_series[0] to list
    var_series_ls = var_series[0].tolist()
    
    # Calculate percentages of '0V', '1V' and '2V' patterns 
    P0V = var_series_ls.count(0)#/len(var_series_ls)
    P1V = var_series_ls.count(1)#/len(var_series_ls)
    P2V = var_series_ls.count(2)#/len(var_series_ls)
    
    return P0V, P1V, P2V

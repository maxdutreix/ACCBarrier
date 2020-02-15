#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 03:19:06 2019

@author: maxencedutreix
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from math import exp
from math import sqrt
from math import pi
from scipy.special import erf
from scipy.integrate import quad
from scipy.stats import norm
from matplotlib import rc
import timeit
import bisect
import sys
import igraph
import scipy.sparse as sparse
import scipy.sparse.csgraph
import matlab.engine
import itertools
import StringIO
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from mpl_toolkits.mplot3d import Axes3D
import copy
import igraph


def Simulation(State, Label, Automata, Optimal_Policy, Init):
    
    Total_Number_of_Simulations = 5000
    Sim_Count = 0
    Number_of_Successes = 0

    
#    plt.figure()
    
    while Sim_Count < Total_Number_of_Simulations:

        
        State_Counter = 0
        #x0=[3.1, 3.1, 3.1] # First (continuous) coordinates where the system starts
        x0 = [0.15,-0.2]
 #       x0 = [5.8,5.8,5.8]
        Time_Step_Total = 400
        Current_State = list(x0)
        j = 0;
        Tag = 0;
        
        
        while j < Time_Step_Total:
            
                      
            Next_State = list(Current_State)

            
            # Now looping through l_mapping to find corresponding label to partitioned state.
            for i in range(len(State)): # i.e. 0 to 26. Therefore, a total length of 27.
                if(Current_State[0] <= State[i][1][0] and Current_State[1] <= State[i][1][1]  and Current_State[0] >= State[i][0][0] and Current_State[1] >= State[i][0][1]):
                    if j == 0:
                        p = Init[i] #Initial automaton state
                    State_no = i; #print State_no # Verify which partitioned state I am in.
                    State_Label = Label[State_no] # Verify which label that partitioned state corresponds to.
                    #print State_Label
                    break
            
            #print p
            
            # Now looping through Automata to find corresponding automata_state via matching transition label 
            for i in range(len(Automata[p])): 
                for ii in range(len(Automata[p][i])):
                    if(Automata[p][i][ii] == State_Label):
                        Current_Auto_State = i; # Saving Current Automaton State.
                        break
    
            
            mode = Optimal_Policy[len(Automata)*State_no + Current_Auto_State] # Find the correct mode within policy
#            mode = u[0]
            

            X1 = norm(loc=0.0, scale=0.18)       

            
            if mode == 0:
            
                Next_State[0] = 6.0*(Current_State[0]**3)*Current_State[1] 
                Next_State[1] = min(max(-0.5, 0.3*Current_State[0]*Current_State[1]+ X1.rvs(1)), 0.5) # For truncation

            else:
                
                Next_State[0] = 7.0*(Current_State[0]**3)*Current_State[1] 
                Next_State[1] = min(max(-0.5, 0.2*Current_State[0]*Current_State[1]+ X1.rvs(1)), 0.5) # For truncation

    

            p = Current_Auto_State; # Updating placeholder index
            
            
            j= j+1;
                       
            Current_State = list(Next_State)
            
            if (Current_State[0] < -0.25 and Current_State[1] > 0.25) or (Current_State[0] > 0.25 and Current_State[1] < -0.25):
                break
            
            if j == 1 or j == 2:
                if (Current_State[0] > -0.25 and Current_State[0] < 0.0 and Current_State[1] > 0.25 and Current_State[1] < 0.5):
                    Tag = 1
                    break

            if (Current_State[0] > -0.5 and Current_State[0] < -0.25 and Current_State[1] > 0.0 and Current_State[1] < 0.25) or (Current_State[0] > 0.25 and Current_State[0] < 0.5 and Current_State[1] > -0.25 and Current_State[1] < 0.0):
                    Tag = 1
                    break                

          
        Sim_Count += 1
        print Sim_Count
        if Tag == 1: #We consider the simulation to be a success if the system was in the accepting region for at least the last half of the simulation
            Number_of_Successes += 1

    
    return float(Number_of_Successes)/float(Total_Number_of_Simulations)    
    

def BMDP_Probability_Interval_Computation_Barrier(Target_Set, Domain, Reachable_States):
    
    
    #Computes the lower and upper bound probabilities of transition from state
    #to state using the reachable sets in R_set and the target sets in Target_Set
    #2 is the number of modes of the system   
    
    Lower = np.array(np.zeros((2, Target_Set.shape[0],Target_Set.shape[0])))
    Upper = np.array(np.zeros((2,Target_Set.shape[0],Target_Set.shape[0])))
    Pre_States = [[[] for x in range(Target_Set.shape[0])] for y in range(2)]
    Is_Bridge_State = np.zeros((2,Target_Set.shape[0])) # Will not be used
    Bridge_Transitions = [[[] for x in range(Target_Set.shape[0])] for y in range(2)] #Will not be used
 
        
    
    eng = matlab.engine.start_matlab() #Start Matlab Engine
    

    
    for k in range(Target_Set.shape[0]):
        for j in range(Target_Set.shape[1]):
            for h in range(len(Reachable_States[k][j])):
                
    
                out = StringIO.StringIO()
                err = StringIO.StringIO()
                            
                Res = eng.Bounds_Computation_Synthesis(matlab.double(list(itertools.chain.from_iterable(Target_Set[j].tolist()))), matlab.double(list(itertools.chain.from_iterable(Target_Set[Reachable_States[k][j][h]].tolist()))), matlab.double(list([0.0, 1.0])), matlab.double(Domain), matlab.double([k]), stdout=out,stderr=err)
           
                H = Res[0][0]
                L = Res[0][1]
                if H > 0:
                    if L == 0:
                        Is_Bridge_State[k][j] = 1
                        Bridge_Transitions[k][j].append(Reachable_States[k][j][h])
                else:
                    Reachable_States[k][j].remove(Reachable_States[k][j][h])
                        
                Lower[k][j][h] = L
                Upper[k][j][h] = H
                
                
                         

    
    return Lower,Upper, Reachable_States, Is_Bridge_State, Bridge_Transitions, Pre_States




def Build_Product_BMDP(T_l, T_u, A, L, Acc, Reachable_States, Is_Bridge_State, Bridge_Transitions):
    
    # Constructs the product between an IMC (defined by lower transition matrices
    # T_l and T_u and an Automata A according to Labeling function L
    # For simplicity, each state has the same number of actions
    
    #For init, we assume initial state is 0
    Init = np.zeros((T_l.shape[1])) # Corresponds to number of system states
    Init = Init.astype(int)    
    
    Is_A = np.zeros(T_l.shape[1]*len(A))
    Is_N_A = np.zeros(T_l.shape[1]*len(A))
    Which_A = [[] for x in range(T_l.shape[1]*len(A))]
    Which_N_A = [[] for x in range(T_l.shape[1]*len(A))]
    
    New_Reachable_States = [[[] for x in range(T_l.shape[1]*len(A))] for y in range(T_l.shape[0])]
    New_Is_Bridge_State = np.zeros((T_l.shape[0], T_l.shape[1]*len(A)))
    New_Bridge_Transitions = [[[] for x in range(T_l.shape[1]*len(A))] for y in range(T_l.shape[0])]
    
    IA_l = np.zeros((T_l.shape[0],T_l.shape[1]*len(A), T_l.shape[2]*len(A)))
    IA_u = np.zeros((T_l.shape[0],T_l.shape[1]*len(A), T_l.shape[2]*len(A)))
    
    for x in range(len(Acc)): # Which states in the product are accepting or not.
        for i in range(len(Acc[x][0])):
            for j in range(T_l.shape[1]):
                Is_N_A[len(A)*j + Acc[x][0][i]] = 1
                Which_N_A[len(A)*j + Acc[x][0][i]].append(x)
        
        for i in range(len(Acc[x][1])):
            for j in range(T_l.shape[1]):
                Is_A[len(A)*j + Acc[x][1][i]] = 1
                Which_A[len(A)*j + Acc[x][1][i]].append(x)            
                
    
    for y in range(T_l.shape[0]): # For every action (mode)
        for i in range(T_l.shape[1]):
            for j in range(len(A)):
                for k in range(T_l.shape[2]):
                    for l in range(len(A)):
                        
                        if L[k] in A[j][l]:
                            
                            if y == 0 and j == 0: # To account for true first state. y == 0 because it goes through it once and for all for the first action (mode)
                                Init[k] = l
    
                            IA_l[y, len(A)*i+j, len(A)*k+l] = T_l[y,i,k] # then, if the transition exists, then 
                            IA_u[y, len(A)*i+j, len(A)*k+l] = T_u[y,i,k] # probability T_up and T_down are the same from the IMC abstraction probability interval
                            
                            # If there is more than 1 mode, then take into account of the action from the policy
                            
                            if T_u[y,i,k] > 0:
                                New_Reachable_States[y][len(A)*i+j].append(len(A)*k+l)
                                if T_l[y,i,k] == 0:
                                    New_Is_Bridge_State[y, len(A)*i+j] = 1 # Keep track of new bridge states within the IMC states
                                    New_Bridge_Transitions[y][len(A)*i+j].append(len(A)*k+l)
                            
                        else:
                            IA_l[y, len(A)*i+j, len(A)*k+l] = 0.0
                            IA_u[y, len(A)*i+j, len(A)*k+l] = 0.0
    
                 

    Is_A = Is_A.astype(int)
    Is_N_A = Is_N_A.astype(int)
    New_Is_Bridge_State = New_Is_Bridge_State.astype(int)                         

    return (IA_l, IA_u, Is_A, Is_N_A, Which_A, Which_N_A, New_Reachable_States, New_Is_Bridge_State, New_Bridge_Transitions, Init) 







def Find_Greatest_Accepting_BSCCs(IA1_l, IA1_u, Is_Acc, Is_NAcc, Wh_Acc_Pair, Wh_NAcc_Pair, Al_Act_Pot, Al_Act_Perm, first, Reachable_States, Bridge_Transition, Is_Bridge_State, Acc, Potential_Policy, Permanent_Policy, Is_In_Permanent_Comp, List_Permanent_Acc_BSCC, Previous_A_BSCC):

    G = np.zeros((IA1_l.shape[1],IA1_l.shape[2]))

    
    if first == 1:
        Al_Act_Pot = list([])
        Al_Act_Perm = list([])
        for y in range(IA1_l.shape[1]): # For all the system states
            Al_Act_Pot.append(range(IA1_l.shape[0])) # Appending all allowable actions in the beginning
            Al_Act_Perm.append(range(IA1_l.shape[0]))
    
    
    for k in range(IA1_u.shape[0]): # Out of all the upper bounds of every mode, if at least one is greater than 0, then, G is 1 for that transition between the two states
        for i in range(IA1_u.shape[1]): # Perhaps, might need to specify which actions allow us to make G equal to one.
            for j in range(IA1_u.shape[2]):
                if IA1_u[k,i,j] > 0: # It's an array, not a list. 
                    G[i,j] = 1

    Counter_Status2 = 0 #Indicates which Status2 BSCC we are currently inspecting
    Counter_Status3 = 0 #Indicates which Status2 BSCC we are currently inspecting
    Which_Status2_BSCC = [] #Tells you with respect to which BSCC are the states duplicated
    Has_Found_BSCC_Status_2 = list([]) #0 if you found a BSCC in the duplicate, 1 otherwise
    List_Found_BSCC_Status_2 = list([]) #Will contain the set of states for which an accepting BSCC has been found in duplicates
    Original_SCC_Status_2 = list([]) #Keeps track of the original SCC before duplication
    Which_Status3_BSCC = list([])
    Number_Duplicates2 = 0 #Tells you how many BSCCs have been duplicated so far for status 2
    Number_Duplicates3 = 0 #Tells you how many BSCCs have been duplicated so far for status 3
    Status2_Act = list([]) #List which keeps track of the allowed actions for duplicate states
    Status3_Act = list([])
    List_Status3_Found = list([])

    if first == 0:
        Deleted_States = []
        Prev_A = set().union(*Previous_A_BSCC)
        Deleted_States.extend(list(set(range(G.shape[0])) - set(Prev_A)))
        
        Ind = list(set(Prev_A))
        Ind.sort()
        
        G = np.delete(np.array(G),Deleted_States,axis=0)
        G = np.delete(np.array(G),Deleted_States,axis=1)
        

    else:
        Ind = range(G.shape[0])
     
#    Ind = range(G.shape[0])    
        
    first = 0 

   
    C,n = SSCC(G) # Search SCC using G. n = number of SCCs. C contains all the SCCs    
    tag = 0; # Trackers for indices
    m = 0 ;

    
    SCC_Status = [0]*n ###Each SCC has 'status': 0: looking for potential BSCCs, 1: looking for permanent BSCCs, 2: duplicate potential BSCCs , 3: duplicate permanent BSCCs
   
    G_Pot_Acc_BSCCs = list([]) #Contains the set of greatest potential accepting BSCCs
    G_Per_Acc_BSCCs = list([]) #Contains the set of greatest permanent accepting BSCC
    
    
    for i in range(len(List_Permanent_Acc_BSCC)): #Have to add them now since they are deleted from the graph upon searching for the BSCC
        for j in range(len(List_Permanent_Acc_BSCC[i])):
            G_Pot_Acc_BSCCs.append(List_Permanent_Acc_BSCC[i][j])
            G_Per_Acc_BSCCs.append(List_Permanent_Acc_BSCC[i][j])
    
    List_G_Pot = [] #Lists the greatest potential BSCCs (which are not cross listed with permanent BSCCs)
    Is_In_Potential_Acc_BSCC = np.zeros(IA1_l.shape[1]) #Is the state in the largest potential accepting BSCC?
    Which_Potential_Acc_BSCC = np.zeros(IA1_l.shape[1]) #Keeps track of which accepting BSCC does each state belong to (if applicable)
    Which_Potential_Acc_BSCC.astype(int)
    Is_In_Potential_Acc_BSCC.astype(int)
    Bridge_Potential_Accepting = [] #List which contains the bridge states for each potential accepting BSCC
    Maybe_Permanent = [] #List which contains potential components before checking whether these components are permanent or not
    
    
    
    while tag == 0:
        
        if len(C) == 0:
            break
        
        
        skip = 1 # To reset skip tag. Assume that I skip
        SCC = C[m];
        

        
#        print len(SCC)

        Orig_SCC = []
        for k in range(len(SCC)):
            Orig_SCC.append(Ind[SCC[k]]) # absolute index/label of states is added to the list of states in the Orig_SCC
        BSCC = 1
        # if there are no accepting states in the given SCC, then continue, because it cannot be a winning component.
    
        # Search through Orig_SCC, in which you would find the absolute index/label of states, which I can use to find the corresponding Is_Accepting TRUTH/FALSE value.
        
        if len(Has_Found_BSCC_Status_2) != 0:
            if SCC_Status[m] == 2 and Has_Found_BSCC_Status_2[Which_Status2_BSCC[Counter_Status2]] == 1:
                Counter_Status2 += 1
                if m < (len(C)-1): # To avoid searching beyond the range of the C, which is the list of SCCs. Related to C[m]
                    m += 1 # Tag to move on to the next SCC in the list of SCCs in C.                   
                    continue # then while through the next SCC. Related SCC = C[m]
                else:                   
                    break 



        for l in range(len(Orig_SCC)):
            if Is_Acc[Orig_SCC[l]] == 1: # Keep searching until the given state in the Orig_SCC is accepting.
                skip = 0
                break
            # If none of the states in the Orig_SCC is accepting, then THIS SCC IS NOT WORTH MY TIME!

        if skip == 1: # If skip tag was activated, because no accepting state was found in the Orig_SCC,
            if SCC_Status[m] == 0: #These states will never be a potential or permanent accepting BSCCs since they do have an accepting state (either from the beginning or accepting states were leaky and removed), removed all allowed actions
                for i in range(len(SCC)):
                    Al_Act_Pot[Ind[SCC[i]]] = list([])
                    Al_Act_Perm[Ind[SCC[i]]] = list([])            
            if m < (len(C)-1): # To avoid searching beyond the range of the C, which is the list of SCCs. Related to C[m]
                m += 1 # Tag to move on to the next SCC in the list of SCCs in C.
                continue # then while through the next SCC. Related SCC = C[m]
            else: 
                break     # if at end of the list. want to break.

        # Allowable_Action = [[[]]], which is what we want eventually
            
        
        
        Leak = list([])
        Check_Tag = 1
        Reach_in_R = [[[] for y in range(IA1_u.shape[0])] for x in range(len(Orig_SCC))] # Reach_in_R contains all the reachable non-leaky states inside the SCC, with respect to state i.
        Pre = [[[] for y in range(IA1_u.shape[0])] for x in range(len(Orig_SCC))] # Creating list of list of lists, to account for mode, state, transitions. Modes are nested inside state.
        All_Leaks = list([])
        Check_Orig_SCC = np.zeros(len(Orig_SCC), dtype=int)
        

            
            

        while (len(Leak) != 0 or Check_Tag == 1):
                       
                        
            if SCC_Status[m] == 0:
                                                              
                ind_leak = []
                Leak = []
                      
                for i in range(len(Orig_SCC)): # Original SCC that contains all SCCs
                    if Check_Orig_SCC[i] == -1 :
                        continue # -1 is a tag for leaky state which should be skipped over.
                    tag_m = 0# tag_mode
                    
                    for k in range(len(Al_Act_Pot[Orig_SCC[i]])): # Loop through all the allowable actions from the current state
                            # The state number if the index for the allowable action array
        
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) # Removes all leaky states from Orig_SCC
                        Diff_List1 = list(set(Reachable_States[Al_Act_Pot[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) # All Reachable states (that is outside the Set_All_Leaks) of the current state. Al_Act[i][k] means action from the set of allowable actions with respect to current state i
                        Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[Al_Act_Pot[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]])) # After taking out bridge transitions, then if it's not 0, then it's a leaky state 
                        # Bridge_Transition contains all the states that contain the "dashed transitions", with respect to the current state
                        if Check_Tag == 1: # tag to create "Pre" and Reach_in_R.
                            # "Orig_SCC[i]"th state inside the Al_Act, and the kth action of that list inside Al_Act

                            Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]].extend(list(set(Reachable_States[Al_Act_Pot[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - set(Diff_List1)))
                            for j in range(len(Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]])):
                                Pre[Orig_SCC.index(Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]][j])][Al_Act_Pot[Orig_SCC[i]][k-tag_m]].append(Orig_SCC[i]) # With respect to state i, "pre" contains all the states inside the SCC that can reach State i. The reason for having "Pre" is to know which states inside the SCC lead to the state i, (which is useful if state i is a leaky state)
                       
#                        if Orig_SCC[i] Al_Act_Perm[Orig_SCC[i]]
    
                        if (len(Diff_List2) != 0) or (sum(IA1_u[Al_Act_Pot[Orig_SCC[i]][k-tag_m], Orig_SCC[i], Reach_in_R[i][Al_Act_Pot[Orig_SCC[i]][k-tag_m]]])<1) : # Reach_in_R means reachable states inside scc
                            # the sum statement means that if all the upper bounds of the transitions within the SCC don't add up to 1, then there is always the possibility of a leakiness.
                            
                            Al_Act_Perm[Orig_SCC[i]].remove(Al_Act_Pot[Orig_SCC[i]][k-tag_m]) #If the action cannot make a potential BSCC, then it cannot make a permanent BSCC either
                            Al_Act_Pot[Orig_SCC[i]].remove(Al_Act_Pot[Orig_SCC[i]][k-tag_m]) # Remove the leaky mode from that list of allowable action for the current state i.
                            tag_m += 1 # To account for the missing index in the "for k" loop.
                            BSCC = 0 # Because once action is removed, might have changed the whole SCC, so might as well just keep BSCC to be 0.
                            
                    if len(Al_Act_Pot[Orig_SCC[i]]) == 0: # If there are no more available actions for the current state i.                       
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                        
                if len(Leak) != 0: # It means that a new leaky state is found, because Leak=[] every loop of the while
                    All_Leaks.extend(Leak) # Then add to All_leaks. extend means "adding" a list without brackets
                    BSCC = 0 # To confirm that previous SCC is surely not a BSCC, until further verifications to see if SCC is a BSCC
                    for i in range(len(Leak)): # for all the newly found leaky states
                        Check_Orig_SCC[ind_leak[i]] = -1 # The state in the SCC is tagged "leaky"
                        for j in range(len(Pre[ind_leak[i]])): # Loop through all the actions of the pre-states, with respect to the leaky current state i.
                            for k in range(len(Pre[ind_leak[i]][j])): # Loop through all the states in each respective action of the pre-states of the leaky current state i.
                                Reach_in_R[Orig_SCC.index(Pre[ind_leak[i]][j][k])][j].remove(Leak[i]) # Removes leaky state from the reachable set of states of all OTHER states in the SCC, if the leaky state is reachable from those states
            # Have to loop through all the modes for which there are all SCCs.
                Check_Tag = 0  # Changes Check_Tag after having populated the "Pre" and Reach_in_R for all the states in set of SCCs. But now need to do it for all actions (modes).
 
            if SCC_Status[m] == 1:
                
                

 
                ind_leak = []
                Leak = []                     
                for i in range(len(Orig_SCC)): # Original SCC that contains all SCCs
                    if Check_Orig_SCC[i] == -1 :
                        continue # -1 is a tag for leaky state which should be skipped over.
                    tag_m = 0# tag_mode
                     


                    for k in range(len(Al_Act_Perm[Orig_SCC[i]])): # Loop through all the allowable actions from the current state
                            # The state number if the index for the allowable action array
              
                
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) # Removes all leaky states from Orig_SCC

                        Diff_List1 = list(set(Reachable_States[Al_Act_Perm[Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) # All Reachable states (that is outside the Set_All_Leaks) of the current state. Al_Act[i][k] means action from the set of allowable actions with respect to current state i

                        if (len(Diff_List1) != 0): # If some state is reachable outside of the SCC, then the SCC is not permanent for wrt that action                           
                            Al_Act_Perm[Orig_SCC[i]].remove(Al_Act_Perm[Orig_SCC[i]][k-tag_m]) #If the action cannot make a potential BSCC, then it cannot make a permanent BSCC either
                            tag_m += 1 # To account for the missing index in the "for k" loop.
                            BSCC = 0 # Because once action is removed, might have changed the whole SCC, so might as well just keep BSCC to be 0.
                                                    
                    if len(Al_Act_Perm[Orig_SCC[i]]) == 0: # If there are no more available actions for the current state i.                        
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                if len(Leak) != 0: # It means that a new leaky state is found, because Leak=[] every loop of the while
                    All_Leaks.extend(Leak) # Then add to All_leaks. extend means "adding" a list without brackets
                    BSCC = 0 # To confirm that previous SCC is surely not a BSCC, until further verifications to see if SCC is a BSCC
                    for i in range(len(Leak)): # for all the newly found leaky states
                        Check_Orig_SCC[ind_leak[i]] = -1 # The state in the SCC is tagged "leaky"
                Check_Tag = 0  # Changes Check_Tag after having populated the "Pre" and Reach_in_R for all the states in set of SCCs. But now need to do it for all actions (modes).


            if SCC_Status[m] == 2:
                
                                                             
                ind_leak = []
                Leak = []                      
                for i in range(len(Orig_SCC)): # Original SCC that contains all SCCs
                    if Check_Orig_SCC[i] == -1 :
                        continue # -1 is a tag for leaky state which should be skipped over.
                    tag_m = 0# tag_mode
                    
                    for k in range(len(Status2_Act[Counter_Status2][Orig_SCC[i]])): # Loop through all the allowable actions from the current state
                            # The state number if the index for the allowable action array      
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) # Removes all leaky states from Orig_SCC
                        Diff_List1 = list(set(Reachable_States[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) # All Reachable states (that is outside the Set_All_Leaks) of the current state. Al_Act[i][k] means action from the set of allowable actions with respect to current state i
                        Diff_List2 = list(set(Diff_List1) - set(Bridge_Transition[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]])) # After taking out bridge transitions, then if it's not 0, then it's a leaky state 
                        # Bridge_Transition contains all the states that contain the "dashed transitions", with respect to the current state
                        if Check_Tag == 1: # tag to create "Pre" and Reach_in_R.
                            # "Orig_SCC[i]"th state inside the Al_Act, and the kth action of that list inside Al_Act

                            Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]].extend(list(set(Reachable_States[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - set(Diff_List1)))
                            for j in range(len(Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]])):
                                Pre[Orig_SCC.index(Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]][j])][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]].append(Orig_SCC[i]) # With respect to state i, "pre" contains all the states inside the SCC that can reach State i. The reason for having "Pre" is to know which states inside the SCC lead to the state i, (which is useful if state i is a leaky state)
                       
    
                        if (len(Diff_List2) != 0) or (sum(IA1_u[Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m], Orig_SCC[i], Reach_in_R[i][Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]]])<1) : # Reach_in_R means reachable states inside scc
                            # the sum statement means that if all the upper bounds of the transitions within the SCC don't add up to 1, then there is always the possibility of a leakiness.
                            
#                            Al_Act_Perm[Orig_SCC[i]].remove(Al_Act_Pot[Orig_SCC[i]][k-tag_m]) #If the action cannot make a potential BSCC, then it cannot make a permanent BSCC either
                            Status2_Act[Counter_Status2][Orig_SCC[i]].remove(Status2_Act[Counter_Status2][Orig_SCC[i]][k-tag_m]) # Remove the leaky mode from that list of allowable action for the current state i.
                            tag_m += 1 # To account for the missing index in the "for k" loop.
                            BSCC = 0 # Because once action is removed, might have changed the whole SCC, so might as well just keep BSCC to be 0.
                            
                    if len(Status2_Act[Counter_Status2][Orig_SCC[i]]) == 0: # If there are no more available actions for the current state i.                        
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                        
                if len(Leak) != 0: # It means that a new leaky state is found, because Leak=[] every loop of the while
                    All_Leaks.extend(Leak) # Then add to All_leaks. extend means "adding" a list without brackets
                    BSCC = 0 # To confirm that previous SCC is surely not a BSCC, until further verifications to see if SCC is a BSCC
                    for i in range(len(Leak)): # for all the newly found leaky states
                        Check_Orig_SCC[ind_leak[i]] = -1 # The state in the SCC is tagged "leaky"
                        for j in range(len(Pre[ind_leak[i]])): # Loop through all the actions of the pre-states, with respect to the leaky current state i.
                            for k in range(len(Pre[ind_leak[i]][j])): # Loop through all the states in each respective action of the pre-states of the leaky current state i.
                                Reach_in_R[Orig_SCC.index(Pre[ind_leak[i]][j][k])][j].remove(Leak[i]) # Removes leaky state from the reachable set of states of all OTHER states in the SCC, if the leaky state is reachable from those states
            # Have to loop through all the modes for which there are all SCCs.
                Check_Tag = 0  # Changes Check_Tag after having populated the "Pre" and Reach_in_R for all the states in set of SCCs. But now need to do it for all actions (modes).
 

            if SCC_Status[m] == 3:
                
                                
                ind_leak = []
                Leak = []                     
                for i in range(len(Orig_SCC)): # Original SCC that contains all SCCs
                    if Check_Orig_SCC[i] == -1 :
                        continue # -1 is a tag for leaky state which should be skipped over.
                    tag_m = 0# tag_mode
            
                    for k in range(len(Status3_Act[Counter_Status3][Orig_SCC[i]])): # Loop through all the allowable actions from the current state
                          
                        # The state number if the index for the allowable action array
                                                   
                                        
                        Set_All_Leaks = set(Orig_SCC) - set(All_Leaks) # Removes all leaky states from Orig_SCC           
                        
                        
                        Diff_List1 = list(set(Reachable_States[Status3_Act[Counter_Status3][Orig_SCC[i]][k-tag_m]][Orig_SCC[i]]) - Set_All_Leaks) # All Reachable states (that is outside the Set_All_Leaks) of the current state. Al_Act[i][k] means action from the set of allowable actions with respect to current state i
    

    
                        if (len(Diff_List1) != 0): # If some state is reachable outside of the SCC, then the SCC is not permanent for wrt that action                           
                            Status3_Act[Counter_Status3][Orig_SCC[i]].remove(Status3_Act[Counter_Status3][Orig_SCC[i]][k-tag_m]) #If the action cannot make a potential BSCC, then it cannot make a permanent BSCC either
                            tag_m += 1 # To account for the missing index in the "for k" loop.
                            BSCC = 0 # Because once action is removed, might have changed the whole SCC, so might as well just keep BSCC to be 0.
                            
                    if len(Status3_Act[Counter_Status3][Orig_SCC[i]]) == 0: # If there are no more available actions for the current state i.                        
                        Leak.append(Orig_SCC[i])
                        ind_leak.append(i)
                if len(Leak) != 0: # It means that a new leaky state is found, because Leak=[] every loop of the while
                    All_Leaks.extend(Leak) # Then add to All_leaks. extend means "adding" a list without brackets
                    BSCC = 0 # To confirm that previous SCC is surely not a BSCC, until further verifications to see if SCC is a BSCC
                    for i in range(len(Leak)): # for all the newly found leaky states
                        Check_Orig_SCC[ind_leak[i]] = -1 # The state in the SCC is tagged "leaky"
                Check_Tag = 0  # Changes Check_Tag after having populated the "Pre" and Reach_in_R for all the states in set of SCCs. But now need to do it for all actions (modes).



               
        if BSCC == 0: # Means actions are modified/removed, and subsequently the state may be removed if all actions are removed. Need to check the connectivity of the remaining states of the remaining states in the SCC.
            
            
            SCC = list(set(Orig_SCC) - set(All_Leaks))
            for k in range(len(SCC)):
                SCC[k] = Ind.index(SCC[k])
            
            if SCC_Status[m] == 0: #Looking for greatest potential 
                
                
            #Could be optimized, convert back non-leaky states to indices of reduced graph
                                
                if len(SCC) != 0: # if some states are left in the SCC
                    SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                    New_G = np.zeros((len(SCC), len(SCC)))# Create new graph
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Pot[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:
                                    New_G[i,j] = 1
                        
                    C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label               
                    for j in range(len(C_new)):
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] # Converting C_new to SCC label of the states
                        C.append(C_new[j]) # put them in the front.
                        SCC_Status.append(0)

            if SCC_Status[m] == 1: #Looking for permanent
                
                
                
            #Could be optimized, convert back non-leaky states to indices of reduced graph               
                if len(SCC) != 0: # if some states are left in the SCC
                    SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                    New_G = np.zeros((len(SCC), len(SCC)))# Create new graph
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Perm[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Perm[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:                                        
                                    New_G[i,j] = 1
                    C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label               
                    

                    
                    for j in range(len(C_new)):
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] # Converting C_new to SCC label of the states
                        C.append(C_new[j]) # put them in the front.
                        SCC_Status.append(1)

            if SCC_Status[m] == 2:
                                                
                            
                if len(SCC) != 0: # if some states are left in the SCC
                    Duplicate_Actions = copy.deepcopy(Status2_Act[Counter_Status2])
                    SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                    New_G = np.zeros((len(SCC), len(SCC)))# Create new graph
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Pot[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:
                                    New_G[i,j] = 1

                    C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label               
                    for j in range(len(C_new)):
                        Status2_Act.append(Duplicate_Actions)
                        Which_Status2_BSCC.append(Which_Status2_BSCC[Counter_Status2])
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] # Converting C_new to SCC label of the states
                        C.append(C_new[j]) # put them in the front.
                        SCC_Status.append(2)                                        
                Counter_Status2 += 1

            if SCC_Status[m] == 3: 
                
                
                if len(SCC) != 0: #
                    Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                    SCC = sorted(SCC, key=int)               
                    New_G = np.zeros((len(SCC), len(SCC)))
                    for i in range(len(SCC)):
                        for k in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                            for j in range(len(SCC)):
                                if IA1_u[Al_Act_Pot[Ind[SCC[i]]][k], Ind[SCC[i]], Ind[SCC[j]]] > 0:
                                    New_G[i,j] = 1

                    C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label               
                    for j in range(len(C_new)):
                        Status3_Act.append(Duplicate_Actions)
                        Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])
                        for k in range(len(C_new[j])):
                            C_new[j][k] = SCC[C_new[j][k]] # Converting C_new to SCC label of the states
                        C.append(C_new[j]) # put them in the front.
                        SCC_Status.append(3)                                        
                Counter_Status3 += 1                
            
        else: # it means the SCC is a BSCC
            
            
            Bridge_States = []           
            if SCC_Status[m] == 0:
                            

                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: # accepting, then add it as accepting and vice versa
                        acc_states.append(SCC[j])
                        indices = [] # Establish index list
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): # loop through to find which accepting pair-conditions are sufficed by the given state
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) # Add the respective absolute index/label for the states
                        ind_acc.append(indices) # Add the accumulated list of indices with respect to the BSCC

                    if Is_NAcc[Ind[SCC[j]]] == 1: # do the same thing for non-accepting states in the BSCC
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                    for i in range(len(Al_Act_Pot[Ind[SCC[j]]])): # For a given list of allowable actions for a given state in the SCC
                        if Is_Bridge_State[Al_Act_Pot[Ind[SCC[j]]][i]][Ind[SCC[j]]] == 1: # Is the given state in the SCC a bridge state? If yes,                                                                  
                            Diff_List = np.setdiff1d(Reachable_States[Al_Act_Pot[Ind[SCC[j]]][i]][Ind[SCC[j]]], Orig_SCC) # Subtract all states within SCC from the list of reachable states from the current state. Then-
                            if len(Diff_List) != 0: # is there anything that remains?
                                Inevitable = 0  # If so, then inevitability/permanence = 0, which means that the current SCC is no bueno in the permanency test
                            Bridge_States.append(Ind[SCC[j]]) # Then bridge_states list is constructed for the given SCC.
                            #not using this Bridge_States variable at the moment

                Acc_Tag = 0
                Accept = [] #Contains unmatched accepting states
                                                   
                if len(non_acc_states) == 0: # If there are no non-accepting states,
                    Acc_Tag = 1 # then activate tag for accepting BSCC.
                    for j in range(len(acc_states)): # Subsequently, add to the list of accepting BSCC.
                        Accept.append(acc_states[j])
                
                else:
                    
                    Non_Accept_Remove = [[] for x in range(len(Acc))] # Contains all non-accepting states which prevent the bscc to be accepting for all pairs                        
                    for j in range(len(ind_acc)): # Recall that ind_acc contains the accumulated list of indices of the pairs with respect to the BSCC
                        for l in range(len(ind_acc[j])): # ind_acc[j] contains the indices for the relevant DRA pair, to which the state of the BSCC was complying for acceptance
                            Check_Tag = 0
                            Keep_Going = 0
                            for w in range(len(ind_non_acc)): # Same thing for non accepting
                                if ind_acc[j][l] in ind_non_acc[w]: # Checks if accepting index is in list of non_accepting indices ?? YES
                                    Check_Tag = 1 # Means that the current index in the accepting states doesn't make the BSCC accepting
                                    if len(Non_Accept_Remove[ind_acc[j][l]]) == 0: # If the list of non-accepting states that must be removed is empty, then
                                        Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w]) # add to list of Non-Accepting states to be removed in the BSCC.
                                        Keep_Going = 1 # Stops checking the state, because we know it has to be removed
                                    elif Keep_Going == 0: # when there are no states to be removed in the BSCC
                                        break
                            if Check_Tag == 0: 
                                Accept.append(acc_states[j]) # add the list of accepting states
                                Acc_Tag = 1  
                

                if Acc_Tag == 1: #If the potential greatest BSCC is accepting
                    #Compute the policy that maximizes the lower bound probability of reaching unmatched accepting states 
                    SCC.sort()
                    Accept.sort()
                    Potential_Policy_BSCC = np.zeros(len(SCC))
                    Permanent_Policy_BSCC = np.zeros(len(SCC))  
                    for i in range(len(Accept)):   #Converts indices of SCC for reachability computation
                        Act1 = Al_Act_Perm[Ind[Accept[i]]][0]
                        Act2 = Al_Act_Pot[Ind[Accept[i]]][0] 
                        Accept[i] = SCC.index(Accept[i])
                        Permanent_Policy_BSCC[Accept[i]] = Act1
                        Potential_Policy_BSCC[Accept[i]] = Act2
                           
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    

                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Al_Act_Pot[Ind[SCC[i]]])):
                                if IA1_u_BSCC[Al_Act_Pot[Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Al_Act_Pot[Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Al_Act_Pot[Indices[i]])  
                    
                    # Computes the optimal action to maximize the upper-bound   
                    (Dummy_Reach, Dummy_Upp_Bounds, Dummy_Chain, Potential_Policy_BSCC, Dum) = Maximize_Upper_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Potential_Policy_BSCC, BSCC_Allowed_Actions) # Minimizes Upper Bound
                    for i in range(len(SCC)):
                        Potential_Policy[Ind[SCC[i]]] = Potential_Policy_BSCC[i]
                        G_Pot_Acc_BSCCs.append(Ind[SCC[i]])
                    

                    Maybe_Permanent.append(SCC)
                    
                                       
                    # Now, need to check if the BSCC doesn't leak, and if it doesn't, we can directly check if it is permanent or not   
                                       
                    if Inevitable == 1: #The current allowed actions cannot make the BSCC leak. To check for permanence (that is, no possibility of creating a sub non-accepting BSCC with this BSCC), we compute the policy that maximizes the lower_bound probability of reaching the accepting states. If this lower bound is zero for some states, then these are not permanent with respect to the BSCC


                          
                        
                        (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions) # Minimizes Upper Bound
                        Bad_States = []
                        for i in range(len(Dummy_Low_Bounds)):
                            if Dummy_Low_Bounds[i] == 0: #If some states have a lower-bound zero of reaching an accepting state inside the BSCC, then it means there will always be a scenario where those states form a non-accepting BSCC, and therefore cannot be part of a permanent BSCC
                                Bad_States.append(SCC[i])
                        if len(Bad_States) == 0:                            
                            List_Permanent_Acc_BSCC.append([])
                            for i in range(len(SCC)):                                
                                Permanent_Policy[Ind[SCC[i]]] = Permanent_Policy_BSCC[i]
                                Potential_Policy[Ind[SCC[i]]] = Permanent_Policy[Ind[SCC[i]]] #Make both policies equal to avoid any bridge state
                                G_Per_Acc_BSCCs.append(Ind[SCC[i]])
                                G_Pot_Acc_BSCCs.append(Ind[SCC[i]])
                                Is_In_Permanent_Comp[Ind[SCC[i]]] = 1
                                List_Permanent_Acc_BSCC[-1].append(Ind[SCC[i]])
                            Maybe_Permanent.pop()
                        else:
                                  

                            SCC_New = list(set(SCC) - set(Bad_States)) #Create new set of states without the states to be removed
                            SCC_New.sort()        
                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                #SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
#                                    for k in range(len(SCC)):
#                                        SCC[k] = Ind.index(SCC[k])
                                    
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                              
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    C.append(C_new[j]) # put them in the front.
                                    SCC_Status.append(1)
                          

                            SCC_New = list(Bad_States) #Create new set of states without the states to be removed
                            SCC_New.sort()        
                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                #SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
#                                    for k in range(len(SCC)):
#                                        SCC[k] = Ind.index(SCC[k])
                                    
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                              
                                for j in range(len(C_new)):
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    C.append(C_new[j]) # put them in the front.
                                    SCC_Status.append(1)                            
                             
                                                  
                    else:
                        #We now need to check this SCC for permanence
                        C.append(SCC)
                        SCC_Status.append(1)
                        
                                                    
                else: #If it is not accepting, then we need to remove the non-accepting states prevening it from being accepting                    
                    
                    Check_Tag2 = 0
                    Count_Duplicates = 0
                    #We now have BSCCs which share the same states. Need to figure out what to do with respect to the actions
                    
                    
                    for j in range(len(Non_Accept_Remove)):
                        if len(Non_Accept_Remove[j]) != 0:
                            Count_Duplicates += 1
                    
                    for j in range(len(Non_Accept_Remove)): #Loop through the set of states to remove
                        if len(Non_Accept_Remove[j]) != 0:
                    
                            if Check_Tag2 == 0 and Count_Duplicates > 1:
                                Duplicate_Actions = copy.deepcopy(Al_Act_Pot)
                                Has_Found_BSCC_Status_2.append(0)
                                List_Found_BSCC_Status_2.append([])
                                Original_SCC_Status_2.append(SCC)
                                Check_Tag2 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) #Create new set of states without the states to be removed
                            SCC_New.sort()

                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
#                                    for k in range(len(SCC)):
#                                        SCC[k] = Ind.index(SCC[k])
                                    
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                                
                                for j in range(len(C_new)):
                                    if Count_Duplicates > 1:
                                        Status2_Act.append(Duplicate_Actions)
                                        Which_Status2_BSCC.append(Number_Duplicates2)                                    
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    
                                    C.append(C_new[j]) # put them in the front.
                                   
                                    if Count_Duplicates > 1:
                                        SCC_Status.append(2)
                                    else:    
                                        SCC_Status.append(0)
                                        
                    if Check_Tag2 == 1 and Count_Duplicates > 1:
                        Number_Duplicates2 += 1

                        
            elif SCC_Status[m] == 1: #Checking for permanence of the SCC
                
                #print SCC

                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                
                               
                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: # accepting, then add it as accepting and vice versa
                        acc_states.append(SCC[j])
                        indices = [] # Establish index list
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): # loop through to find which accepting pair-conditions are sufficed by the given state
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) # Add the respective absolute index/label for the states
                        ind_acc.append(indices) # Add the accumulated list of indices with respect to the BSCC

                    if Is_NAcc[Ind[SCC[j]]] == 1: # do the same thing for non-accepting states in the BSCC
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                Acc_Tag = 0
                Accept = [] #Contains unmatched accepting states
                                                    
                if len(non_acc_states) == 0: # If there are no non-accepting states,
                    Acc_Tag = 1 # then activate tag for accepting BSCC.
                    for j in range(len(acc_states)): # Subsequently, add to the list of accepting BSCC.
                        Accept.append(acc_states[j])
                
                else:                                        
                  
                    Non_Accept_Remove = [[] for x in range(len(Acc))] # Contains all non-accepting states which prevent the bscc to be accepting for all pairs                        
                    for j in range(len(ind_acc)): # Recall that ind_acc contains the accumulated list of indices of the pairs with respect to the BSCC
                        for l in range(len(ind_acc[j])): # ind_acc[j] contains the indices for the relevant DRA pair, to which the state of the BSCC was complying for acceptance
                            Check_Tag = 0
                            Keep_Going = 0
                            for w in range(len(ind_non_acc)): # Same thing for non accepting
                                if ind_acc[j][l] in ind_non_acc[w]: # Checks if accepting index is in list of non_accepting indices ?? YES
                                    Check_Tag = 1 # Means that the current index in the accepting states doesn't make the BSCC accepting
                                    if len(Non_Accept_Remove[ind_acc[j][l]]) == 0: # If the list of non-accepting states that must be removed is empty, then
                                        Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w]) # add to list of Non-Accepting states to be removed in the BSCC.
                                        Keep_Going = 1 # Stops checking the state, because we know it has to be removed
                                    elif Keep_Going == 0: # when there are no states to be removed in the BSCC
                                        break
                            if Check_Tag == 0: 
                                Accept.append(acc_states[j]) # add the list of accepting states
                                Acc_Tag = 1  


                if Acc_Tag == 1: #If the BSCC is accepting
                   
                    
                    
                    #Compute the policy that maximizes the lower bound probability of reaching unmatched accepting states 
                    SCC.sort()
                    Accept.sort()
                    Permanent_Policy_BSCC = np.zeros(len(SCC))
                    for i in range(len(Accept)):   #Converts indices of SCC for reachability computation                       
                        Act = Al_Act_Perm[Ind[Accept[i]]][0]
                        Accept[i] = SCC.index(Accept[i])
                        Permanent_Policy_BSCC[i] = Act
                        
                                      
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    
                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Al_Act_Perm[Ind[SCC[i]]])):
                                if IA1_u_BSCC[Al_Act_Perm[Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Al_Act_Perm[Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Al_Act_Perm[Indices[i]])  
                    
                    
                                     
                    (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions) # Minimizes Upper Bound
                    Bad_States = []
                    for i in range(len(Dummy_Low_Bounds)):
                        if Dummy_Low_Bounds[i] == 0: #If some states have a lower-bound zero of reaching an accepting state inside the BSCC, then it means there will always be a scenario where those states form a non-accepting BSCC, and therefore cannot be part of a permanent BSCC
                            Bad_States.append(SCC[i])


                    if len(Bad_States) == 0:                    
                        if SCC not in List_Permanent_Acc_BSCC:
                            List_Permanent_Acc_BSCC.append([])
                            for i in range(len(SCC)):
                                Permanent_Policy[Ind[SCC[i]]] = Permanent_Policy_BSCC[i]
                                Potential_Policy[Ind[SCC[i]]] = Permanent_Policy[Ind[SCC[i]]] #To avoid bridge states during refinement
                                G_Per_Acc_BSCCs.append(Ind[SCC[i]])
                                Is_In_Permanent_Comp[Ind[SCC[i]]] = 1
                                List_Permanent_Acc_BSCC[-1].append(Ind[SCC[i]])
                    else:                      
                        SCC_New = list(set(SCC) - set(Bad_States)) #Create new set of states without the states to be removed
                        SCC_New.sort()          
                    #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                        if len(SCC_New) != 0: # if some states are left in the SCC
                            SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                            for i in range(len(SCC_New)):
                                for k in range(len(Al_Act_Perm[Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Al_Act_Perm[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
#                                    for k in range(len(SCC)):
#                                        SCC[k] = Ind.index(SCC[k])
                                
                            C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label              
                            for j in range(len(C_new)):
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                C.append(C_new[j]) # put them in the front.
                                SCC_Status.append(1)
                         
                        SCC_New = list(Bad_States) 
                        SCC_New.sort()          
                    #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                        if len(SCC_New) != 0: # if some states are left in the SCC
                            SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                            for i in range(len(SCC_New)):
                                for k in range(len(Al_Act_Perm[Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Al_Act_Perm[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
#                                    for k in range(len(SCC)):
#                                        SCC[k] = Ind.index(SCC[k])
                                
                            C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label              
                            for j in range(len(C_new)):
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                C.append(C_new[j]) # put them in the front.
                                SCC_Status.append(1)


                else: #If the BSCC is not accepting, then we need to remove the non-accepting states prevening it from being accepting                                        
                    #We now have BSCCs which share the same states. Need to figure out what to do with respect to the actions
                    Check_Tag3 = 0
                    
                    
                    Count_Duplicates = 0                   
                    for j in range(len(Non_Accept_Remove)):
                        if len(Non_Accept_Remove[j]) != 0:
                            Count_Duplicates += 1
                                                               
                            
                    for j in range(len(Non_Accept_Remove)): #Loop through the set of states to remove
                        if len(Non_Accept_Remove[j]) != 0: 
                            if Check_Tag3 == 0 and Count_Duplicates > 1:
                                Duplicate_Actions = copy.deepcopy(Al_Act_Pot)
                                Check_Tag3 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) #Create new set of states without the states to be removed
                            SCC_New.sort()
                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                #SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Perm[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Perm[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                  
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                                #print C_new                                
                                for j in range(len(C_new)):

                                    if Count_Duplicates > 1:
                                        Status3_Act.append(Duplicate_Actions)
                                        Which_Status3_BSCC.append(Number_Duplicates3)
                                        List_Status3_Found.append([])#Will contain all permanent BSCCs found after duplication
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states                                       
                                    C.append(C_new[j]) # put them in the front.
                                    
                                    if Count_Duplicates > 1:
                                        
                                        SCC_Status.append(3)
                                    else:
                                        SCC_Status.append(1)
#                   
                    if Check_Tag3 == 1 and Count_Duplicates > 1:
                        Number_Duplicates3 += 1
                    
                        
            elif SCC_Status[m] == 2:
                
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                Inevitable = 1 #Tag to see if BSCC is an inevitable BSCC
                       
                
                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: # accepting, then add it as accepting and vice versa
                        acc_states.append(SCC[j])
                        indices = [] # Establish index list
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): # loop through to find which accepting pair-conditions are sufficed by the given state
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) # Add the respective absolute index/label for the states
                        ind_acc.append(indices) # Add the accumulated list of indices with respect to the BSCC

                    if Is_NAcc[Ind[SCC[j]]] == 1: # do the same thing for non-accepting states in the BSCC
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                    for i in range(len(Status2_Act[Counter_Status2][Ind[SCC[j]]])): # For a given list of allowable actions for a given state in the SCC
                        #print 'i = '; print i
                        if Is_Bridge_State[Status2_Act[Counter_Status2][Ind[SCC[j]]][i]][Ind[SCC[j]]] == 1: # Is the given state in the SCC a bridge state? If yes,                                                                  
                            Diff_List = np.setdiff1d(Reachable_States[Status2_Act[Counter_Status2][Ind[SCC[j]]][i]][Ind[SCC[j]]], Orig_SCC) # Subtract all states within SCC from the list of reachable states from the current state. Then-
                            if len(Diff_List) != 0: # is there anything that remains?
                                Inevitable = 0  # If so, then inevitability/permanence = 0, which means that the current SCC is no bueno in the permanency test
                            Bridge_States.append(Ind[SCC[j]]) # Then bridge_states list is constructed for the given SCC.
                            #not using this Bridge_States variable at the moment

                Acc_Tag = 0
                Accept = [] #Contains unmatched accepting states
                                                   
                if len(non_acc_states) == 0: # If there are no non-accepting states,
                    Acc_Tag = 1 # then activate tag for accepting BSCC.
                    for j in range(len(acc_states)): # Subsequently, add to the list of accepting BSCC.
                        Accept.append(acc_states[j])
                
                else:
                                               
                    Non_Accept_Remove = [[] for x in range(len(Acc))] # Contains all non-accepting states which prevent the bscc to be accepting for all pairs                        
                    for j in range(len(ind_acc)): # Recall that ind_acc contains the accumulated list of indices of the pairs with respect to the BSCC
                        for l in range(len(ind_acc[j])): # ind_acc[j] contains the indices for the relevant DRA pair, to which the state of the BSCC was complying for acceptance
                            Check_Tag = 0
                            Keep_Going = 0
                            for w in range(len(ind_non_acc)): # Same thing for non accepting
                                if ind_acc[j][l] in ind_non_acc[w]: # Checks if accepting index is in list of non_accepting indices ?? YES
                                    Check_Tag = 1 # Means that the current index in the accepting states doesn't make the BSCC accepting
                                    if len(Non_Accept_Remove[ind_acc[j][l]]) == 0: # If the list of non-accepting states that must be removed is empty, then
                                        Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w]) # add to list of Non-Accepting states to be removed in the BSCC.
                                        Keep_Going = 1 # Stops checking the state, because we know it has to be removed
                                    elif Keep_Going == 0: # when there are no states to be removed in the BSCC
                                        break
                            if Check_Tag == 0: 
                                Accept.append(acc_states[j]) # add the list of accepting states
                                Acc_Tag = 1  
                    #print Acc_Tag
                

                if Acc_Tag == 1: #If the potential greatest BSCC is accepting
                    #Compute the policy that maximizes the lower bound probability of reaching unmatched accepting states 
                    
                    Has_Found_BSCC_Status_2[Which_Status2_BSCC[Counter_Status2]] = 1
                    SCC.sort()
                    Accept.sort()
                    Potential_Policy_BSCC = np.zeros(len(SCC)) 
                    for i in range(len(Accept)):   #Converts indices of SCC for reachability computation
                        Act = Al_Act_Pot[Ind[Accept[i]]][0]
                        Accept[i] = SCC.index(Accept[i])
                        Potential_Policy_BSCC[i] = Act
                                      
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    

                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Status2_Act[Counter_Status2][Ind[SCC[i]]])):
                                if IA1_u_BSCC[Status2_Act[Counter_Status2][Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Status2_Act[Counter_Status2][Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Status2_Act[Counter_Status2][Indices[i]])  
                    
                    # Computes the optimal action to maximize the upper-bound   
                    (Dummy_Reach, Dummy_Upp_Bounds, Dummy_Chain, Potential_Policy_BSCC, Dum) = Maximize_Upper_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Potential_Policy_BSCC, BSCC_Allowed_Actions) # Minimizes Upper Bound
                    for i in range(len(SCC)):
                        Potential_Policy[Ind[SCC[i]]] = Potential_Policy_BSCC[i]
                        List_Found_BSCC_Status_2[Which_Status2_BSCC[Counter_Status2]].append(Ind[SCC[i]])
                         
                    C.append(Original_SCC_Status_2[Which_Status2_BSCC[Counter_Status2]]) #Feeding the original SCC for Permanence check
                    
                    SCC_Status.append(1)                

                else: #If it is not accepting, then we need to remove the non-accepting states prevening it from being accepting                    
                    
                    
                    Check_Tag2 = 0
                    #We now have BSCCs which share the same states. Need to figure out what to do with respect to the actions
                    
                    for j in range(len(Non_Accept_Remove)): #Loop through the set of states to remove
                        if len(Non_Accept_Remove[j]) != 0:
                            if Check_Tag2 == 0:
                                Duplicate_Actions = copy.deepcopy(Status2_Act[Counter_Status2])
                                Check_Tag2 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) #Create new set of states without the states to be removed
                            SCC_New.sort()

                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Al_Act_Pot[Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Al_Act_Pot[Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
#                                    for k in range(len(SCC)):
#                                        SCC[k] = Ind.index(SCC[k])
                                    
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                                #print C_new
                                
                                for j in range(len(C_new)):
                                    Status2_Act.append(Duplicate_Actions)
                                    Which_Status2_BSCC.append(Which_Status2_BSCC[Counter_Status2])                                   
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    
                                    C.append(C_new[j]) # put them in the front.
                                    SCC_Status.append(2)
                        

                                                                            
                Counter_Status2 +=1


            elif SCC_Status[m] == 3:  

                
                ind_acc = [] #Contains the Automaton indices of accepting states in BSCC
                acc_states = [] #Contains the Accepting States in BSCC
                ind_non_acc = [] #Contains the Automaton indices of non-accepting states in BSCC
                non_acc_states = [] #Contans the non Accepting States in BSCC
                
                for j in range(len(SCC)):
                                       
                    if Is_Acc[Ind[SCC[j]]] == 1: # accepting, then add it as accepting and vice versa
                        acc_states.append(SCC[j])
                        indices = [] # Establish index list
                        for n in range(len(Wh_Acc_Pair[Ind[SCC[j]]])): # loop through to find which accepting pair-conditions are sufficed by the given state
                            indices.append(Wh_Acc_Pair[Ind[SCC[j]]][n]) # Add the respective absolute index/label for the states
                        ind_acc.append(indices) # Add the accumulated list of indices with respect to the BSCC

                    if Is_NAcc[Ind[SCC[j]]] == 1: # do the same thing for non-accepting states in the BSCC
                        non_acc_states.append(SCC[j])
                        indices = []
                        for n in range(len(Wh_NAcc_Pair[Ind[SCC[j]]])):
                            indices.append(Wh_NAcc_Pair[Ind[SCC[j]]][n])
                        ind_non_acc.append(indices)                         
                          
                Acc_Tag = 0
                Accept = [] #Contains unmatched accepting states
                                                    
                if len(non_acc_states) == 0: # If there are no non-accepting states,
                    Acc_Tag = 1 # then activate tag for accepting BSCC.
                    for j in range(len(acc_states)): # Subsequently, add to the list of accepting BSCC.
                        Accept.append(acc_states[j])
                
                else:                                        
                  
                    Non_Accept_Remove = [[] for x in range(len(Acc))] # Contains all non-accepting states which prevent the bscc to be accepting for all pairs                        
                    for j in range(len(ind_acc)): # Recall that ind_acc contains the accumulated list of indices of the pairs with respect to the BSCC
                        for l in range(len(ind_acc[j])): # ind_acc[j] contains the indices for the relevant DRA pair, to which the state of the BSCC was complying for acceptance
                            Check_Tag = 0
                            Keep_Going = 0
                            for w in range(len(ind_non_acc)): # Same thing for non accepting
                                if ind_acc[j][l] in ind_non_acc[w]: # Checks if accepting index is in list of non_accepting indices ?? YES
                                    Check_Tag = 1 # Means that the current index in the accepting states doesn't make the BSCC accepting
                                    if len(Non_Accept_Remove[ind_acc[j][l]]) == 0: # If the list of non-accepting states that must be removed is empty, then
                                        Non_Accept_Remove[ind_acc[j][l]].append(non_acc_states[w]) # add to list of Non-Accepting states to be removed in the BSCC.
                                        Keep_Going = 1 # Stops checking the state, because we know it has to be removed
                                    elif Keep_Going == 0: # when there are no states to be removed in the BSCC
                                        break
                            if Check_Tag == 0: 
                                Accept.append(acc_states[j]) # add the list of accepting states
                                Acc_Tag = 1  


                if Acc_Tag == 1: #If the BSCC is accepting                
                
                    #Compute the policy that maximizes the lower bound probability of reaching unmatched accepting states 
                    SCC.sort()
                    Accept.sort()
                    Permanent_Policy_BSCC = np.zeros(len(SCC))  
                    for i in range(len(Accept)):   #Converts indices of SCC for reachability computation
                        Act = Al_Act_Perm[Ind[Accept[i]]][0]
                        Accept[i] = SCC.index(Accept[i])
                        Permanent_Policy_BSCC[i] = Act
                          
                                    
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for i in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for j in range(len(SCC)):
                            if i == 0:
                                Indices.append(Ind[SCC[j]])
                            BSCC_Reachable_States[-1].append([])                    
                    
                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    
                    for i in range(len(SCC)):
                        for j in range(len(SCC)):
                            for l in range(len(Status3_Act[Counter_Status3][Ind[SCC[i]]])):
                                if IA1_u_BSCC[Status3_Act[Counter_Status3][Ind[SCC[i]]][l], i,j] > 0:
                                    BSCC_Reachable_States[Status3_Act[Counter_Status3][Ind[SCC[i]]][l]][i].append(j)
                    
                    
                    BSCC_Allowed_Actions = []
                    for i in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Status3_Act[Counter_Status3][Indices[i]])  
                    
                                     
                    (Dummy_Reach, Dummy_Low_Bounds, Dummy_Chain, Permanent_Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Accept, 0, 0, BSCC_Reachable_States, [], Permanent_Policy_BSCC, BSCC_Allowed_Actions) # Minimizes Upper Bound
                    Bad_States = []
                    for i in range(len(Accept)):
                        Permanent_Policy_BSCC[Accept[i]] = Status3_Act[Counter_Status3][Ind[SCC[Accept[i]]]][0]
                        
                    for i in range(len(Dummy_Low_Bounds)):
                        if Dummy_Low_Bounds[i] == 0: #If some states have a lower-bound zero of reaching an accepting state inside the BSCC, then it means there will always be a scenario where those states form a non-accepting BSCC, and therefore cannot be part of a permanent BSCC
                            Bad_States.append(SCC[i])
                    
                    if len(Bad_States) == 0:
                        Existing_Lists = []
                        for i in range(len(List_Status3_Found[Which_Status3_BSCC[Counter_Status3]])):
                            Existing_Lists.append(List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][i][0])
                        
                        if SCC not in Existing_Lists:
                            #List_Permanent_Acc_BSCC.append([]) Will be taken care of later
                            List_Status3_Found[Which_Status3_BSCC[Counter_Status3]].append([])
                            List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1].append([]) #First list will contain the states, second list the actions
                            List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1].append([]) #First list will contain the states, second list the actions                        
                            for i in range(len(SCC)):                            
                                List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1][0].append(Ind[SCC[i]])
                                List_Status3_Found[Which_Status3_BSCC[Counter_Status3]][-1][1].append(Permanent_Policy_BSCC[i])
                                #G_Per_Acc_BSCCs.append(Ind[SCC[i]]) Will be taken care of later as well
                                Is_In_Permanent_Comp[Ind[SCC[i]]] = 1
                                #List_Permanent_Acc_BSCC[-1].append(Ind[SCC[i]]) willl be taken care of later
                    else:                      
                        SCC_New = list(set(SCC) - set(Bad_States)) #Create new set of states without the states to be removed
                        SCC_New.sort()          
                    #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                        if len(SCC_New) != 0: # if some states are left in the SCC
                            Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                            SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                            for i in range(len(SCC_New)):
                                for k in range(len(Status3_Act[Counter_Status3][Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Status3_Act[Counter_Status3][Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
#                                    for k in range(len(SCC)):
#                                        SCC[k] = Ind.index(SCC[k])
                                
                            C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label              
                            for j in range(len(C_new)):
                                Status3_Act.append(Duplicate_Actions)
                                Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])                                
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                C.append(C_new[j]) # put them in the front.
                                SCC_Status.append(3)
  

                        SCC_New = list(Bad_States) #Create new set of states without the states to be removed
                        SCC_New.sort()          
                    #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                        if len(SCC_New) != 0: # if some states are left in the SCC
                            Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                            SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                            New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                            for i in range(len(SCC_New)):
                                for k in range(len(Status3_Act[Counter_Status3][Ind[SCC_New[i]]])):
                                    for j in range(len(SCC_New)):
                                        if IA1_u[Status3_Act[Counter_Status3][Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                            New_G[i,j] = 1
#                                    for k in range(len(SCC)):
#                                        SCC[k] = Ind.index(SCC[k])
                                
                            C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label              
                            for j in range(len(C_new)):
                                Status3_Act.append(Duplicate_Actions)
                                Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])                                
                                for k in range(len(C_new[j])):
                                    C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                C.append(C_new[j]) # put them in the front.
                                SCC_Status.append(3)                       

                else: #If the BSCC is not accepting, then we need to remove the non-accepting states prevening it from being accepting                                        
                    #We now have BSCCs which share the same states. Need to figure out what to do with respect to the actions

                    
                    Check_Tag3 = 0 
                    for j in range(len(Non_Accept_Remove)): #Loop through the set of states to remove
                        if len(Non_Accept_Remove[j]) != 0: 
                            if Check_Tag3 == 0:
                                Duplicate_Actions = copy.deepcopy(Status3_Act[Counter_Status3])
                                Check_Tag3 = 1
                            SCC_New = list(set(SCC) - set(Non_Accept_Remove[j])) #Create new set of states without the states to be removed
                            SCC_New.sort()
                        #Could be optimized, convert back non-leaky states to indices of reduced graph                                                        
                            if len(SCC_New) != 0: # if some states are left in the SCC
                                #SCC = sorted(SCC, key=int)  # Sort the states in the SCC              
                                New_G = np.zeros((len(SCC_New), len(SCC_New)))# Create new graph
                                for i in range(len(SCC_New)):
                                    for k in range(len(Status3_Act[Counter_Status3][Ind[SCC_New[i]]])):
                                        for j in range(len(SCC_New)):
                                            if IA1_u[Status3_Act[Counter_Status3][Ind[SCC_New[i]]][k], Ind[SCC_New[i]], Ind[SCC_New[j]]] > 0:
                                                New_G[i,j] = 1
                                  
                                C_new, n_new = SSCC(New_G) # C_new corresponds to SCC state label
                                #print C_new                                
                                for j in range(len(C_new)):
                                    Status3_Act.append(Duplicate_Actions)
                                    Which_Status3_BSCC.append(Which_Status3_BSCC[Counter_Status3])
                                    for k in range(len(C_new[j])):
                                        C_new[j][k] = SCC_New[C_new[j][k]] # Converting C_new to SCC label of the states
                                    C.append(C_new[j]) # put them in the front.
                                    SCC_Status.append(3)
#                   
                
                Counter_Status3 +=1                 
                    
        m +=1
        if m == len(C): tag = 1
      
    for i in range(len(Maybe_Permanent)): #Looping through the potential BSCC to see which ones turned out to be permanent
        List_Potential_States = []
        List_Bridge_States = []
        BSCC_Converted_Indices = []
        for j in range(len(Maybe_Permanent[i])):            
            if Is_In_Permanent_Comp[Ind[Maybe_Permanent[i][j]]] == 0: #If the state turns out not to be a permanent component
                BSCC_Converted_Indices.append(Ind[Maybe_Permanent[i][j]])
                List_Potential_States.append(Ind[Maybe_Permanent[i][j]])
                Which_Potential_Acc_BSCC[Ind[Maybe_Permanent[i][j]]] = len(List_G_Pot)
                Is_In_Potential_Acc_BSCC[Ind[Maybe_Permanent[i][j]]] = 1
                if Is_Bridge_State[Potential_Policy[Ind[Maybe_Permanent[i][j]]]][Ind[Maybe_Permanent[i][j]]] == 1:
                    List_Bridge_States.append(Ind[Maybe_Permanent[i][j]])
        if len(List_Potential_States) != 0: #That is, the BSCC is not entirely Permanent 
            List_G_Pot.append(BSCC_Converted_Indices)
            Bridge_Potential_Accepting.append(List_Bridge_States)


            
    for i in range(Number_Duplicates2): #Taking care of the states who were in duplicate SCCs when searching potential BSCCs
        print 'Num_Dup2'
        print 'Original SCC'
        if Has_Found_BSCC_Status_2[i] == 1: #We now know that the original SCC (which was not accepting due to some non-accepting states) can potentially be made accepting        
            Non_Permanent_States = []
            for j in range(len(Original_SCC_Status_2[i])):
                G_Pot_Acc_BSCCs.append(Ind[Original_SCC_Status_2[i][j]])
                if Is_In_Permanent_Comp[Ind[Original_SCC_Status_2[i][j]]] == 0:
                    Non_Permanent_States.append(Ind[Original_SCC_Status_2[i][j]])
                    Is_In_Potential_Acc_BSCC[Ind[Original_SCC_Status_2[i][j]]] = 1
                    Which_Potential_Acc_BSCC[Ind[Original_SCC_Status_2[i][j]]] = len(List_G_Pot)
            List_G_Pot.append(Non_Permanent_States) 
            Remaining_States = list(set(Original_SCC_Status_2[i]) - set(List_Found_BSCC_Status_2[i]))
            for j in range(len(Remaining_States)):
                Potential_Policy[Ind[Remaining_States[i]]] = Al_Act_Pot[Ind[Remaining_States[i]]][0] #Any action that could generate the duplicated BSCC works                
            List_Bridge_States = []
            for j in range(len(Non_Permanent_States)):          
                if Is_Bridge_State[Potential_Policy[Non_Permanent_States[j]]][Non_Permanent_States[j]] == 1: 
                    List_Bridge_States.append(Non_Permanent_States[j])
            Bridge_Potential_Accepting.append(List_Bridge_States) 


                       
    for i in range(Number_Duplicates3): #Taking care of the states who were in duplicate SCCs when searching permanent BSCCs
        if (len(List_Status3_Found[i])!= 0):
            Graph = np.zeros((len(List_Status3_Found[i]),len(List_Status3_Found[i])))
            for j in range(len(List_Status3_Found[i])): #Check connectedness of all states
                Graph[j,j] = 1
                for k in range(j+1, len(List_Status3_Found[i])):
                                     
                    if (set(List_Status3_Found[i][j][0]).intersection(set(List_Status3_Found[i][k][0]))) != 0:
                        Graph[j,k] = 1
                        Graph[k,j] = 1
            
            

            Comp_Graph = csr_matrix(Graph)
            Num_Comp, labels =  connected_components(csgraph=Comp_Graph, directed=False, return_labels=True)            
            C = [[] for x in range(Num_Comp)]
    
            for k in range(len(labels)):
                C[labels[i]].append(i)
                
            for k in range(len(C)):
        
                Component = []
                if len(C[k]) == 1:   #if the component is not connected to any other
                    for l in range(len(List_Status3_Found[i][C[k][0]][0])):

                        Permanent_Policy[Ind[List_Status3_Found[i][C[k][0]][0][l]]] = List_Status3_Found[i][C[k][0]][1][l]
                        Component.append(Ind[List_Status3_Found[i][C[k][0]][0][l]])
                    
                    List_Permanent_Acc_BSCC.append(Component)
  
                      
                else:
                                        
                    States_To_Reach = []
                    for l in range(len(List_Status3_Found[i][C[k][0]][0])):
                        Permanent_Policy[Ind[List_Status3_Found[i][C[k][0]][0][l]]] = List_Status3_Found[i][C[k][0]][1][l]
                        States_To_Reach.append(Ind[List_Status3_Found[i][C[k][0]][0][l]])
                        Component.append(Ind[List_Status3_Found[i][C[k][0]][0][l]])
                   
                    States_For_Reachability = []
                    for l in range(1, len(C[k])):
                        for m in range(len(List_Status3_Found[i][C[k][l]][0])):
                            if Ind[List_Status3_Found[i][C[k][l]][0][m]] not in States_To_Reach:
                                States_For_Reachability.append(Ind[List_Status3_Found[i][C[k][l]][0][m]])
                                Component.append(Ind[List_Status3_Found[i][C[k][l]][0][m]])


                    Component.sort()
                    Policy_BSCC = np.zeros(len(Component))                   
                    # Creating the list of reachable states etc.  (very computationally inefficient)
                    BSCC_Reachable_States = []
                    Indices = []
                    for y in range(IA1_l.shape[0]):
                        BSCC_Reachable_States.append([])
                        for x in range(len(Component)):
                            if y == 0:
                                Indices.append(Component[x])
                            BSCC_Reachable_States[-1].append([])
                            
                    
                    Target = []
                    for y in range(len(States_To_Reach)):
                        Target.append(Indices.index(States_To_Reach[y]))

                    IA1_l_BSCC = np.array(IA1_l[:,Indices,:])
                    IA1_l_BSCC = np.array(IA1_l_BSCC[:,:,Indices])
                    IA1_u_BSCC = np.array(IA1_u[:,Indices,:])
                    IA1_u_BSCC = np.array(IA1_u_BSCC[:,:,Indices])
                    

                    for y in range(len(Component)):
                        for x in range(len(Component)):
                            for l in range(len(Al_Act_Perm[Component[y]])):
                                if IA1_u_BSCC[Al_Act_Perm[Component[y]][l], y,x] > 0:
                                    BSCC_Reachable_States[Al_Act_Perm[Component[y]][l]][y].append(x)                                        
                    BSCC_Allowed_Actions = []
                    for y in range(len(Indices)):
                        BSCC_Allowed_Actions.append(Al_Act_Perm[Indices[y]])  
                    
                    # Computes the optimal action to maximize the upper-bound   
                    (Dummy_Reach, Dummy_Upp_Bounds, Dummy_Chain, Policy_BSCC, Dum) = Maximize_Lower_Bound_Reachability(IA1_l_BSCC, IA1_u_BSCC, Target, 0, 0, BSCC_Reachable_States, [], Policy_BSCC, BSCC_Allowed_Actions)

                    for y in range(len(States_For_Reachability)):
                        Permanent_Policy[States_For_Reachability[y]] = Policy_BSCC[Component.index(States_For_Reachability[y])]
                    
                    List_Permanent_Acc_BSCC.append(Component)
                    
    print List_Permanent_Acc_BSCC                                     
    print List_G_Pot                    
                         


                       
    return G_Pot_Acc_BSCCs, G_Per_Acc_BSCCs, Potential_Policy, Permanent_Policy, Al_Act_Pot, Al_Act_Perm, first, Is_In_Permanent_Comp, List_Permanent_Acc_BSCC, List_G_Pot, Which_Potential_Acc_BSCC, Is_In_Potential_Acc_BSCC, Bridge_Potential_Accepting




def Maximize_Lower_Bound_Reachability(IA_l, IA_u, Q1, Num_States, Automata_size, Reach, Init, Optimal_Policy, Actions):
    
    #Q1 is the target state
#    Optimal_Policy = np.zeros(IA_l.shape[1])
#    Optimal_Policy = Optimal_Policy.astype(int)
    
    Ascending_Order = []
    Index_Vector = np.zeros((IA_l.shape[1],1))
    Is_In_Q1 = np.zeros((IA_l.shape[1]))
    
    for k in range(IA_l.shape[1]):                               
        if k in Q1:            
            Index_Vector[k,0] = 1.0
            Ascending_Order.append(k)
            Is_In_Q1[k] = 1
           
        else:            
            Index_Vector[k,0] = 0.0
            Ascending_Order.insert(0,k)

    d = {k:v for v,k in enumerate(Ascending_Order)} 
    Sort_Reach = []

    for i in range(len(Reach)):
        Sort_Reach.append([])
        for j in range(IA_l.shape[1]):
            Sort_Reach[-1].append([])
    
    for j in range(IA_l.shape[1]):        
        if Is_In_Q1[j] == 0:
            for k in range(len(Actions[j])):
                Reach[Actions[j][k]][j].sort(key=d.get)
                Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
        else:
            continue
               
    Phi_Min = Phi_Synthesis_Max_Lower(IA_l, IA_u, Ascending_Order, Q1, Reach, Sort_Reach, Actions)
    Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Min), Index_Vector)
        
    for i in range(IA_l.shape[1]):
        if Is_In_Q1[i] == 1: continue
        List_Values = []
        for k in range(len(Actions[i])):
            List_Values.append(IA_l.shape[0]*i+Actions[i][k])            
        Values = Steps_Low[List_Values]
        Index_Vector[i,0] = np.amax(Values)
        Optimal_Policy[i] = Actions[i][np.argmax(Values)]
    

    for i in range(len(Q1)):    
        Index_Vector[Q1[i],0] = 1.0

    Success_Intervals = []       
    for i in range(IA_l.shape[1]):       
#        Success_Intervals[i].append(Steps_Low[i][0])
        Success_Intervals.append(Index_Vector[i,0]) 
        
     
    Terminate_Check = 0
    Convergence_threshold = 0.01
    Previous_Max_Difference = 1
           
    
    while Terminate_Check == 0:
                   
        Previous_List = copy.copy(Ascending_Order)
               
        for i in range(len(Q1)):
            Success_Intervals[Q1[i]] = 1.0
       
        Ascending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Ascending_Order = list(Ascending_Order[(Success_Array).argsort()]) 
        
        d = {k:v for v,k in enumerate(Ascending_Order)} 
        Sort_Reach = []

        for i in range(len(Reach)):
            Sort_Reach.append([])
            for j in range(IA_l.shape[1]):
                Sort_Reach[-1].append([])
        
        for j in range(IA_l.shape[1]):        
            if Is_In_Q1[j] == 0:
                for k in range(len(Actions[j])):
                    Reach[Actions[j][k]][j].sort(key=d.get)
                    Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
            else:
                continue
        
        if Previous_List != Ascending_Order:
            Phi_Min = Phi_Synthesis_Max_Lower(IA_l, IA_u, Ascending_Order, Q1, Reach, Sort_Reach, Actions)
#            print 'New Phi'
        
        #Steps_Low = np.dot(Phi_Min, Steps_Low)

        Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Min), Index_Vector)
    
        List_Values = list([])
        Bounds_All_Act = list([])
        for i in range(IA_l.shape[1]):
            if Is_In_Q1[i] == 1: 
                Bounds_All_Act.append(list([]))
                for j in range(len(Actions[i])):
                    Bounds_All_Act[-1].append(1.0)
                continue
            List_Values.append([])
            for k in range(len(Actions[i])):
                List_Values[-1].append(IA_l.shape[0]*i+Actions[i][k])
            Values = list(Steps_Low[List_Values[-1]])
            Bounds_All_Act.append(Values)
            Index_Vector[i,0] = np.amax(Values)
            Optimal_Policy[i] = Actions[i][np.argmax(Values)]    
            
        
        for i in range(len(Q1)):    
            Index_Vector[Q1[i],0] = 1.0
                         
        Max_Difference = 0
                       
        for i in range(IA_l.shape[1]):
                                  
            Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Index_Vector[i,0]))        
            Success_Intervals[i] = Index_Vector[i,0]
        
        #print Max_Difference
            
        if Max_Difference < Convergence_threshold:              
            Terminate_Check = 1    
    
    Bounds = []
    Prod_Bounds = []
    
    Indices = [int(i*IA_l.shape[0]+Optimal_Policy[i]) for i in range(len(Optimal_Policy))]
    Phi_Min = np.array(Phi_Min[Indices,:])
    
    for i in range(Num_States):
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
    
    for i in range(len(Success_Intervals)):
        Prod_Bounds.append(Success_Intervals[i])
        
    return (Bounds, Prod_Bounds, Phi_Min, Optimal_Policy, Bounds_All_Act)











def Maximize_Upper_Bound_Reachability(IA_l, IA_u, Q1, Num_States,Automata_size, Reach, Init, Optimal_Policy, Actions):
    
    #Q1 is the target state
#    Optimal_Policy = np.zeros(IA_l.shape[1])
#    Optimal_Policy = Optimal_Policy.astype(int)
    
    Descending_Order = []
    Index_Vector = np.zeros((IA_l.shape[1],1)) 
    Is_In_Q1 = np.zeros((IA_l.shape[1]))
    
    for k in range(IA_l.shape[1]):                               
        if k in Q1:            
            Index_Vector[k,0] = 1.0
            Descending_Order.insert(0,k)
            Is_In_Q1[k] = 1
           
        else:            
            Index_Vector[k,0] = 0.0
            Descending_Order.append(k)

    d = {k:v for v,k in enumerate(Descending_Order)} 
    Sort_Reach = []

    for i in range(len(Reach)):
        Sort_Reach.append([])
        for j in range(IA_l.shape[1]):
            Sort_Reach[-1].append([])
    
    for j in range(IA_l.shape[1]):        
        if Is_In_Q1[j] == 0:
            for k in range(len(Actions[j])):
                Reach[Actions[j][k]][j].sort(key=d.get)
                Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
        else:
            continue
                
    Phi_Max = Phi_Synthesis_Max_Upper(IA_l, IA_u, Descending_Order, Q1, Reach, Sort_Reach, Actions)   
    Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Max), Index_Vector)
        
    for i in range(IA_l.shape[1]):
        if Is_In_Q1[i] == 1: continue
        List_Values = []
        for k in range(len(Actions[i])):
            List_Values.append(IA_l.shape[0]*i+Actions[i][k])
        Values = list(Steps_Low[List_Values])
        Index_Vector[i,0] = np.amax(Values)
        Optimal_Policy[i] = Actions[i][np.argmax(Values)]

    for i in range(len(Q1)):    
        Index_Vector[Q1[i],0] = 1.0

    Success_Intervals = list([])    
 
    for i in range(IA_l.shape[1]):       
#        Success_Intervals[i].append(Steps_Low[i][0])
        Success_Intervals.append(Index_Vector[i,0]) 


    Terminate_Check = 0
    Convergence_threshold = 0.01
    Previous_Max_Difference = 1
    count = 0
           

    while Terminate_Check == 0:
        
        count += 1
                   
        Previous_List = copy.copy(Descending_Order)
#        print Previous_List
               
        for i in range(len(Q1)):
            Success_Intervals[Q1[i]] = 1.0
       
        Descending_Order = np.array(range(len(Success_Intervals)))
        Success_Array = np.array(Success_Intervals)
        Descending_Order = list(Descending_Order[(-Success_Array).argsort()]) 


        
        d = {k:v for v,k in enumerate(Descending_Order)} 
        Sort_Reach = list([])

        for i in range(len(Reach)):
            Sort_Reach.append([])
            for j in range(IA_l.shape[1]):
                Sort_Reach[-1].append([])
        
        for j in range(IA_l.shape[1]):        
            if Is_In_Q1[j] == 0:
                for k in range(len(Actions[j])):
                    Reach[Actions[j][k]][j].sort(key=d.get)
                    Sort_Reach[Actions[j][k]][j] = list(Reach[Actions[j][k]][j])
            else:
                continue
        
        if Previous_List != Descending_Order:
            Phi_Max = Phi_Synthesis_Max_Upper(IA_l, IA_u, Descending_Order, Q1, Reach, Sort_Reach, Actions)
#            print 'New Phi'
        
        
        
#        Steps_Low = np.dot(Phi_Max, Index_Vector[:,0])

        Steps_Low = sparse.csr_matrix.dot(sparse.csr_matrix(Phi_Max), Index_Vector[:,0])

        List_Values = list([])
        Bounds_All_Act = list([])
    
        for i in range(IA_l.shape[1]):
            if Is_In_Q1[i] == 1:
                Bounds_All_Act.append(list([]))
                for j in range(len(Actions[i])):
                    Bounds_All_Act[-1].append(1.0)
                continue                
            List_Values.append([])         
            for k in range(len(Actions[i])):
                List_Values[-1].append(IA_l.shape[0]*i+Actions[i][k])    
            Values = list(Steps_Low[List_Values[-1]])  
            Bounds_All_Act.append(Values)
            Index_Vector[i,0] = np.amax(Values)
            Optimal_Policy[i] = Actions[i][np.argmax(Values)] 
        
        for i in range(len(Q1)):    
            Index_Vector[Q1[i],0] = 1.0
                                   
        Max_Difference = 0
        
               
        for i in range(IA_l.shape[1]):                                
            Max_Difference = max(Max_Difference, abs(Success_Intervals[i] - Index_Vector[i,0]))        
            Success_Intervals[i] = Index_Vector[i,0]
         
        #print Max_Difference    
        if Max_Difference < Convergence_threshold:              
            Terminate_Check = 1    
    
    Bounds = []
    Prod_Bounds = []
    
    Indices = [int(i*IA_l.shape[0]+Optimal_Policy[i]) for i in range(len(Optimal_Policy))]
    Phi_Max = np.array(Phi_Max[Indices,:])
    
    for i in range(Num_States):
#        Bounds.append(Success_Intervals[i*Automata_size][0])
        Bounds.append(Success_Intervals[i*Automata_size+Init[i]])
    
    for i in range(len(Success_Intervals)):
#        Prod_Bounds.append(Success_Intervals[i][0])
        Prod_Bounds.append(Success_Intervals[i])
        
    return (Bounds, Prod_Bounds, Phi_Max, Optimal_Policy, Bounds_All_Act)




def Phi_Synthesis_Max_Lower(Lower, Upper, Order_A, q1, Reach, Reach_Sort, Action):
    
    Phi_min = np.zeros((Upper.shape[1]*Upper.shape[0], Upper.shape[1]))

    for j in range(Upper.shape[1]):
        
        if j in q1:
            continue
        else:
    
            for k in range(len(Action[j])):
                Up = Upper[Action[j][k]][j][:]
                Low = Lower[Action[j][k]][j][:]                 
                Sum_1_A = 0.0
                Sum_2_A = sum(Low[Reach[Action[j][k]][j]])    
                Phi_min[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][0]] = min(Low[Reach_Sort[Action[j][k]][j][0]] + 1 - Sum_2_A, Up[Reach_Sort[Action[j][k]][j][0]])  
          
                for i in range(1, len(Reach_Sort[Action[j][k]][j])):
                                 
                    Sum_1_A = Sum_1_A + Phi_min[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i-1]]
                    if Sum_1_A >= 1:
                        break
                    Sum_2_A = Sum_2_A - Low[Reach_Sort[Action[j][k]][j][i-1]]
                    Phi_min[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i]] = min(Low[Reach_Sort[Action[j][k]][j][i]] + 1 - (Sum_1_A+Sum_2_A), Up[Reach_Sort[Action[j][k]][j][i]])                 
    return Phi_min





def Phi_Synthesis_Max_Upper(Lower, Upper, Order_D, q1, Reach, Reach_Sort, Action):
    
    Phi_max = np.zeros((Upper.shape[1]*Upper.shape[0], Upper.shape[1]))
    
    for j in range(Upper.shape[1]):
        
        if j in q1:
            continue
        else:
    
            for k in range(len(Action[j])):

                Up = Upper[Action[j][k]][j][:]
                Low = Lower[Action[j][k]][j][:] 
                Sum_1_D = 0.0
                Sum_2_D = sum(Low[Reach[Action[j][k]][j]])
                Phi_max[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][0]] = min(Low[Reach_Sort[Action[j][k]][j][0]] + 1 - Sum_2_D, Up[Reach_Sort[Action[j][k]][j][0]])  
          
                for i in range(1, len(Reach_Sort[Action[j][k]][j])):
                                 
                    Sum_1_D = Sum_1_D + Phi_max[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i-1]]
                    if Sum_1_D >= 1:
                        break
                    Sum_2_D = Sum_2_D - Low[Reach_Sort[Action[j][k]][j][i-1]]
                    Phi_max[j*Upper.shape[0]+Action[j][k]][Reach_Sort[Action[j][k]][j][i]] = min(Low[Reach_Sort[Action[j][k]][j][i]] + 1 - (Sum_1_D+Sum_2_D), Up[Reach_Sort[Action[j][k]][j][i]])  
               
    return Phi_max


def Bounds_Tightening(Lower_Bound_Matrix, Upper_Bound_Matrix):
    
    
    for j in range(Lower_Bound_Matrix.shape[0]):
        Sum_Low = sum(Lower_Bound_Matrix[j][:])
        Sum_High = sum(Upper_Bound_Matrix[j][:])
        for i in range(Lower_Bound_Matrix.shape[1]):
            Res_Up = 1 - Sum_High - Upper_Bound_Matrix[j][i] - Lower_Bound_Matrix[j][i]
            Res_Down = 1 - Sum_Low -  Lower_Bound_Matrix[j][i] - Upper_Bound_Matrix[j][i]
            Lower_Bound_Matrix[j][i] = Lower_Bound_Matrix[j][i] + max(0, Res_Up)
            Upper_Bound_Matrix[j][i] = Upper_Bound_Matrix[j][i] + min(0, Res_Down)
            Sum_High = Sum_High + Res_Down
            Sum_Low = Sum_Low + Res_Up

    
    return Lower_Bound_Matrix, Upper_Bound_Matrix


def SSCC(graph):
    
    #Search for all Strongly Connected Components in a Graph

    #set of visited vertices
    used = set()
    
    #call first depth-first search
    list_vector = [] #vertices in topological sorted order
    for vertex in range(len(graph)):
       if vertex not in used:
          (list_vector,used) = first_dfs(vertex, graph, used, list_vector)              
    list_vector.reverse()
    
    #preparation for calling second depth-first search
    graph_t = reverse_graph(graph)
    used = set()
    
    #call second depth-first search
    components= []
    list_components = [] #strong-connected components
    scc_quantity = 0 #quantity of strong-connected components 
    for vertex in list_vector:
        if vertex not in used:
            scc_quantity += 1
            list_components = []
            (list_components, used) = second_dfs(vertex, graph_t, list_components, list_vector, used)
#            print(list_components)
            components.append(list_components)
            
#    print(scc_quantity)
    
    return components, scc_quantity



def Raw_Refinement(State, Space): 
       
    New_St = []       
    for i in range(len(State)):

       a1 = Space[State[i]][1][0] - Space[State[i]][0][0]
       a2 = Space[State[i]][1][1] - Space[State[i]][0][1]
    
       if a1 > a2:
                               
           New_St.append([(Space[State[i]][0][0],Space[State[i]][0][1]),((Space[State[i]][1][0] + Space[State[i]][0][0])/2.0,Space[State[i]][1][1])])
           New_St.append([((Space[State[i]][1][0] + Space[State[i]][0][0])/2.0 , Space[State[i]][0][1]),(Space[State[i]][1][0],Space[State[i]][1][1])])
  
       else:
           
           New_St.append([(Space[State[i]][0][0] , (Space[State[i]][1][1]+Space[State[i]][0][1])/2.0),(Space[State[i]][1][0],Space[State[i]][1][1])])
           New_St.append([(Space[State[i]][0][0] , Space[State[i]][0][1]),(Space[State[i]][1][0],(Space[State[i]][1][1]+Space[State[i]][0][1])/2.0)])
   
    return New_St





def first_dfs(vertex, graph, used, list_vector):
    used.add(vertex)
    for v in range(len(graph)):   
        if graph[vertex][v] == 1 and v not in used:   
            (list_vector, used) = first_dfs(v, graph, used, list_vector)
    list_vector.append(vertex)
    return(list_vector, used)

    
def second_dfs(vertex, graph_t, list_components, list_vector, used):
    used.add(vertex)
    for v in list_vector:   
        if graph_t[vertex][v] == 1 and v not in used:   
            (list_components, used) = second_dfs(v, graph_t, list_components, list_vector, used)
    list_components.append(vertex)
    return(list_components, used)
    		                   
    
def reverse_graph(graph):
    graph_t = list(zip(*graph))
    return graph_t



import numpy as np 
from time import time


def gaussian(v,sig):
    #takes a number and returns gaussian function of each value
    return np.exp(-v**2/(2*sig**2))


def mutate(current_strategy,s):
    """
    takes in a player's current strategy, a non-negative
    real number and perturbs the strategy by adding a 
    a random number in range (-s,s) selected uniformly
    the results is then truncated to 0 or 1 to ensure the
    all the outputs are in range
    """

    mutated_strategy = s*(2*np.random.rand(len(current_strategy))-1)
    mutated_strategy = mutated_strategy + current_strategy
    
    #truncating
    for i in range(len(mutated_strategy)):
        if mutated_strategy[i]>1:
            mutated_strategy[i]=1
        elif mutated_strategy[i]<0:
            mutated_strategy[i]=0
    
    return mutated_strategy


def payoff(p,q,game_parameters,discounting_factor):
    #takes two strategies p,q, game parameters,
    #discounting_factor to give the mean payoff

    x_ind = np.array([0,1,2,3])
    y_ind = np.array([0,2,1,3])
    #constructing the transition matrix
    M = np.zeros((4,4))
    for i in range(4):
        M[i,0] = p[x_ind[i]]*q[y_ind[i]]
        M[i,1] = p[x_ind[i]]*(1-q[y_ind[i]])
        M[i,2] = (1-p[x_ind[i]])*q[y_ind[i]]
        M[i,3] = (1-p[x_ind[i]])*(1-q[y_ind[i]])
    
    #initial distribution
    initial_distribution = np.array([p[4]*q[4],p[4]*(1-q[4]),(1-p[4])*q[4],(1-p[4])*(1-q[4])])

    #final distribution of action
    mat = np.linalg.inv(np.identity(4) - discounting_factor*M)
    action_distribution = (1-discounting_factor)*(np.matmul(initial_distribution,mat))

    #final payoffs:
    pix = np.dot(action_distribution,game_parameters[x_ind])
    piy = np.dot(action_distribution,game_parameters[y_ind])

    return pix,piy


def sample_run(learning_rule_x,learning_rule_y,p_initial,q_initial,game_parameters,discounting_factor,s,sig,convergence_threshold,error_threshold):
    """
    sample_run(learning_rule_x, learning_rule_x, p_initial, q_initial,
    game_parameters, discounting_factor, s, sig, convergence_threshold,
    error_threshold) takes as input learning rules and initial strategies
    for x and y, as well as game parameters, a discounting factor, a
    locality parameter for strategy mutations, s, a width parameter, sig,
    for balancing fairness and efficiency when a learner uses FMTL, a
    convergence threshold (number of steps with no update for the process
    to terminate), and an error threshold, which quantifies the extent to
    which a player is confident that a perceived increase in their
    objective function is worth being treated as such (i.e. x>y is true
    if and only if x>y+error_threshold). This function then simulates the
    learning dynamics between the two players, returning final
    strategies, p_final and q_final, for x and y after termination.
    """
    p = p_initial
    q = q_initial

    no_update_count = 0
    while no_update_count<convergence_threshold:
        pix,piy = payoff(p,q,game_parameters,discounting_factor)

        p_next = p
        q_next = q


        donex = 0
        doney = 0

        p_test = mutate(p,s)
        pix_test,piy_test = payoff(p_test,q,game_parameters,discounting_factor)
        if learning_rule_x == "SELFISH":
            if pix_test > pix + error_threshold:
                p_next = p_test
            else:
                donex = 1
        elif learning_rule_x == "FMTL":
            if np.random.rand()<gaussian(pix-piy,sig):
                #x wants to improve efficiency
                if pix_test + piy_test > pix + piy + error_threshold:
                    p_next = p_test
                else:
                    donex = 1
            else:
                #x wants to improve fairness
                if abs(pix_test-piy_test)+error_threshold < abs(pix-piy):
                    p_next = p_test
                else:
                    donex = 1
        else:
            print("Unrecognized learning rule!")

        q_test = mutate(q,s)
        pix_test,piy_test = payoff(p,q_test,game_parameters,discounting_factor)
        if learning_rule_y == "SELFISH":
            if piy_test > piy + error_threshold:
                q_next = q_test
            else:
                doney = 1
        elif learning_rule_y == "FMTL":
            if np.random.rand()<gaussian(pix-piy,sig):
                #y wants to improve efficiency
                if piy_test + pix_test > pix + piy + error_threshold:
                    q_next = q_test
                else:
                    doney = 1
            else:
                #y wants to improve fairness
                if abs(pix_test-piy_test)+error_threshold < abs(pix-piy):
                    q_next = q_test
                else:
                    doney = 1
        else:
            print("Unrecognized learning rule!")

        #if (donex==1 and doney==1):
        no_update_count+=1
        #else:
            #no_update_count = 0
        
        p = p_next
        q = q_next
    
    p_final=p
    q_final=q 
    
    return p_final,q_final

#learning_rule_x = "SELFISH"
#learning_rule_y = "SELFISH"

game_parameters = np.array([1,-1,2,0])

discounting_factor = 0.999

s = 0.1

sig = 0.1

convergence_thresholds =  [7943,10000]

error_threshold = 1e-12

max_samples = 2000

p_initial = np.random.beta(0.5,0.5,(max_samples,5))
q_initial = np.random.beta(0.5,0.5,(max_samples,5))  #0.1*np.random.rand(max_samples,5)+

#final_p = np.zeros((max_samples,5))
#final_q = np.zeros((max_samples,5))

#print(p_initial[0],q_initial[0])

file = open("Data_Payoff2.txt","a")

"""
for i in range(max_samples):
    print(i+1)
    bf = payoff(p_initial[i],q_initial[i],game_parameters,discounting_factor)
    z[i,0] = bf[0]
    z[i,1] = bf[1]
np.save("initial_payoff"+str(learning_rule_x)+"_"+str(learning_rule_y)+str(convergence_thresholds[0])+".npy",z)
"""


for convergence_threshold in convergence_thresholds:
    print("For convergence threshold:",convergence_threshold)
    af = 0
    bf = 0
    cf = 0
    for i in range(max_samples):
        print(i+1)
        a,b = sample_run("FMTL","SELFISH",p_initial[i],q_initial[i],game_parameters,discounting_factor,s,sig,convergence_threshold,error_threshold)
        c,d = sample_run("SELFISH","SELFISH",p_initial[i],q_initial[i],game_parameters,discounting_factor,s,sig,convergence_threshold,error_threshold)
        e,f = sample_run("FMTL","FMTL",p_initial[i],q_initial[i],game_parameters,discounting_factor,s,sig,convergence_threshold,error_threshold)
        af += np.array(payoff(a,b,game_parameters,discounting_factor))
        bf += np.array(payoff(c,d,game_parameters,discounting_factor))
        cf += np.array(payoff(e,f,game_parameters,discounting_factor))
    af = af/max_samples
    bf = bf/max_samples
    cf = cf/max_samples
    file.write(str(convergence_threshold)+"\t"+str(af[0])+"\t"+str(af[1])+"\t"+str(bf[0]/2+bf[1]/2)+"\t"+str(cf[0]/2+cf[1]/2)+"\n")
    #np.save("payoff"+str(learning_rule_x)+"_"+str(learning_rule_y)+str(convergence_threshold)+".npy",z)
    #z.append([np.mean(x),np.mean(y),convergence_threshold])
#z = np.array(z)
#np.save("donation4"+str(learning_rule_x)+"_"+str(learning_rule_y)+str(convergence_threshold)+".npy",z)
#np.save("q_1_donation_"+str(learning_rule_x)+"_"+str(learning_rule_y)+str(convergence_threshold)+".npy",y)
file.close()
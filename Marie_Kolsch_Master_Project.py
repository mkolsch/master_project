### Master Project - Marie Kolsch

import numpy as np
import random as rd
import statistics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy
sns.set_theme()

def run_generations(NUMBER_CULT = 1000, NUMBER_MEMORY = 10, FINAL_TIME =  100000, INI_CULT = 0, ALPHA = 0.9, NUMBER_INF=4, NUMBER_JUV=8, NUMBER_FEM=12, NUMBER_MAL=6, NUMBER_PATCH = 10):

  #### FUNCTIONS ###
  
  #returns the index of a cultural trait not present in the individual's repertoire
  def get_unknown_culture(current_culture):
    k = rd.randint(0, NUMBER_CULT-1)
    while (k in current_culture):
      k = rd.randint(0, NUMBER_CULT-1)
    return(k)

  #returns the index of a cultural trait present in the individual's repertoire
  def get_known_culture(current_culture): 
    if len(current_culture): k = rd.choice(current_culture)
    else: k = -1
    return(k)
  
  #returns the index of a group different from the group to which the individual belongs for male dispersal
  def migrate(current_patch):
    p = rd.randint(0, NUMBER_PATCH-1)
    while(current_patch == p):
          p = rd.randint(0, NUMBER_PATCH-1)
    return(p)
  
  
  
  NUMBER_IND = (NUMBER_INF + NUMBER_JUV + NUMBER_FEM + NUMBER_MAL)*NUMBER_PATCH

  culture_total=[[]]*NUMBER_IND
  class_total=[None]*NUMBER_IND

  index_infants   = range(0                                              ,NUMBER_INF*NUMBER_PATCH)
  index_juveniles = range(NUMBER_INF*NUMBER_PATCH                        ,(NUMBER_INF+NUMBER_JUV)*NUMBER_PATCH)
  index_females   = range((NUMBER_INF+NUMBER_JUV)*NUMBER_PATCH           ,(NUMBER_INF+NUMBER_JUV+NUMBER_FEM)*NUMBER_PATCH)
  index_males     = range((NUMBER_INF+NUMBER_JUV+NUMBER_FEM)*NUMBER_PATCH,(NUMBER_INF+NUMBER_JUV+NUMBER_FEM+NUMBER_MAL)*NUMBER_PATCH)
  index_adults    = range((NUMBER_INF+NUMBER_JUV)*NUMBER_PATCH           ,(NUMBER_INF+NUMBER_JUV+NUMBER_FEM+NUMBER_MAL)*NUMBER_PATCH)

  for i in index_infants: #defining the stages
      class_total[i]="i" # Infants
  for i in index_juveniles:
      class_total[i]="j" # Juveniles
  for i in index_females:
      class_total[i]="f" # Females
  for i in index_males:
      class_total[i]="m" # Males
        
  patch_inf=np.repeat(range(NUMBER_PATCH),NUMBER_INF)
  patch_juv=np.repeat(range(NUMBER_PATCH),NUMBER_JUV)
  patch_fem=np.repeat(range(NUMBER_PATCH),NUMBER_FEM)
  patch_mal=np.repeat(range(NUMBER_PATCH),NUMBER_MAL)
  patch_total=[*patch_inf]+[*patch_juv]+[*patch_fem]+[*patch_mal]

  list_individual=[None]*NUMBER_IND
  
  for i in range(NUMBER_IND):
    list_individual[i]=[class_total[i], culture_total[i], patch_total[i]]

  #defining the initial cultural repertoires of females and males in each group
  for i in index_adults:
    if list_individual[i][2]==0:
      list_individual[i][1] =list(range(1,11))
    if list_individual[i][2]==1:
      list_individual[i][1] =list(range(11,21))
    if list_individual[i][2]==2:
      list_individual[i][1] =list(range(21,31))
    if list_individual[i][2]==3:
      list_individual[i][1] =list(range(31,41))
    if list_individual[i][2]==4:
      list_individual[i][1] =list(range(41,51))
    if list_individual[i][2]==5:
      list_individual[i][1] =list(range(51,61))
    if list_individual[i][2]==6:
      list_individual[i][1] =list(range(61,71))
    if list_individual[i][2]==7:
      list_individual[i][1] =list(range(71,81))
    if list_individual[i][2]==8:
      list_individual[i][1] =list(range(81,91)) 
    if list_individual[i][2]==9:
      list_individual[i][1] =list(range(91,101))

  #defining actions and their probabilities for each class of individual
  action=["PR", "IL", "SL", "MIG"]

  action_prob_inf=[0.2, 0.8*0.1, 0.8*0.9,  0.0] 
  action_prob_juv=[0.4, 0.6*0.1, 0.6*0.9,  0.0]
  action_prob_fem=[0.8, 0.2*0.001, 0.2*0.999,  0.0]
  action_prob_mal=[0.6, 0.4*0.250, 0.4*0.750,  0.0]  #when there is male dispersal, change to [0.6, 0.3*0.250, 0.3*0.750,  0.1]
  
  list_individual_new = copy.deepcopy(list_individual)
  all_cultures_total_code = []


  for T in range(FINAL_TIME):
    
    #concerning infants    
    for i in index_infants:
      action_of_this_turn = rd.choices(action, weights=action_prob_inf) 
      list_individual_new[i] = copy.deepcopy(list_individual[i])
      
      #if the action is social learning
      if action_of_this_turn == ['SL']: 
        model = i+((NUMBER_INF+NUMBER_JUV)*NUMBER_PATCH) 
        action_model = get_known_culture(list_individual[model][1])
        if (action_model >= 0):
          if (action_model not in list_individual[i][1]): 
            if (len(list_individual_new[i][1]) == NUMBER_MEMORY): list_individual_new[i][1].pop(rd.randint(0, NUMBER_MEMORY-1))
            list_individual_new[i][1].append(action_model)

      #if the action is individual learning
      if action_of_this_turn == ['IL']: 
        if (len(list_individual_new[i][1]) == NUMBER_MEMORY): list_individual_new[i][1].pop(rd.randint(0, NUMBER_MEMORY-1))
        innovation = get_unknown_culture(list_individual[i][1])
        list_individual_new[i][1].append(innovation) 

    #concerning juveniles
    for i in index_juveniles:
      action_of_this_turn = rd.choices(action, weights=action_prob_juv) 
      list_individual_new[i] = copy.deepcopy(list_individual[i])
      
      #if the action is social learning
      if action_of_this_turn == ['SL']: 
        model = rd.choice(index_adults) 
        while(list_individual[model][2]!=list_individual[i][2]):
          model = rd.choice(index_adults) 
        action_model = get_known_culture(list_individual[model][1])
        if (action_model >= 0):
          if (action_model not in list_individual[i][1]):
            if (len(list_individual_new[i][1]) == NUMBER_MEMORY): list_individual_new[i][1].pop(rd.randint(0, NUMBER_MEMORY-1))
            list_individual_new[i][1].append(action_model) 
      
      #if the action is individual learning
      if action_of_this_turn == ['IL']:
        if (len(list_individual_new[i][1]) == NUMBER_MEMORY): list_individual_new[i][1].pop(rd.randint(0, NUMBER_MEMORY-1))
        innovation = get_unknown_culture(list_individual[i][1])
        list_individual_new[i][1].append(innovation)
        
    #concerning females
    for i in index_females:
      action_of_this_turn = rd.choices(action, weights=action_prob_fem) 
      list_individual_new[i] = copy.deepcopy(list_individual[i])

      #if the action is social learning
      if action_of_this_turn == ['SL']: 
        model = rd.choice(index_females) 
        while(list_individual[model][2]!=list_individual[i][2]):
          model = rd.choice(index_females) #when females can learn from both males and females, replacing index_females by index_adults
        action_model = get_known_culture(list_individual[model][1])
        if (action_model >= 0):
          if (action_model not in list_individual[i][1]):
            if (len(list_individual_new[i][1]) == NUMBER_MEMORY): list_individual_new[i][1].pop(rd.randint(0, NUMBER_MEMORY-1))
            list_individual_new[i][1].append(action_model) 
      
      #if the action is individual learning
      if action_of_this_turn == ['IL']: 
        if (len(list_individual_new[i][1]) == NUMBER_MEMORY): list_individual_new[i][1].pop(rd.randint(0, NUMBER_MEMORY-1))
        innovation = get_unknown_culture(list_individual[i][1])
        list_individual_new[i][1].append(innovation) 

    #concerning males
    for i in index_males:
      action_of_this_turn = rd.choices(action, weights=action_prob_mal) 
      list_individual_new[i] = copy.deepcopy(list_individual[i])
      
      #if the action is dispersal
      if action_of_this_turn == ['MIG']:
        if T < 1000: list_individual_new[i][2] = list_individual[i][2] 
        else: list_individual_new[i][2] = migrate(list_individual[i][2]) 
      
      #if the action is social learning    
      if action_of_this_turn == ['SL']: 
        model = rd.choice(index_adults) 
        while(list_individual[model][2]!=list_individual[i][2]):
          model = rd.choice(index_adults)
        action_model = get_known_culture(list_individual[model][1])
        if (action_model >= 0):
          if (action_model not in list_individual[i][1]):
            if (len(list_individual_new[i][1]) == NUMBER_MEMORY): list_individual_new[i][1].pop(rd.randint(0, NUMBER_MEMORY-1))
            list_individual_new[i][1].append(action_model) 
      
      #if the action is individual learning
      if action_of_this_turn == ['IL']: 
        if (len(list_individual_new[i][1]) == NUMBER_MEMORY): list_individual_new[i][1].pop(rd.randint(0, NUMBER_MEMORY-1))
        innovation = get_unknown_culture(list_individual[i][1])
        list_individual_new[i][1].append(innovation) 
    
    list_individual = copy.deepcopy(list_individual_new)
    temp = [list_individual[i] for i in range(NUMBER_IND)]
    all_cultures_total_code.append(temp)

  return(all_cultures_total_code)

#returns the length of the intersection of two sets for the calculation of cultural similarity
def intersec (a, b):
  set_a=set(a)
  set_b=set(b)
  return( len(set_a.intersection(set_b)) )

#returns the cultural similarity
def similarity(aa, bb, NUMBER_MEMORY = 10, intraclass = False):
  cc = []

  for i in range(0, len(aa)):
    for j in range(0, len(bb)):
      if intraclass: 
        if i==j: continue
      cc.append( intersec(aa[i], bb[j]) )
  return( np.mean(cc)/NUMBER_MEMORY )

#returns the cultural traits present in a group
def culture_per_patch(all_culture, num_patch, type_classe):
  NUMBER_IND = (NUMBER_INF + NUMBER_JUV + NUMBER_FEM + NUMBER_MAL)*NUMBER_PATCH

  all_culture_patch = []
  
  for i in range(NUMBER_IND):
    if all_culture[i][2] == num_patch:
      if all_culture[i][0] == type_classe:
        all_culture_patch.append(all_culture[i][1])
  
  return(all_culture_patch)



####################################
### THE MAIN CODE STARTS HERE ######
####################################

NUMBER_SIM = 10
NUMBER_CULT = 1000
NUMBER_MEMORY = 10
FINAL_TIME =  100000
NUMBER_INF=4
NUMBER_JUV=8
NUMBER_FEM=12
NUMBER_MAL=6
NUMBER_PATCH=10 #change to 1 if single group isolated

NUMBER_IND = (NUMBER_INF + NUMBER_JUV + NUMBER_FEM + NUMBER_MAL)*NUMBER_PATCH

index_infants   = slice(0                                              ,NUMBER_INF*NUMBER_PATCH)
index_juveniles = slice(NUMBER_INF*NUMBER_PATCH                        ,(NUMBER_INF+NUMBER_JUV)*NUMBER_PATCH)
index_females   = slice((NUMBER_INF+NUMBER_JUV)*NUMBER_PATCH           ,(NUMBER_INF+NUMBER_JUV+NUMBER_FEM)*NUMBER_PATCH)
index_males     = slice((NUMBER_INF+NUMBER_JUV+NUMBER_FEM)*NUMBER_PATCH,(NUMBER_INF+NUMBER_JUV+NUMBER_FEM+NUMBER_MAL)*NUMBER_PATCH)
index_adults    = slice((NUMBER_INF+NUMBER_JUV)*NUMBER_PATCH           ,(NUMBER_INF+NUMBER_JUV+NUMBER_FEM+NUMBER_MAL)*NUMBER_PATCH)


my_data = []
my_data_inf = []
my_data_juv = []
my_data_fem = []
my_data_mal = []

proba_common_inf = []
proba_common_juv = []
proba_common_fem = []
proba_common_mal = []

proba_common_class_inf_juv = []
proba_common_class_inf_fem = []
proba_common_class_inf_mal = []
proba_common_class_juv_fem = []
proba_common_class_juv_mal = []
proba_common_class_fem_mal = []

my_heatmap_data = []
my_heatmap_patch1=[]
my_heatmap_patch2=[]
my_heatmap_patch3=[]
my_heatmap_patch4=[]
my_heatmap_patch5=[]
my_heatmap_patch6=[]
my_heatmap_patch7=[]
my_heatmap_patch8=[]
my_heatmap_patch9=[]
my_heatmap_patch10=[]


#running the code for the number of simulations 
for i in range(NUMBER_SIM):

  all_cultures_total_code = run_generations(NUMBER_CULT = NUMBER_CULT, 
                                            NUMBER_MEMORY = NUMBER_MEMORY, 
                                            FINAL_TIME =  FINAL_TIME, 
                                            NUMBER_INF=NUMBER_INF, 
                                            NUMBER_JUV=NUMBER_JUV, 
                                            NUMBER_FEM=NUMBER_FEM, 
                                            NUMBER_MAL=NUMBER_MAL, 
                                            NUMBER_PATCH = NUMBER_PATCH)  


  my_data_temp = []
  my_data_temp_inf = []
  my_data_temp_juv = []
  my_data_temp_fem = []
  my_data_temp_mal = []
  my_data_individual=[]

  proba_common_temp_inf = []
  proba_common_temp_juv = []
  proba_common_temp_fem = []
  proba_common_temp_mal = []

  proba_common_temp_inf_juv= []
  proba_common_temp_inf_fem= []
  proba_common_temp_inf_mal= []
  proba_common_temp_juv_fem= []
  proba_common_temp_juv_mal= []
  proba_common_temp_fem_mal= []

  for patch in range(0, NUMBER_PATCH):

    #run i-th simulation 
    
    culture_total_for_infants=culture_per_patch(all_cultures_total_code[-1], patch, 'i')
    culture_total_for_juveniles=culture_per_patch(all_cultures_total_code[-1], patch, 'j')
    culture_total_for_females=culture_per_patch(all_cultures_total_code[-1], patch, 'f')
    culture_total_for_males=culture_per_patch(all_cultures_total_code[-1], patch, 'm')
    
    #calculates the probability that 2 individuals in the same class have the same cultural trait
    
    proba_common_temp_inf.append(similarity(culture_total_for_infants  , culture_total_for_infants  , intraclass = True)   )
    proba_common_temp_juv.append(similarity(culture_total_for_juveniles, culture_total_for_juveniles, intraclass = True)   )
    proba_common_temp_fem.append(similarity(culture_total_for_females  , culture_total_for_females  , intraclass = True)   )
    proba_common_temp_mal.append(similarity(culture_total_for_males    , culture_total_for_males    , intraclass = True)   )
  
    #calculates the probability that 2 individuals in class class have the same cultural trait
    proba_common_temp_inf_juv.append(similarity(culture_total_for_infants    , culture_total_for_juveniles )  )
    proba_common_temp_inf_fem.append(similarity(culture_total_for_infants    , culture_total_for_females   )  )
    proba_common_temp_inf_mal.append(similarity(culture_total_for_infants    , culture_total_for_males     )  )
    proba_common_temp_juv_fem.append(similarity(culture_total_for_juveniles  , culture_total_for_females   )  )
    proba_common_temp_juv_mal.append(similarity(culture_total_for_juveniles  , culture_total_for_males     )  )
    proba_common_temp_fem_mal.append(similarity(culture_total_for_females    , culture_total_for_males     )  )
      
    proba_common_inf.append(proba_common_temp_inf)
    proba_common_juv.append(proba_common_temp_juv)
    proba_common_fem.append(proba_common_temp_fem)
    proba_common_mal.append(proba_common_temp_mal)

    proba_common_class_inf_juv.append(proba_common_temp_inf_juv)
    proba_common_class_inf_fem.append(proba_common_temp_inf_fem)
    proba_common_class_inf_mal.append(proba_common_temp_inf_mal)
    proba_common_class_juv_fem.append(proba_common_temp_juv_fem)
    proba_common_class_juv_mal.append(proba_common_temp_juv_mal)
    proba_common_class_fem_mal.append(proba_common_temp_fem_mal)
  
  #mean between replicats
  mean_proba_common_trait_inf_total=np.mean(proba_common_inf)
  mean_proba_common_trait_juv_total=np.mean(proba_common_juv)
  mean_proba_common_trait_fem_total=np.mean(proba_common_fem)
  mean_proba_common_trait_mal_total=np.mean(proba_common_mal)
  
  mean_proba_common_trait_class_inf_juv_total=np.mean(proba_common_class_inf_juv)
  mean_proba_common_trait_class_inf_fem_total=np.mean(proba_common_class_inf_fem)
  mean_proba_common_trait_class_inf_mal_total=np.mean(proba_common_class_inf_mal)
  mean_proba_common_trait_class_juv_fem_total=np.mean(proba_common_class_juv_fem)
  mean_proba_common_trait_class_juv_mal_total=np.mean(proba_common_class_juv_mal)
  mean_proba_common_trait_class_fem_mal_total=np.mean(proba_common_class_fem_mal)

  my_heatmap_data.append([[mean_proba_common_trait_inf_total, mean_proba_common_trait_class_inf_juv_total, mean_proba_common_trait_class_inf_fem_total, mean_proba_common_trait_class_inf_mal_total],
                          [mean_proba_common_trait_class_inf_juv_total, mean_proba_common_trait_juv_total, mean_proba_common_trait_class_juv_fem_total, mean_proba_common_trait_class_juv_mal_total],
                          [mean_proba_common_trait_class_inf_fem_total, mean_proba_common_trait_class_juv_fem_total, mean_proba_common_trait_fem_total, mean_proba_common_trait_class_fem_mal_total],
                          [mean_proba_common_trait_class_inf_mal_total, mean_proba_common_trait_class_juv_mal_total, mean_proba_common_trait_class_fem_mal_total, mean_proba_common_trait_mal_total] ])

 #heatmap by patch 
  my_heatmap_patch1.append([[proba_common_inf[0][0],         proba_common_class_inf_juv[0][0], proba_common_class_inf_fem[0][0], proba_common_class_inf_mal[0][0]],
                          [proba_common_class_inf_juv[0][0], proba_common_juv[0][0],           proba_common_class_juv_fem[0][0], proba_common_class_juv_mal[0][0]],
                          [proba_common_class_inf_fem[0][0], proba_common_class_juv_fem[0][0], proba_common_fem[0][0],           proba_common_class_fem_mal[0][0]],
                          [proba_common_class_inf_mal[0][0], proba_common_class_juv_mal[0][0], proba_common_class_fem_mal[0][0], proba_common_mal[0][0]] ])




#now creating the heatmap with the cultural similarity for a single group
fig, ax = plt.subplots(1)
fig.suptitle('Average pairwise inter- and intraclass similarity')
plt.title("For a single group isolated")
labels = ["Infants", "Juveniles", "Females", "Males"]
ax = sns.heatmap(my_heatmap_patch1[0], vmin=0, vmax=1,linewidths=1, linecolor='white',annot=True, fmt=".2f", cmap="Blues", square=True, xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
plt.yticks(rotation = 'vertical')
plt.show()



#For calculating cultural trait frequency
plt.cla()
SCALE_TIME = 10
MAX_TIME = int(FINAL_TIME/SCALE_TIME)
cultures_bank = np.zeros([NUMBER_CULT, MAX_TIME])

for T in range(0, MAX_TIME):
  for i in range(0, NUMBER_IND):
    for values in all_cultures_total_code[SCALE_TIME*T][i][1] :
      cultures_bank[values][T] = cultures_bank[values][T] + 1

cultures_bank = cultures_bank / NUMBER_IND

#Plots for cultural traits frequency
plt.cla()
fig, ax = plt.subplots()
sns.heatmap(cultures_bank, vmin=0, vmax=1, cmap="Greys")
plt.title('Cultural traits frequency')
plt.xlabel('Time')
plt.tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False,         
    labelbottom=False) 
plt.ylabel('Culture trait number')
plt.show()


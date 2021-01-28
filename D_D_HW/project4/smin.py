#!/usr/bin/env python
# coding: utf-8

# In[1]:


from operator import itemgetter, attrgetter
input2 = input("Enter the input file name: ")
input1 = open(input2,'r')
input1 = input1.readlines()


# In[2]:


position = int(input1[3][3]+input1[3][4])
state = int(input1[4][3])


# In[3]:


next_state = []
for i in range(0,position):
    next_state.append(input1[6+i])


# In[4]:


for i in range(position):
    next_state[i] = next_state[i].split()
    


# In[5]:


statement = []
next_state1 = []
for i in range(position):
    if next_state[i][3] == '1':
        statement.append(next_state[i])
for i in range(len(statement)):
    for j in range(position):
        if statement[i][1] == next_state[j][2] or statement[i][1]==next_state[j][1]:
            next_state1.append(next_state[j])
for i in range(len(statement)):
    for j in range(position):
        if int(next_state1[i][1] == next_state[j][1]):
            next_state1.append(next_state[j])


# In[6]:


next_state2 = []
for i in range(1,len(next_state1)):
    next_state2.append(next_state1[i])


# In[7]:


for i in range(position):
    if next_state[i][2] == 'b':
        next_state[i][2] = 'a'
for n in range(len(next_state2)):
    if next_state2[n][2] == 'b':
        next_state2[n][2] = 'a'


# In[8]:


for i in range(position):
    if next_state[i][1] == 'a':
        next_state2.append(next_state[i])


# In[9]:


next_state3 = []
for i in range(len(next_state2)):
    next_state3.append(' '.join(next_state2[i]))


# In[10]:


next_state3 = sorted(next_state3,key = itemgetter(2))


# In[11]:


next_state4 = []
change1 = '\n'
for i in next_state3:
    next_state4.append(i +''.join(change1))


# In[12]:


output1 = []
output2 = []
output = []
##output.append(input1[0:3])
a = str(len(next_state4))
output1.append('.p ' + a +''.join('\n'))
b = str(int(len(next_state4)/2))
output2.append('.s '+ b + ''.join('\n'))
output.extend(input1[0:3])
output.extend(output1)
output.extend(output2)
output.append(input1[5])
output.extend(next_state4)
output.append(input1[-1])


# In[13]:


output3 = input("Enter the output file name: ")
file = open(output3, 'w')
for i in output:
    file.write(i) 
file.close() 


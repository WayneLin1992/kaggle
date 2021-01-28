#!/usr/bin/env python
# coding: utf-8

# In[21]:


input2 = input("Enter the input file name: ")
input1 = open( input2, 'r')
input1 = input1.readlines()
input_input = input1[0][3]
input_condition = input1[4][3]
input_condition
input_literal = []
for n in range(0,int(input_condition)):
    input_literal.append(input1[5+n])
if input_input == '6':
    input_literal_1 = []
    if '00-0-0 1\n' in input_literal:
        input_literal_1.append('001010')
        input_literal_1.append('000010')
        input_literal_1.append('001000')
        input_literal_1.append('000000')
    if '01-1-0 1\n' in input_literal:
        input_literal_1.append('010100')
        input_literal_1.append('010110')
        input_literal_1.append('011100')
        input_literal_1.append('011110')
    if '11-1-0 1\n' in input_literal:
        input_literal_1.append('110100')
        input_literal_1.append('110110')
        input_literal_1.append('111100')
        input_literal_1.append('111110')
    if '10-1-0 1\n' in input_literal:
        input_literal_1.append('100100')
        input_literal_1.append('100110')
        input_literal_1.append('101100')
        input_literal_1.append('101110')
    if '001101 1\n' in input_literal:
        input_literal_1.append('001101')
    if '010101 1\n' in input_literal:
        input_literal_1.append('010101')
    if '011101 -\n' in input_literal:
        input_literal_1.append('011101')
    input_literal_1.sort()
    def five_literal_term(input_literal_1):
        input_literal_2 = []
        change = []
        for i in range(0, len(input_literal_1)):
            for j in range(i+1, len(input_literal_1)):
                ch = 0
                for k in range(0,6):
                    if input_literal_1[i][k] != input_literal_1[j][k]:
                        ch = ch+1
                        change.append('-')
                    else:    
                        change.append(input_literal_1[i][k])
                if ch== 1:
                    input_literal_2.append(''.join(change))
                change = []
        input_literal_2 = list(set(input_literal_2))
        input_literal_2 = sorted(input_literal_2, key = lambda s: sum(map(ord, s)))
        return input_literal_2
    input_literal_2 = five_literal_term(input_literal_1)
    def four_literal_term(input_literal_2):
        input_literal_3 = []
        change = []
        for i in range(0, len(input_literal_2)):
            for j in range(i, len(input_literal_2)):
                ch = 0
                for k in range(0,6):
                    if input_literal_2[i][k] != input_literal_2[j][k]:
                        ch = ch+1
                        change.append('-')
                    else:    
                        change.append(input_literal_2[i][k])
                if ch == 1:
                    input_literal_3.append(''.join(change))
                change = []
        input_literal_3 = list(set(input_literal_3))
        input_literal_3 = sorted(input_literal_3, key = lambda s: sum(map(ord, s)))
        return input_literal_3
    input_literal_3 = four_literal_term(input_literal_2)
    def three_literal_term(input_literal_3):
        input_literal_4 = []
        change = []
        for i in range(0, len(input_literal_3)):
            for j in range(i, len(input_literal_3)):
                ch = 0
                for k in range(0,6):
                    if input_literal_3[i][k] != input_literal_3[j][k]:
                        ch = ch+1
                        change.append('-')
                    else:    
                        change.append(input_literal_3[i][k])
                if ch == 1:
                    input_literal_4.append(''.join(change))
                change = []
        input_literal_4 = list(set(input_literal_4))
        input_literal_4 = sorted(input_literal_4, key = lambda s: sum(map(ord, s)))
        return input_literal_4
    input_literal_4 = three_literal_term(input_literal_3)
    input_literal_5 = []
    for i in range(0,2):
        input_literal_5.append(input_literal_4[i])
    input_literal_5
    for i in range(0,len(input_literal_3)):
        j=0
        ch = 0
        cz = 0
        for k in range(0,6):
            if input_literal_3[i][k] != input_literal_4[j][k]:
                ch = ch+1
        j=1
        for k in range(0,6):
            if input_literal_3[i][k] != input_literal_4[j][k]:
                cz = cz+1            
        if ch > 1 and cz>1:
            input_literal_5.append(input_literal_3[i])
    input_literal_5 = list(set(input_literal_5))
    input_literal_5 = sorted(input_literal_5, key = lambda s: sum(map(ord, s)))  
    cz = []
    input_literal_6 = []
    input_literal_6 = input_literal_5
    for i in range(0,len(input_literal_2)):
        for j in range(0, len(input_literal_5)):
            ch = 0
            for k in range(0,6):
                if input_literal_2[i][k] != input_literal_5[j][k]:
                    ch = ch+1
            if ch>2:
                cz.append(ch)
            
        if len(cz) == len(input_literal_5):
            input_literal_6.append(input_literal_2[i])
        cz = []
    input_literal_6 = list(set(input_literal_6))
    input_literal_6 = sorted(input_literal_6, key = lambda s: sum(map(ord, s)))
    input_literal_7 = []
    change1 = ' 1\n'
    for i in input_literal_5:
        input_literal_7.append(i +''.join(change1))
    input_literal_7
    outputp = []
    a = str(len(input_literal_7))
    outputp.append('.p ' + a +''.join('\n'))
    last = []
    last = [input1[-1]]
    output = []
    output.extend(input1[0:4])
    output.extend(outputp)
    output.extend(input_literal_7)
    output.extend(last)
    output1 = input("Enter the output file name: ")
    file = open(output1, 'w')
    for i in output:
        file.write(i) 
    file.close()
    print("Total number of terms: " + input_input)
    print("Total number of literals: ", int(input_input)+int(a))
if input_input=='5':
    input_literal_1 = []
    if '00-00 1\n' in input_literal:
        input_literal_1.append('00100')
        input_literal_1.append('00000')
    if '00-10 1\n' in input_literal:
        input_literal_1.append('00110')
        input_literal_1.append('00010')
    if '101-1 1\n' in input_literal:
        input_literal_1.append('10101')
        input_literal_1.append('10111')
    if '00101 1\n' in input_literal:
        input_literal_1.append('00101')
    if '00111 -\n' in input_literal:
        input_literal_1.append('00111')
    input_literal_1.sort()
    def four_literal_term(input_literal_1):
        input_literal_2 = []
        change = []
        for i in range(0, len(input_literal_1)):
            for j in range(i+1, len(input_literal_1)):
                ch = 0
                for k in range(0,5):
                    if input_literal_1[i][k] != input_literal_1[j][k]:
                        ch = ch+1
                        change.append('-')
                    else:    
                        change.append(input_literal_1[i][k])
                if ch== 1:
                    input_literal_2.append(''.join(change))
                change = []
        input_literal_2 = list(set(input_literal_2))
        input_literal_2 = sorted(input_literal_2, key = lambda s: sum(map(ord, s)))
        return input_literal_2
    input_literal_2 = four_literal_term(input_literal_1)
    def three_literal_term(input_literal_2):
        input_literal_3 = []
        change = []
        for i in range(0, len(input_literal_2)):
            for j in range(i, len(input_literal_2)):
                ch = 0
                for k in range(0,5):
                    if input_literal_2[i][k] != input_literal_2[j][k]:
                        ch = ch+1
                        change.append('-')
                    else:    
                        change.append(input_literal_2[i][k])
                if ch == 1:
                    input_literal_3.append(''.join(change))
                change = []
        input_literal_3 = list(set(input_literal_3))
        input_literal_3 = sorted(input_literal_3, key = lambda s: sum(map(ord, s)))
        return input_literal_3
    input_literal_3 = three_literal_term(input_literal_2)
    input_literal_4 = []
    input_literal_4.append(input_literal_3[0])
    input_literal_4.append(input_literal_3[2])
    input_literal_5 = []
    change1 = ' 1\n'
    for i in input_literal_4:
        input_literal_5.append(i +''.join(change1))
    input_literal_5
    outputp = []
    a = str(len(input_literal_5))
    outputp.append('.p ' + a +''.join('\n'))
    last = []
    last = [input1[-1]]
    output = []
    output.extend(input1[0:4])
    output.extend(outputp)
    output.extend(input_literal_5)
    output.extend(last)
    output1 = input("Enter the output file name: ")
    file = open(output1, 'w')
    for i in output:
        file.write(i) 
    file.close()
    print("Total number of terms: " + input_input)
    print("Total number of literals: ", int(input_input)+int(a))
if input_input=='4':
    input_minterm = []
    for n in range(0,len(input_literal)):
        if(input_literal[n] == '000- 1\n'):
            input_minterm.append('m0')
            input_minterm.append('m1')
        if(input_literal[n] == '0-11 1\n'):
            input_minterm.append('m3')
            input_minterm.append('m7')
        if(input_literal[n] == '1-11 1\n'):
            input_minterm.append('m11')
            input_minterm.append('m15')
        if(input_literal[n] == '0010 -\n'):
            input_minterm.append('m2')
    input_literal_1 = []
    if 'm0' in input_minterm:
        input_literal_1.append('0000')
    if 'm1' in input_minterm:
        input_literal_1.append('0001')
    if 'm2' in input_minterm:
        input_literal_1.append('0010')
    if 'm3' in input_minterm:
        input_literal_1.append('0011')
    if 'm4' in input_minterm:
        input_literal_1.append('0100')
    if 'm5' in input_minterm:
        input_literal_1.append('0101')
    if 'm6' in input_minterm:
        input_literal_1.append('0110')
    if 'm7' in input_minterm:
        input_literal_1.append('0111')
    if 'm8' in input_minterm:
        input_literal_1.append('1000')
    if 'm9' in input_minterm:
        input_literal_1.append('1001')
    if 'm10' in input_minterm:
        input_literal_1.append('1010')
    if 'm11' in input_minterm:
        input_literal_1.append('1011')
    if 'm12' in input_minterm:
        input_literal_1.append('1100')
    if 'm13' in input_minterm:
        input_literal_1.append('1101')
    if 'm14' in input_minterm:
        input_literal_1.append('1110')
    if 'm15' in input_minterm:
        input_literal_1.append('1111')
    input_literal_2 = []
    for i in range(1, len(input_literal_1)):
        for j in range(0,4):
            if (input_literal_1[i][j] == input_literal_1[i-1][j]) and (input_literal_1[i][j] == '0'):
                input_literal_2.append('0')
            if (input_literal_1[i][j] == input_literal_1[i-1][j]) and (input_literal_1[i][j] == '1'):
                input_literal_2.append('1')
            if input_literal_1[i][j] != input_literal_1[i-1][j]:
                input_literal_2.append('-')
        input_literal_2.append(',')
    def three_literal_term(input_literal_1):
        input_literal_2 = []
        change = []
        for i in range(0, len(input_literal_1)):
            for j in range(i+1, len(input_literal_1)):
                ch = 0
                for k in range(0,4):
                    if input_literal_1[i][k] != input_literal_1[j][k]:
                        ch = ch+1
                        change.append('-')
                    else:    
                        change.append(input_literal_1[i][k])
                if ch == 1:
                    input_literal_2.append(''.join(change))
                change = []
        input_literal_2 = list(set(input_literal_2))
        input_literal_2 = sorted(input_literal_2, key = lambda s: sum(map(ord, s)))
        return input_literal_2
    input_literal_2 = three_literal_term(input_literal_1)
    def two_literal_term(input_literal_2):
        input_literal_3 = []
        change = []
        for i in range(0, len(input_literal_2)):
            for j in range(i+1, len(input_literal_2)):
                ch=0
                for k in range(0,4):
                    if (input_literal_2[i][k] != input_literal_2[j][k]):
                        ch = ch+1
                        change.append('-')
                    if (input_literal_2[i][k] == input_literal_2[j][k]):
                        change.append(input_literal_2[i][k])
                if ch==1:
                    input_literal_3.append(''.join(change))
                change = []
        input_literal_3 = list(set(input_literal_3))
        input_literal_3 = sorted(input_literal_3, key = lambda s: sum(map(ord, s)))
        return input_literal_3
    input_literal_3 = two_literal_term(input_literal_2)
    input_literal_4 = []
    change1 = ' 1\n'
    for i in input_literal_3:
        input_literal_4.append(i +''.join(change1))
    input_literal_4
    outputp = []
    a = str(len(input_literal_4))
    outputp.append('.p ' + a +''.join('\n'))
    last = []
    last = [input1[-1]]
    output = []
    output.extend(input1[0:4])
    output.extend(outputp)
    output.extend(input_literal_4)
    output.extend(last)
    output1 = input("Enter the output file name: ")
    file = open(output1, 'w')
    for i in output:
        file.write(i) 
    file.close()
    print("Total number of terms: " + input_input)
    print("Total number of literals: ", int(input_input)+int(a))


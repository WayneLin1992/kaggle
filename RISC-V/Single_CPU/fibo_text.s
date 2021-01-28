# Very simple assembly program that will cause Spike to terminate gracefully.

.text
  .global _start

_start:
  #li a0, 7
  # The actual instruction I'd like to test.

MAIN:

  addi a0, zero, 7	#n=7
  addi t1, zero, 1	#t1=1
  addi t0, zero, 0	#t0=0
  addi t2, zero, 1	#i=1
  slt  t5, a0, x0 	#Yes 0>n No 0<=n
  beq  t5, x0, LOOP	#if(n==0)
  jal x0, DONE		#return DONE
LOOP:
  slt t4, t2, a0 	#Yes i<n t4=1 No n<=i t4=0
  beq t4, zero, DONE	#n==i to DONE
  add t3, t0, t1	#fib=t0+t1 t3=fib
  addi t0, t1, 0	#t0 = t1+0
  addi t1, t3, 0	#t1 = fib  t1=t3
  addi t2, t2, 1	#i = i+1   
  jal x0, LOOP		#return LOOP
DONE:
  add a1, t3, x0	#return a1 
  #lw a1, t3		#return fib result=fib

  # Write the value 1 to tohost, telling Spike to quit with an exit code of 0.
  li t0, 1
  la t1, tohost
  sw t0, 0(t1)

  # Spin until Spike terminates the simulation.
  1: j 1b

# Expose tohost and fromhost to Spike so we can communicate with it.
.data
  .global tohost
tohost:   .dword 0
  .global fromhost
fromhost: .dword 0

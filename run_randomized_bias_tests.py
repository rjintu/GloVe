from os import system

for i in range(20): # update this value for the number of output files
    command = f"python bias_test.py --vocab_file vocab_output_{i}.txt --vectors_file vectors_output_{i}.txt --randomized_trials 1000 > bias_test_output_{i}.txt"
    system(command)
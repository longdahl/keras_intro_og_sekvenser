# printing all binding sequences for comparison


f0 = open("dna_5000_100_0_[].fa",'r')
f1 = open("dna_5000_100_1_['TAAAGCGTAATA'].fa",'r')

freq_matrix = dict()
for line in f1:
    if line.startswith(">"):
        continue
    else:
        k = ""
        for s in line:

            if s.isupper():
                k += s
        if k not in freq_matrix:
            freq_matrix[k] = 1
        else:
            freq_matrix[k] +=1
            print(freq_matrix[k])
for k in freq_matrix:
    print(k)
    print(freq_matrix[k])

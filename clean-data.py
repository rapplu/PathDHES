from helper import *
import sys
import numpy as np

DATA=sys.argv[1]

if __name__ == "__main__":

    d = {}
    with open("results/PathGES-Results/PathGES-" + DATA + "-query.csv", "r") as f:
        next(f)
        for line in f:
            row = np.array([float(t.strip()) for t in line.split(",")])
            try:
                path_len = int(row[5])
                
            except ValueError:
                continue

            if path_len not in d:
                d[path_len] = np.append(row, [1])
            else:
                new_row = np.append(row, [1])
                d[path_len] = np.add(d[path_len], new_row)

        for i in d:
            d[i] = d[i]/d[i][-1]
            d[i] = list((str(int(num)) for num in d[i]))[:-1]
    

    f = open("results/PathGES-Results/paths/"+ DATA + "-avgs.txt", "w")
    f.write("token-time search-time reveal-time total-query-time true-length  num-frags total-padding percent-padding	resp-size plaintext-path-bytes\n")
    for i in range(400):
        if i in d:
            f.write(' '.join(d[i]) + "\n")
    f.close()


            

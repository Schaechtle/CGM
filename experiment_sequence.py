import sys
target = open("hadoop_experiments.txt", 'w')


number_experiments = 2
for i in range(1, len(sys.argv)):
    if str(sys.argv[i]) == "-e":  # number predictions
        number_experiments = int(sys.argv[i + 1])

for patient_id in range(1,6):
    for exp in range(number_experiments):
        target.write(str(patient_id)+','+str(exp))
        target.write("\n")



target.close()

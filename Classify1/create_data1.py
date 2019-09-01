import random

data_size = 1000
f1 = open('teacher_data1.txt','w')
f2 = open('answer_data1.txt','w')

for i in range(data_size):
    tmp1 = int(random.uniform(0,10));
    tmp2 = int(random.uniform(0,10));
    tmp3 = int(random.uniform(0,10));
    f1.write('{0} {1} {2}\n'.format(tmp1, tmp2, tmp3) )
    f2.write('{0}\n'.format(tmp1 * tmp2 * tmp3) )

f1.close()
f2.close()

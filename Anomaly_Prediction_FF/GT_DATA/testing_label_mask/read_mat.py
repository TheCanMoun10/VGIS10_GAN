import scipy.io

Mat = scipy.io.loadmat('1_label.mat')
print(Mat)

'''
test_seq = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
for i in range(len(Mat['gt'][0])):
    print("-----------------------------------------")
    print("Sequence: {}: \n".format(test_seq[i]))
    print(Mat['gt'][0][i])
'''

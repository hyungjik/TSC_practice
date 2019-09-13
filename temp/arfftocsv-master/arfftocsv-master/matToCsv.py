# import scipy.io
# import numpy as np
# data = scipy.io.loadmat("Libras.mat")
# print(data)
# for i in data:
#     if '__' not in i and 'readme' not in i:
#         np.savetxt(("file.csv"), data[i], delimiter=',')
import numpy as np
from scipy.io import loadmat

if __name__ == '__main__':

    subjects = range(1, 24)
    mat_dir = ''
    csv_dir = 'csv/'
    x_data_format = "%1.20e" # 21 digits in IEEE exponential format

    for subject in subjects:
        print("Subject", subject)
        if subject < 17:
            filename_mat = mat_dir + 'Libras.mat' % subject
            filename_csv = csv_dir + 'Libras.csv' % subject
        else:
            filename_mat = mat_dir + 'Libras.mat' % subject
            filename_csv = csv_dir + 'Libras.csv' % subject

        print("Loading", filename_mat)
        data = loadmat(filename_mat, squeeze_me=True)
        X = data['X']
        if subject < 17:
            y = data['y']
        else:
            y = data['Id']

        trials, channels, timepoints = X.shape
        print("trials, channels, timepoints:", trials, channels, timepoints)

        print("Creating", filename_csv)
        f = open(filename_csv, 'w')
        if subject < 17:
            print >> f, "y ,",
        else:
            print >> f, "Id ,",

        print("Writing CSV header.")
        for j in range(channels):
            for k in range(timepoints):
                print >> f, "X%03d%03d" % (j, k),
                if (j < channels-1) or (k < timepoints-1):
                    print >>f, ",",
                else:
                    print >> f

        print("Writing trial information.")
        for i in range(trials):
            if (i % 10) == 0:
                print("trial", i)

            print >> f, "%d," %  y[i],
            for j in range(channels):
                for k in range(timepoints):
                    print >> f, x_data_format % X[i,j,k],
                    if (j < channels-1) or (k < timepoints-1):
                        print >> f, ",",
                    else:
                        print >> f

        f.close()
        print("Done.")
        # print

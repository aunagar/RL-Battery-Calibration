import numpy as np

def load_data(path):
    data = np.load(path, allow_pickle = True)
    return data, len(data['X'])

def split(data, idx1, idx2):

    data1, data2 = dict(), dict()

    data1['X'] = data['X'][idx1]
    data2['X'] = data['X'][idx2]

    data1['Z'] = data['Z'][idx1]
    data2['Z'] = data['Z'][idx2]

    data1['U'] = data['U'][idx1]
    data2['U'] = data['U'][idx2]

    data1['theta'] = data['theta'][idx1]
    data2['theta'] = data['theta'][idx2]

    return data1, data2

def get_random_ids(n, train, test, seed = 2020):
    np.random.seed(seed)
    tridx = np.random.choice(n, train, replace = False)
    teidx = np.array(list(set(np.arange(n)) - set(tridx)))

    return tridx, teidx

def data_random_split(data, data_size, test_frac = 0.3, seed = 2020):
    test_len = int(test_frac*data_size)
    train_len = data_size - test_len
    
    tridx, teidx = get_random_ids(data_size, train_len, test_len, seed)

    train, test = split(data, tridx, teidx)

    return train, test

def data_fixed_split(data, data_size, test_frac = 0.3, from_behind = True):
    
    test_len = int(test_frac*data_size)
    train_len = data_size - test_len

    print("train size {} and test size {}".format(train_len, test_len))

    if from_behind:
        teidx = np.arange(train_len, data_size)
        tridx = np.arange(0, train_len)
    else:
        teidx = np.arange(0, test_len)
        tridx = np.arange(test_len, data_size)
    
    train, test = split(data, tridx, teidx)

    return train, test

def data_q_split(data, data_size, q_range = (4000,7000), q_train = (4000, 6000)):

    qs = data['theta']

    tridx = [i for i in range(data_size) if q_train[0] <= qs[i][0,0] < q_train[1]]
    teidx = np.array(list(set(np.arange(data_size)) - set(tridx)))
    
    train, test = split(data, tridx, teidx)

    return train, test  


import pickle

import tables



def load_hdf5(fname):
    '''
    load data from hdf5
    :param fname:
    :return:
    '''
    with tables.open_file(fname) as h5f:
        data = h5f.root.image[:]
    return data


def save_hdf5(data, ofname, compress=False, complevel=5, complib='zlib'):
    '''
    Save data to hdf5 file
    :param data:
    :param ofname:
    :param compress:
    :param complevel:
    :param complib:
    :return:
    '''
    with tables.open_file(ofname, 'w') as h5f:
        atom = tables.Atom.from_dtype(data.dtype)
        shape = data.shape
        if compress:
            filters = tables.Filters(complevel=complevel, complib=complib)
            ca = h5f.create_carray(h5f.root, 'image', atom, shape, filters=filters)
        else:
            ca = h5f.create_carray(h5f.root, 'image', atom, shape)
        ca[:] = data[:]


def load_pickle(f):
    '''
    Load a pickle file and return the contents
    :param f: filename
    :return: data from filename
    '''
    inputfile = open(f, 'rb')
    loaded = pickle.load(inputfile)
    inputfile.close()
    return loaded


def dump_pickle(a, f):
    '''
    Save data to a pickle file.
    :param a: data
    :param f: filename
    :return:
    '''
    output = open(f, 'wb')
    pickle.dump(a, output, protocol=pickle.HIGHEST_PROTOCOL)
    output.close()



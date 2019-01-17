import numpy as np
import os
from utilities import noteToHz
from scipy.sparse import coo_matrix, vstack, save_npz, load_npz

def load_path(bins_per_octave, fmin, fmax, sample_rate, thresh):
    name = "bins_per_octave={}&fmin={}&fmax={}&sample_rate={}&thresh={}.npz".format(bins_per_octave,
                                                                                    fmin,
                                                                                    fmax,
                                                                                    sample_rate,
                                                                                    thresh)
    if not os.path.exists(os.path.join("..", "stored_train")):
        os.mkdir(os.path.join("..", "stored_train"))
    elif os.path.exists(os.path.join("..", "stored_train", name)):
        #self.sparse_ker = load_npz(os.path.join("..", "stored_train", "cqt_sparse_ker.npz"))
        loading = True

def load_path(bins_per_octave, fmin, fmax, sample_rate, thresh):
    pass

class CQT:
    def __init__(self, bins_per_octave=24, fmin=34, fmax=22050, sample_rate=44100, thresh=0.0054):
        """
        """
        # TODO: Set fmin to None
        if fmin == None:
            # Set minimum freq to C0 by default
            fmin = noteToHz("C0")

        # TODO: Store on the basis of params
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate
        loading = False
        # if not os.path.exists(os.path.join("..", "stored_train")):
        #     os.mkdir(os.path.join("..", "stored_train"))
        # elif os.path.exists(os.path.join("..", "stored_train", "cqt_sparse_ker.npz")):
        #     self.sparse_ker = load_npz(os.path.join("..", "stored_train", "cqt_sparse_ker.npz"))
        #     self.sparse_ker.todense()
        #     loading = True
        Q = 1.0/(2.0**(1.0/bins_per_octave - 1))
        K = int(np.ceil(bins_per_octave*np.log2(fmax/fmin))) # Try out floor or round
        fft_len = int(2.0**np.ceil(np.log2(Q*sample_rate/fmin)))
        if loading:
            self.fft_len = fft_len
            return
        temp_ker = np.zeros((fft_len, 1))
        sparse_ker = []

        for k in range(K, 0, -1):
            # if k%10:
            #     print(str(100-100*k/K)+"% done")
            len = np.ceil(Q*sample_rate/(fmin*2.0**((k-1.0)/bins_per_octave)))
            temp_ker = (np.hamming(len)/len)*np.exp(2*np.pi*(1j)*Q*np.arange(len)/len)
            ker = np.fft.fft(temp_ker, fft_len)
            ker[ker<=thresh] = 0
            sparse_ker += [coo_matrix(ker, dtype=np.complex128)]
        sparse_ker.reverse()
        sparse_ker = vstack(sparse_ker).tocsc().transpose().conj() / fft_len
        save_npz(os.path.join("..", "stored_train", "cqt_sparse_ker.npz"), sparse_ker)
        self.sparse_ker = sparse_ker
        self.num_freqs = sparse_ker.shape[0]
        self.fft_len = fft_len
    #print 'Initialized OK.'

    def specgram(self, data, sample_rate=44100, hop_length=5000):
        print("starting spectrogram")
        num_time_frames = int(np.ceil(len(data)/hop_length)-1)

        # We're returning a spectrogram of size
        # Number of Frequencies by Number of Time Frames
        # (i.e. we've essentially quantized the freq-time space)
        specgram = np.zeros((225, num_time_frames))

        for t in range(num_time_frames):
            print("{}%".format(t/num_time_frames))
            sample_index = t * hop_length
            specgram[:, t] = (np.fft.fft(data, self.fft_len).reshape(1, self.fft_len) * self.sparse_ker)[0]
            #abs(self.sparse_ker * np.fft.fft(data[sample_index:sample_index + self.fft_len]))
            #(np.fft.fft(data, self.fft_len).reshape(1, self.fft_len) * self.sparse_ker)[0]
        #print("Specshape ES LA {}".format(specgram.shape))
        print(specgram)
        return specgram
        # out = np.zeros((225, 1))
        # for data_hop in range(int(len(data)/hop_length)):
        #     print(data_hop)
        #     print(out.shape)
        #     john = (np.fft.fft(data[data_hop:data_hop + hop_length], self.fft_len).reshape(1, self.fft_len) * self.sparse_ker)[0]
        #     print(john.shape)
        #     out = np.vstack((out, john))
        # print("OUT SHAPE {}".format(out.shape))
        # return out

# length = 44100
# x = np.random.random(length)
# cqt = CQT()
# print(cqt.constQ(x))
# print("Done")

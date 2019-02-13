# rate, data = audioToData('../sample_data/odetojoy.wav')
# if data.ndim > 1:
#     data = data[:,0]
# time = np.arange(len(data))*1.0/rate
#
# plt.plot(time,data)
# plt.show()
#
# print("The shape of the data is {}".format(data.shape))
# f, t, Zxx = signal.stft(data, fs=rate)
# print(f.shape)
# print("T is {}".format(t.shape))
# print(Zxx.shape)
# plt.pcolormesh(t, f, np.abs(Zxx))
# plt.show()
#
# spec_data = CQT().specgram(data) # TODO: Give cqt sample rate
#
# plt.imshow(spec_data, aspect='auto', cmap='inferno', origin='lower')
# plt.show()

# nfft = 2**12
# pxx, freq, bins, plot = plt.specgram(data, NFFT=nfft, Fs=2, cmap="inferno")
# plt.show()

#print("The shape post_cqt is {}".format(post_cqt.shape))
#f, t, Sxx = signal.spectrogram(post_cqt, rate)
#print(f.shape)
#print("New t is {}".format(t.shape))
#print(Sxx.shape)

#pxx, freq, bins, plot = plt.specgram(Sxx, NFFT=nfft, Fs=2, cmap="inferno")
#plt.show()
#plt.pcolormesh(t, f, np.abs(Sxx))
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()

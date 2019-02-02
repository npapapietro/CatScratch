from scipy.io import wavfile



hello = ["hello{}.wav".format(x+1) for x in range (3)]
gbye = ["goodbye{}.wav".format(x+1) for x in range (3)]

sf, snd = wavfile.read("CatScratch/"+hello[0])
print(sf)

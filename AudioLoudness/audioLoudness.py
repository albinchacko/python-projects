import soundfile as sf
import pyloudnorm as pyln

data, rate = sf.read("mixkit-retro-game-emergency-alarm-1000.wav") # load audio (with shape (samples, channels))
print(data.shape)
meter = pyln.Meter(rate) # create BS.1770 meter
loudness = meter.integrated_loudness(data) # measure loudness
print(loudness)

# loudness normalize audio to -12 dB LUFS
loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -10)

sf.write("mixkit-retro-game-emergency-alarm-1000.wav", loudness_normalized_audio, rate)
from time import time
from utils import make_batch
from models import WaveNet, Generator
from IPython.display import Audio
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

inputs, targets = make_batch('./voice.wav')
output_path = './output.wav'
num_time_samples = inputs.shape[1]
num_channels = 1
gpu_fraction = 1
# sample_rate = 44100
sample_rate = 32000

y, sr = librosa.load('./voice.wav', duration=5.0)
print(y.shape)
print(sr)
# librosa.output.write_wav('./file_trim_5s.wav', y, sr)



model = WaveNet(num_time_samples=num_time_samples, num_channels=num_channels, gpu_fraction=gpu_fraction)

print('inputs.shape = ', inputs.shape)
Audio(inputs.reshape(inputs.shape[1]), rate=44100)
print('inputs.shape = ', inputs.shape)
print('targets.shape = ', targets.shape)

tic = time()
model.test(inputs, targets)
toc = time()
print('Training time = {} seconds'.format(toc-tic))

generator = Generator(model)

input_ = inputs[:, 0:1, 0]

tic = time()
predictions = generator.run(input_, sample_rate)
Audio(predictions, rate=44100)


# print('prediction.shape = ', predictions.shape)
# predictions_output = predictions[0, :]
# print('predictions_output.shape=', predictions_output.shape)
# toc = time()
# print('Sampling time = {} seconds'.format(toc-tic))
# print('Completed sampling rate = {}'.format(predictions.shape[1]))

# librosa.output.write_wav(output_path, predictions[0, :], sample_rate)
# wav_out = predictions_output.astype(dtype=np.float16)
# librosa.output.write_wav(output_path, wav_out, 22050)
# print('Output signal saved to {} successfully'.format(output_path))


# from scipy.io import wavfile
# # wav_out = np.asarray(predictions_output)
# wav_out = predictions.astype(dtype=np.int16)

# wavfile.write('./testout.wav', 44100, wav_out)

# librosa.display.waveplot(predictions[0, :], sr=sample_rate)
# plt.show()

# X = librosa.stft(predictions[0, :])
# Xdb = librosa.amplitude_to_db(abs(X))
# librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')

# plt.show()


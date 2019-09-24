from time import time
from utils import make_batch
from models import WaveNet, Generator
from IPython.display import Audio

inputs, targets = make_batch('./voice.wav')
num_time_samples = inputs.shape[1]
num_channels = 1
gpu_fraction = 1

model = WaveNet(num_time_samples=num_time_samples, num_channels=num_channels, gpu_fraction=gpu_fraction)

Audio(inputs.reshape(inputs.shape[1]), rate=44100)

tic = time()
model.train(inputs, targets)
toc = time()
print('Training time = {} seconds'.format(toc-tic))

# generator = Generator(model)

# input_ = inputs[:, 0:1, 0]

# tic = time()
# predictions = generator.run(input_, 32000)
# toc = time()

# print('Generating time = {} seconds'.format(toc-tic))


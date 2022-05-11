# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import mmwave as mm
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
from mmwave.tracking import EKF
from mmwave.tracking import gtrack_visualize
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import h5py
import sys

# get the mat file from cli argv
f = h5py.File(sys.argv[1],'r')

# Radar specific parameters
NUM_RX = int(f.get('/adcRawData/dim/numRxChan')[0][0])
VIRT_ANT = 8

# Data specific parameters
NUM_CHIRPS = int(f.get('/adcRawData/dim/numChirpsPerFrame')[0][0])
NUM_ADC_SAMPLES = int(f.get('/adcRawData/dim/numSamples')[0][0])
RANGE_RESOLUTION = f.get('/adcRawData/rfParams/rangeResolutionsInMeters')[0][0]
DOPPLER_RESOLUTION = f.get('/adcRawData/rfParams/dopplerResolutionMps')[0][0]
NUM_FRAMES = int(f.get('/adcRawData/dim/numFrames')[0][0])

# DSP processing parameters
SKIP_SIZE = 4
ANGLE_RES = 1
ANGLE_RANGE = 90
ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
BINS_PROCESSED = 112

# Read in adc data file
offset = 0
FRAME_LENGTH = NUM_CHIRPS * NUM_RX * NUM_ADC_SAMPLES * 2
adc_data = np.zeros(NUM_FRAMES * FRAME_LENGTH, dtype=np.uint16)
for x in f.get('/adcRawData/data'):
    dataset = f[x[0]]
    dataset.read_direct(adc_data, dest_sel=np.s_[offset:offset+FRAME_LENGTH])
    offset += FRAME_LENGTH

# separate adc_data into frames
adc_data = adc_data.reshape(NUM_FRAMES, -1)

# create `all_data` which formats data into complex samples
def organize(raw_frame, num_chirps, num_rx, num_samples):
    ret = np.zeros(len(raw_frame) // 2, dtype=complex)
    for i in range(num_rx):
        ret[i::num_rx] = raw_frame[i::2*num_rx] + 1j * raw_frame[i+num_rx::2*num_rx]
    return ret.reshape((num_chirps, num_samples, num_rx)).swapaxes(1,2)
all_data = np.apply_along_axis(organize, 1, adc_data, num_chirps=NUM_CHIRPS, num_rx=NUM_RX, num_samples=NUM_ADC_SAMPLES)

# assert that sizes match what is expected
assert(len(all_data) == NUM_FRAMES)
assert(len(all_data[0]) == NUM_CHIRPS) # in the sample code, NUM_CHIRPS is doubled for some reason...
assert(len(all_data[0][0]) == NUM_RX)
assert(len(all_data[0][0][0]) == NUM_ADC_SAMPLES)

# Start DSP processing
range_azimuth = np.zeros((ANGLE_BINS, BINS_PROCESSED))
num_vec, steering_vec = dsp.gen_steering_vec(ANGLE_RANGE, ANGLE_RES, VIRT_ANT)
tracker = EKF()

for frame in all_data:
    """ 1 (Range Processing) """

    # --- range fft
    radar_cube = dsp.range_processing(frame)

    """ 2 (Capon Beamformer) """
    # --- static clutter removal
    mean = radar_cube.mean(0)
    radar_cube = radar_cube - mean

    # --- capon beamforming
    beamWeights   = np.zeros((VIRT_ANT, BINS_PROCESSED), dtype=np.complex_)
    radar_cube = np.concatenate((radar_cube[0::2, ...], radar_cube[1::2, ...]), axis=1)
    # Note that when replacing with generic doppler estimation functions, radarCube is interleaved and
    # has doppler at the last dimension.
    for i in range(BINS_PROCESSED):
        range_azimuth[:,i], beamWeights[:,i] = dsp.aoa_capon(radar_cube[:, :, i].T, steering_vec, magnitude=True)

    """ 3 (Object Detection) """
    heatmap_log = np.log2(range_azimuth)

    # --- cfar in azimuth direction
    first_pass, _ = np.apply_along_axis(
        func1d=dsp.ca_,
        axis=0,
        arr=heatmap_log,
        l_bound=1.5,
        guard_len=4,
        noise_len=16
    )

    # --- cfar in range direction
    second_pass, noise_floor = np.apply_along_axis(
        func1d=dsp.ca_,
        axis=0,
        arr=heatmap_log.T,
        l_bound=2.5,
        guard_len=4,
        noise_len=16
    )

    # --- classify peaks and caclulate snrs
    noise_floor = noise_floor.T
    first_pass = (heatmap_log > first_pass)
    second_pass = (heatmap_log > second_pass.T)
    peaks = (first_pass & second_pass)
    peaks[:SKIP_SIZE, :] = 0
    peaks[-SKIP_SIZE:, :] = 0
    peaks[:, :SKIP_SIZE] = 0
    peaks[:, -SKIP_SIZE:] = 0
    pairs = np.argwhere(peaks)
    azimuths, ranges = pairs.T
    snrs = heatmap_log[pairs[:,0], pairs[:,1]] - noise_floor[pairs[:,0], pairs[:,1]]

    """ 4 (Doppler Estimation) """
    # --- get peak indices
    # beamWeights should be selected based on the range indices from CFAR.
    dopplerFFTInput = radar_cube[:, :, ranges]
    beamWeights  = beamWeights[:, ranges]

    # --- estimate doppler values
    # For each detected object and for each chirp combine the signals from 4 Rx, i.e.
    # For each detected object, matmul (numChirpsPerFrame, numRxAnt) with (numRxAnt) to (numChirpsPerFrame)
    dopplerFFTInput = np.einsum('ijk,jk->ik', dopplerFFTInput, beamWeights)
    if not dopplerFFTInput.shape[-1]:
        continue
    dopplerEst = np.fft.fft(dopplerFFTInput, axis=0)
    dopplerEst = np.argmax(dopplerEst, axis=0)
    dopplerEst[dopplerEst[:]>=NUM_CHIRPS/2] -= NUM_CHIRPS

    """ 5 (Extended Kalman Filter) """
    # --- convert bins to units
    ranges = ranges * RANGE_RESOLUTION
    azimuths = (azimuths - (ANGLE_BINS // 2)) * (np.pi / 180)
    dopplers = dopplerEst * DOPPLER_RESOLUTION
    snrs = snrs

    # --- put into EKF
    tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
    targetDescr, tNum = tracker.step()

    """ 6 (Visualize Output) """
    frame = gtrack_visualize.get_empty_frame()
    try:
        frame = gtrack_visualize.update_frame(targetDescr, int(tNum[0]), frame)
    except:
        pass
    frame = gtrack_visualize.draw_points(tracker.point_cloud, len(ranges), frame)
    if not gtrack_visualize.show(frame, wait=1):
        break

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

import re

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import librosa
import pandas as pd

from scipy.interpolate import interp1d

import pyrubberband as pyrb

#bad practice, but will suffice because of the deadline
import sys
sys.path.append('..')

import lib.midi as midi
import pretty_midi

import pdb


def map_score(perf):
    """ associate a performance midi with a kern score based on filename conventions """
    regex = re.compile('(\d\d\d)_bwv(\d\d\d)(f|p)')
    #info = regex.search(perf)
    #num, bwv, part = info.group(1,2,3)
    #bwv = int(bwv)
    #book = 1 + int(bwv > 869)
    #score = 'wtc{}{}{:02d}'.format(book,part,bwv - 845 - (book-1)*24)

    return '{}'.format(perf)

def plot_events(ax, events, stride=512, num_windows=2000):
    timings = np.cumsum(events[:,-1])
    x = np.zeros([num_windows,128])
    for i in range(num_windows):
        time = (stride*i)/44100.
        k = np.argmin(time>=timings)
        x[i] = events[k,:128]

    ax.imshow(x.T[::-1][30:90], interpolation='none', cmap='Greys', aspect=num_windows/250)

def colorplot(ax, x, y, aspect=4):
    cmap = colors.ListedColormap(['white','red','orange','black'])
    bounds = [0,1,2,3,4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(x.T*2 + y.T, interpolation='none', cmap=cmap, aspect=aspect, norm=norm)

def pianoroll(events, fs=44100, stride=512):
    notes = events[:,:-1]
    timing = np.cumsum(events[:,-1])
    num_windows = int(timing[-1]*(44100./stride))+1
    
    x = np.zeros([num_windows,128])
    for i in range(num_windows):
        t = (i*stride)/fs
        x[i] = notes[np.argmin(t>timing)]
        
    return x

        
def pscore(score, alignment, stride=512, start=False):
    epsilon = 1e-4
    notes = score[:,:-1]
    score_time, perf_time = zip(*alignment)
    num_windows = int(alignment[-1][1]*(44100./stride))+1
    
    x = np.zeros([num_windows,128])
    for i in range(num_windows):
        t = (i*stride)/44100.                           # time (in seconds) in the performance
        if start:                                       # if start time is given
            if t < perf_time[0]: continue

        j = np.argmin(t>np.array(perf_time))            # index of the first event in performance that ends after time t
        s = score_time[j]                               # time (in beats) in the score
        if s > np.sum(score[:,-1]): continue
        k = np.argmin(s>np.cumsum(score[:,-1])+epsilon) # index of the first event in score that ends after time s
        x[i] = notes[k]
    
    return x

#we just hardcode the program to 0.
def midi_gt_performance(gt_alignment, score_midi_file):
    gt_seconds_map = np.loadtxt(gt_alignment)
    score2perf = interp1d(gt_seconds_map[:, 0], gt_seconds_map[:, 1])
    
    pm = pretty_midi.PrettyMIDI(score_midi_file)
    
    for instrument in pm.instruments:
        for note in instrument.notes:
            note_dur = note.end - note.start
            note.start = score2perf(note.start)
            note.end = note.start + note_dur
    
    return pm
    
# This function just puts things on different sides of the stereo image, whether they
#are written to disk or not depends on the caller.
def sonic_gt_evaluation(gt_alignment, perf_audio, perf_midi_file, score_midi_file):
    performance_audio = librosa.load(perf_audio, mono=True, sr=44100) #turn to mono
    gt_seconds_map = np.loadtxt(gt_alignment)
    
    #load both performance and score midi
    perf_events,perf_start,perf_end = midi.load_midi_events(perf_midi_file, strip_ends=False)
    score_events,score_start,score_end = midi.load_midi_events(score_midi_file, strip_ends=False)
      
    ground_prettymidi = midi_gt_performance(gt_alignment, score_midi_file)

    #sonify groundroll with fluidsynth and pretty_midi
    sonified_gt = ground_prettymidi.fluidsynth()
    
    #truncate based on the known interpolation bounds, which are the perf start and end
    sonified_gt = sonified_gt[int(perf_start*44100):int(perf_end*44100)]
    
    #truncate it according to where the performance should start, since the gt is basically the score mapped to the performance
    #sonified_gt = sonified_gt[int(perf_start*44100):]
    
    #truncate performance to remove any trailing or leading silences
    truncated_performance = performance_audio[0][int(perf_start*44100):int(perf_end*44100)]
    
    #should they start the same or end the same?
    #I believe we are more certain about the end.. so let's pad the shorter one to match the longer one, assuming that they should end the same.
    
    #padded_performance = np.zeros(len(sonified_gt))
    #begin = abs(len(truncated_performance) - len(sonified_gt))
    #padded_performance[begin:] = truncated_performance
    
    stereo_sonification = np.stack([sonified_gt, truncated_performance], axis=0)
    
    stereo_sonification = librosa.util.normalize(stereo_sonification, axis=1)
    
    return stereo_sonification
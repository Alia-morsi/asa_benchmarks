import sys
import numpy as np
import librosa, pretty_midi
import lib.midi as midi
import lib.util as util
import pandas as pd
from lib.pitchclass_mctc.wrappers import PitchClassCTC
import vamp
import madmom
import soundfile as sf

from essentia.standard import HPCP
from essentia.pytools.spectral import hpcpgram


def interpolate_ground_truth(score_midi, perf, score_beat_annotation='', perf_beat_annotation='', fs=44100, stride=512, lmbda=0.1):
    score_beat_annot_df =  pd.read_csv(score_beat_annotation, delimiter='\t', header=None)
    perf_beat_annot_df =  pd.read_csv(perf_beat_annotation, delimiter='\t', header=None)
    
    if len(score_beat_annot_df) != len(perf_beat_annot_df):
        return []
        
    score_notes, score_tpb = midi.load_midi(score_midi)
    score_note_onsets = list(zip(*score_notes))[1]
    
    #The performance midi is only loaded to get the performance start time
    perf_events,perf_start,perf_end = midi.load_midi_events(perf + '.midi')
    
    interpolated_performance = np.interp(score_note_onsets, score_beat_annot_df[0], perf_beat_annot_df[0])
    interpolated_gt_annotation = np.array(list(zip(score_note_onsets, interpolated_performance)))
    
    #not sure why we have this statement, but it could be for the interpolation success
    return np.insert(interpolated_gt_annotation, 0, (score_notes[0][1], perf_start), axis=0)

def align_chroma(score_midi, perf, fs=44100, stride=512, n_fft=4096):
    score_synth = pretty_midi.PrettyMIDI(score_midi).fluidsynth(fs=fs)
    perf,_ = librosa.load(perf + '.wav', sr=fs)
    score_chroma = librosa.feature.chroma_stft(y=score_synth, sr=fs, tuning=0, norm=2,
                                               hop_length=stride, n_fft=n_fft)
    score_logch = librosa.power_to_db(score_chroma, ref=score_chroma.max())
    perf_chroma = librosa.feature.chroma_stft(y=perf, sr=fs, tuning=0, norm=2,
                                              hop_length=stride, n_fft=n_fft)
    perf_logch = librosa.power_to_db(perf_chroma, ref=perf_chroma.max())
    D, wp = librosa.sequence.dtw(X=score_logch, Y=perf_logch)
    path = np.array(list(reversed(np.asarray(wp))))

    return np.array([(s,t) for s,t in dict(reversed(wp)).items()])*(stride/fs)


def align_spectra(score_midi, perf, fs=44100, stride=512, n_fft=4096):
    score_synth = pretty_midi.PrettyMIDI(score_midi).fluidsynth(fs=fs)
    perf,_ = librosa.load(perf + '.wav', sr=fs)
    score_spec = np.abs(librosa.stft(y=score_synth, hop_length=stride, n_fft=n_fft))**2
    score_logspec = librosa.power_to_db(score_spec, ref=score_spec.max())
    perf_spec = np.abs(librosa.stft(y=perf, hop_length=stride, n_fft=n_fft))**2
    perf_logspec = librosa.power_to_db(perf_spec, ref=perf_spec.max())
    D, wp = librosa.sequence.dtw(X=score_logspec, Y=perf_logspec)
    path = np.array(list(reversed(np.asarray(wp))))

    return np.array([(s,t) for s,t in dict(reversed(wp)).items()])*(stride/fs)


def align_hpcp(score_midi, perf, fs=44100, stride=2048, n_fft=4096):
    score_synth = pretty_midi.PrettyMIDI(score_midi).fluidsynth(fs=fs)
    perf,_ = librosa.load(perf + '.wav', sr=fs)
    score_hpcpgram = hpcpgram(score_synth, sampleRate=fs, hopSize=2048, frameSize=4096)
    perf_hpcpgram = hpcpgram(perf, sampleRate=fs)
    
    #why do they use this power_to_db?
    D, wp = librosa.sequence.dtw(X=score_hpcpgram.T, Y=perf_hpcpgram.T)
    path = np.array(list(reversed(np.asarray(wp))))
    
    return np.array([(s,t) for s,t in dict(reversed(wp)).items()])*(stride/fs)

def align_dce(score_midi, perf, fs=44100, stride=512, n_fft=4096):
    score_synth = madmom.audio.signal.Signal(pretty_midi.PrettyMIDI(score_midi).fluidsynth(fs=fs), sample_rate=fs)                 
    perf = madmom.audio.signal.Signal(perf + '.wav', sample_rate=fs)                           
    
    #sf.write('tmp.wav', score_synth, fs)
    dcp = madmom.audio.chroma.DeepChromaProcessor()
    
    score_chroma = dcp(score_synth)
    perf_chroma = dcp(perf)                                 
    
    D, wp = librosa.sequence.dtw(X=score_chroma.T, Y=perf_chroma.T)
    path = np.array(list(reversed(np.asarray(wp))))
    
    return np.array([(s, t) for s,t in dict(reversed(wp)).items()]) * 4410/44100
    

def align_nnls(score_midi, perf, fs=44100, stride=2048, n_ffg=4096):
    score_synth = pretty_midi.PrettyMIDI(score_midi).fluidsynth(fs=fs)
    perf,_ = librosa.load(perf + '.wav', sr=fs)
    
    score_chroma = vamp.collect(score_synth, fs, "nnls-chroma:nnls-chroma", step_size=2045, block_size=4096, output="chroma")['matrix'][1]
    perf_chroma = vamp.collect(perf, fs, "nnls-chroma:nnls-chroma", step_size=2048, block_size=4096, output="chroma")['matrix'][1]
    
    D, wp = librosa.sequence.dtw(X=score_chroma.T, Y=perf_chroma.T)
    path = np.array(list(reversed(np.asarray(wp))))
    #I might need to stack the output, if it returns separate thiings for trebble and bass
    
    return np.array([(s,t) for s,t in dict(reversed(wp)).items()])*(2048/44100)


def align_ctc_chroma(score_midi, perf, fs=44100, stride=512, n_fft=4096):
    score_synth = pretty_midi.PrettyMIDI(score_midi).fluidsynth(fs=fs)
    perf,_ = librosa.load(perf + '.wav', sr=fs)
    pclass_extractor = PitchClassCTC()
    perf_hopsize_cqt, perf_fs, perf_pclass = pclass_extractor(perf, fs)
    score_hopsize_cqt, score_fs, score_synth_pclass = pclass_extractor(score_synth, fs)
    
    #if by default they aren't the same sampling rate and hop size, then some manual intervention needs to be done.
    if perf_hopsize_cqt != score_hopsize_cqt or score_fs != perf_fs:
        print('mismatch!')
        return []
    
    D, wp = librosa.sequence.dtw(X=score_synth_pclass, Y=perf_pclass)
    path = np.array(list(reversed(np.asarray(wp))))

    return np.array([(s,t) for s,t in dict(reversed(wp)).items()])*(perf_hopsize_cqt/perf_fs)
    
    
def align_salience(score_midi, perf, fs=44100, stride=512, n_fft=4096):
    return

def align_prettymidi(score_midi, perf, fs=22050, hop=512, note_start=36, n_notes=48, penalty=None):
    '''
    Align a MIDI object in-place to some audio data.
    Parameters
    ----------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing some MIDI content
    audio_data : np.ndarray
        Samples of some audio data
    fs : int
        audio_data's sampling rate, and the sampling rate to use when
        synthesizing MIDI
    hop : int
        Hop length for CQT
    note_start : int
        Lowest MIDI note number for CQT
    n_notes : int
        Number of notes to include in the CQT
    penalty : float
        DTW non-diagonal move penalty
    '''
    def extract_cqt(audio_data, fs, hop, note_start, n_notes):
        '''
        Compute a log-magnitude L2-normalized constant-Q-gram of some audio data.
        Parameters
        ----------
        audio_data : np.ndarray
            Audio data to compute CQT of
        fs : int
            Sampling rate of audio
        hop : int
            Hop length for CQT
        note_start : int
            Lowest MIDI note number for CQT
        n_notes : int
            Number of notes to include in the CQT
        Returns
        -------
        cqt : np.ndarray
            Log-magnitude L2-normalized CQT of the supplied audio data.
        frame_times : np.ndarray
            Times, in seconds, of each frame in the CQT
        '''
        # Compute CQT
        cqt = librosa.cqt(
            audio_data, sr=fs, hop_length=hop,
            fmin=librosa.midi_to_hz(note_start), n_bins=n_notes)
        # Transpose so that rows are spectra
        cqt = cqt.T
        # Compute log-amplitude
        cqt = librosa.amplitude_to_db(librosa.magphase(cqt)[0], ref=cqt.max())
        # L2 normalize the columns
        cqt = librosa.util.normalize(cqt, norm=2., axis=1)
        # Compute the time of each frame
        times = librosa.frames_to_time(np.arange(cqt.shape[0]), fs, hop)
        return cqt, times

    audio_data, _ = librosa.load(perf + '.wav', fs)
    midi_object = pretty_midi.PrettyMIDI(score_midi)
    # Get synthesized MIDI audio
    midi_audio = midi_object.fluidsynth(fs=fs)
    # Compute CQ-grams for MIDI and audio
    midi_gram, midi_times = extract_cqt(
        midi_audio, fs, hop, note_start, n_notes)
    audio_gram, audio_times = extract_cqt(
        audio_data, fs, hop, note_start, n_notes)
    # Compute distance matrix; because the columns of the CQ-grams are
    # L2-normalized we can compute a cosine distance matrix via a dot product
    distance_matrix = 1 - np.dot(midi_gram, audio_gram.T)
    D, wp = librosa.sequence.dtw(C=distance_matrix)
    path = np.array([(s,t) for s,t in dict(reversed(wp)).items()])
    result = [(midi_times[x[0]], audio_times[x[1]]) for x in path]
    return np.array(result)


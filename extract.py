#!/usr/bin/python3
import os,sys,errno,csv,re
import pandas as pd
import lib.midi as midilib
import lib.util as util
import shutil
import soundfile as sf

from scipy.io import wavfile

hardcoded_bwvs = {
    'Prelude and Fugue in G Major, WTC I' : 860,
    'Prelude and Fugue in G Minor, WTC II' : 885,
    'Prelude and Fugue in F-sharp Minor, WTC II' : 883,
    'Prelude and Fugue in F Minor, WTC I': 857,
    'Prelude and Fugue in F Major, WTC II': 880,
    'Prelude and Fugue in E-flat Major, WTC I': 876,
    'Prelude and Fugue in D major, WTC I': 850,
    'Prelude and Fugue in D Minor, Book II': 875,
    'Prelude and Fugue in D Major, Book II': 874,
    'Prelude and Fugue in C-sharp Minor, WTC II': 873,
    'Prelude and Fugue in C-sharp Minor, WTC I': 849,
    'Prelude and Fugue in C-sharp Major, WTC I': 848,
    'Prelude and Fugue in B-flat Minor, WTC II': 867,
    'Prelude and Fugue in B-Moll, No. 22 WTKII': 893,
    'Prelude and Fugue in B Minor, WTC II': 893,
    'Prelude and Fugue in A-flat Major, WTC I': 862,
}

exclude = [
    '2011/MIDI-Unprocessed_22_R1_2011_MID--AUDIO_R1-D8_12_Track12_wav.midi', # incomplete

    '2011/MIDI-Unprocessed_19_R1_2011_MID--AUDIO_R1-D7_12_Track12_wav.midi',
    '2011/MIDI-Unprocessed_25_R1_2011_MID--AUDIO_R1-D9_14_Track14_wav.midi',
    '2017/MIDI-Unprocessed_067_PIANO067_MID--AUDIO-split_07-07-17_Piano-e_3-03_wav--1.midi',
    '2013/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_2013_wav--1.midi',
    '2004/MIDI-Unprocessed_XP_03_R1_2004_01-02_ORIG_MID--AUDIO_03_R1_2004_01_Track01_wav.midi',
    '2008/MIDI-Unprocessed_03_R1_2008_01-04_ORIG_MID--AUDIO_03_R1_2008_wav--1.midi',
    '2014/MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_16_R1_2014_wav--1.midi',
    '2011/MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_02_Track02_wav.midi',
    '2008/MIDI-Unprocessed_14_R1_2008_01-05_ORIG_MID--AUDIO_14_R1_2008_wav--1.midi',
    '2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi',
]

partial = {
    '2011/MIDI-Unprocessed_01_R1_2011_MID--AUDIO_R1-D1_03_Track03_wav.midi' : 'f', # just the fugue
    '2011/MIDI-Unprocessed_05_R1_2011_MID--AUDIO_R1-D2_08_Track08_wav.midi' : 'p', # just the prelude
    '2011/MIDI-Unprocessed_05_R1_2011_MID--AUDIO_R1-D2_09_Track09_wav.midi' : 'f', # just the fugue
}

def extract(basename, score_root, data, notes, ticks_per_beat):
    scorename = util.map_score(basename)
    print('Writing {} (associated score {})'.format(basename, scorename))
    midilib.write_midi(basename + '.midi', notes, ticks_per_beat)
    wavfile.write(basename + '.wav', fs, data)
    score = os.path.join(score_root, scorename + '.krn')
    target = 'data/score/{}'.format(scorename + '.midi')
    os.system('hum2mid {} -o {}'.format(score, target))
    

def restructure_asap_files(asap_root, metadata_path):
    asap_metadata_df = pd.read_csv(os.path.join(asap_root, metadata_path))
    #for now, just filter the bachs
    #bach_subset_df = asap_metadata_df[asap_metadata_df['composer'] == 'Bach']
    
    #delete any rows that have audio_performance set to nan
    #bach_subset_df = bach_subset_df[bach_subset_df['audio_performance'].notnull()]
    asap_metadata_df = asap_metadata_df[asap_metadata_df['audio_performance'].notnull()]
    
    #this function should overwrite if exists also.
    #for index, row in bach_subset_df.iterrows():
    for index, row in asap_metadata_df.iterrows():
        basename = row['title']
        performer = os.path.split(row['midi_performance'])[1]
        
        #modify name to avoid overlaps between performances of the same score
        basename = 'asap-{}-{}'.format(basename, performer[:len('.mid')])
        
        shutil.copy(os.path.join(asap_root, row['midi_score']), 'data/score/{}'.format(basename + '.midi'))
        shutil.copy(os.path.join(asap_root, row['midi_score_annotations']), 'data/score/{}'.format(basename + '.txt'))
                    
        shutil.copy(os.path.join(asap_root, row['midi_performance']), 'data/perf/{}'.format(basename + '.midi'))
        shutil.copy(os.path.join(asap_root, row['performance_annotations']), 'data/perf/{}'.format(basename + '.txt'))
        shutil.copy(os.path.join(asap_root, row['audio_performance']), 'data/perf/{}'.format(basename + '.wav')) 
        

#we'll just pass in the default paths
def sonify_interpolated_gt():
    os.makedirs('eval/sonic', exist_ok=True)
    
    interpolation_basepath = 'align/ground-beat-interpol'
    gt_alignments = sorted([f for f in os.listdir(interpolation_basepath) if f.endswith('.txt')])
    
    #parallel to the interpolated ground truths  
    perf_audios = ['{}.wav'.format(os.path.join('data/perf', gt_alignment[:-len('.txt')])) for gt_alignment in gt_alignments]
    perf_midis = ['{}.midi'.format(os.path.join('data/perf', gt_alignment[:-len('.txt')])) for gt_alignment in gt_alignments]  
    score_midis = ['{}.midi'.format(os.path.join('data/score', gt_alignment[:-len('.txt')])) for gt_alignment in gt_alignments]
    
    for gt_alignment, perf_audio, perf_midi_file, score_midi_file in zip(gt_alignments, perf_audios, perf_midis, score_midis):
        if not os.path.exists(perf_audio) or not os.path.exists(perf_midi_file) or not os.path.exists(score_midi_file):
            continue
            
        stereo_sonification = util.sonic_gt_evaluation(os.path.join(interpolation_basepath, gt_alignment), perf_audio, perf_midi_file, score_midi_file)
        if not len(stereo_sonification):
            print('Skipping {} \n'.format(outfile))
            continue
        outfile = os.path.join('eval/sonic/', '{}'.format(gt_alignment[:-len('.txt')] + '.wav'))
        print('Writing {} \n'.format(outfile))
        sf.write(outfile, stereo_sonification.T, 44100)
    return


if __name__ == "__main__":
    os.makedirs('data/perf',exist_ok=True)
    os.makedirs('data/score',exist_ok=True)

    oneup = 0
    score_root = os.path.join(sys.argv[1],'kern')
    root = sys.argv[2]
    with open(os.path.join(root,'maestro-v2.0.0.csv')) as f:
        index = csv.reader(f)

        wtc = re.compile('Prelude and F')
        bwv = re.compile('BWV (\d*)')
        for row in index:
            if row[0] != 'Johann Sebastian Bach': continue # not bach
            if not wtc.search(row[1]): continue # not wtc
            if row[4] in exclude: continue # something wrong with these
        
            identifier = bwv.search(row[1])
            if identifier:
                outfile = 'bwv{}'.format(identifier.group(1))
            else:
                try:
                    outfile = 'bwv{}'.format(hardcoded_bwvs[row[1]])
                except KeyError:
                    print('MISSING:', row[1])

            notes, ticks_per_beat = midilib.load_midi(os.path.join(root,row[4]))
            fs, data = wavfile.read(os.path.join(root,row[5]))

            if row[4] in partial: #special cases
                basename = 'data/perf/{:03d}_{}{}'.format(oneup, outfile, partial[row[4]])
                extract(basename, score_root, data, notes, ticks_per_beat)
                oneup += 1
            else:
                splitpoint = midilib.split(notes)
                pnotes = [n for n in notes if n[1] < splitpoint]
                basename = 'data/perf/{:03d}_{}{}'.format(oneup, outfile, 'p')
                extract(basename, score_root, data[:int(fs*splitpoint)], pnotes, ticks_per_beat)
                oneup += 1
                fnotes = [(n[0],n[1]-splitpoint,n[2]-splitpoint) for n in notes if n[1] >= splitpoint]
                basename = 'data/perf/{:03d}_{}{}'.format(oneup, outfile, 'f')
                extract(basename, score_root, data[int(fs*splitpoint):], fnotes, ticks_per_beat)
                oneup += 1


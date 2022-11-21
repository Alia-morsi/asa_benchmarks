#!/usr/bin/python3
import os,sys,errno,csv,re
import pandas as pd
import lib.midi as midilib
import lib.util as util
import shutil
import soundfile as sf

from scipy.io import wavfile

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
        outfile = os.path.join('eval/sonic/', '{}'.format(gt_alignment[:-len('.txt')] + '.wav'))
        
        if not len(stereo_sonification):
            print('Skipping {} \n'.format(outfile))
            continue
        
        print('Writing {} \n'.format(outfile))
        sf.write(outfile, stereo_sonification.T, 44100)
    return
    


import os, sys, errno, time, multiprocessing
import numpy as np
import argparse

import lib.util as util
import lib.algos as algos

def align_and_save(align, perf, perfdir, score_dir, outdir, kwargs={}):
    perf_transcript = os.path.join(perfdir, perf)
    score = os.path.join(scoredir,util.map_score(perf) + '.midi')
    
    if os.path.isfile(os.path.join(outdir, perf + '.txt')):
        return #to avoid repeating the alignment
    
    alignment = align(score, perf_transcript, *kwargs)
    np.savetxt(os.path.join(outdir, perf + '.txt'), alignment, fmt='%f\t', header='score\t\tperformance')

algo_functions = {
    'spectra' : 'align_spectra',
    'chroma' : 'align_chroma',
    'cqt' : 'align_prettymidi',
    'ctc-chroma': 'align_ctc_chroma',
    'f0-salience': 'align_salience', 
    'hpcpgram': 'align_hpcp', 
    'nnls': 'align_nnls',
    'dce': 'align_dce'
}

data_adapt_functions = {
    'ground-beat-interpol': 'interpolate_ground_truth',
    'rwc_adapt' : '<put_name>'
    }

datasets = [
    'asap-interpolated', 'rwc_adapted'
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Towards ASA Benchmarks')

    parser.add_argument('--data', action='store_true', help='to alternate between alignment mode and dataprep mode. when set, mode is to dataprep')

    parser.add_argument('action', type=str, help='When in alignment mode, set to one of: spectra, chroma, cqt, ctc-chroma, hpcpgram, nnls, dce.\nWhen in dataprep mode, set to ground_beat_interpol')

    parser.add_argument('dataset', type=str, help='dataset name, which should be the same as the folder name. Note that the files must be formatted as expected (See README)')

    args = parser.parse_args()

    data_mode = args.data
    action = args.action
    dataset = args.dataset

    scoredir = os.path.join('data', dataset, 'score')
    perfdir = os.path.join('data', dataset, 'perf')

    #reset vs continue should be an argument, to determine if we should continue or not
    #also skip list should be an argument
        
    outdir = os.path.join('align', dataset, action)
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    start_time = time.time()
    performances = sorted([f[:-len('.midi')] for f in os.listdir(perfdir) if f.endswith('.midi')])

    #if continue:
        #load the sorted performances in the file of algo
        #get the last one  
        #find the index of that last one
        #perhaps that should affect the processing start variable
    
    processing_start = 0
    
    if action == 'ground-beat-interpol':
        score_beat_annotations = sorted([f[:-len('.txt')] for f in os.listdir(scoredir) if f.endswith('.midi')])
        performance_beat_annotations = sorted([f[:-len('.txt')] for f in os.listdir(perfdir) if f.endswith('.midi')])
    
    #this should have a path for if 'not' data
    alignment_algo = getattr(algos, algo_functions[action])

    print('Computing {} alignments'.format(action))
    total = 0 #tracker for the total time
        
    for i in range(0, len(performances)):
        perf = performances[i]
        print('  ', perf, end=' ')
        kwargs = {} #useful for passing extra variables
            
        t0 = time.time()
        perf_path = os.path.join(perfdir, perf)
        score = os.path.join(scoredir,util.map_score(perf) + '.midi')
            
        #wtf is this code.. I think I wanted to keep the same flow at all costs. refactor to have separate paths
        # for the interpolation and the actual alignment.

        if action == 'ground-beat-interpol':
            kwargs['score_beat_annotation'] = os.path.join(scoredir, util.map_score(perf) + '.txt')
            kwargs['perf_beat_annotation'] = os.path.join(perfdir, util.map_score(perf) + '.txt') 
            #I think this shouldn't cause a problem because so far the base for the performances is the same as that of the files.
            
        if os.path.isfile(os.path.join(outdir, perf + '.txt')):
            continue
        
        alignment = alignment_algo(score, perf_path, **kwargs)
            
        if len(alignment) == 0:
            continue
                
        np.savetxt(os.path.join(outdir, perf + '.txt'), alignment, fmt='%f\t', header='score\t\tperformance')

        t1 = time.time()-t0
        print('({} seconds)'.format(t1))
        total += t1

    print('Elapsed time: {} seconds'.format(time.time()-start_time))


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
    'asap', 'asap-interpolated', 'rwc_adapted'
]

if __name__ == "__main__":        
    algo = sys.argv[1]
    dataset = sys.argv[2]
    scoredir = sys.argv[3]
    perfdir = sys.argv[4]

    #reset vs continue should be an argument, to determine if we should continue or not
    #also skip list should be an argument
        
    outdir = os.path.join('align', dataset, algo)
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    start_time = time.time()
    performances = sorted([f[:-len('.midi')] for f in os.listdir(perfdir) if f.endswith('.midi')])

    if continue:
        #load the sorted performances in the file of algo
        #get the last one  
        #find the index of that last one
        #perhaps that should affect the processing start variable
    
    processing_start = 0
    
    #change this to: if data adaptation
    if algo == 'ground-beat-interpol':
        score_beat_annotations = sorted([f[:-len('.txt')] for f in os.listdir(scoredir) if f.endswith('.midi')])
        performance_beat_annotations = sorted([f[:-len('.txt')] for f in os.listdir(perfdir) if f.endswith('.midi')])
    
    alignment_algo = getattr(algos, algo_functions[algo])
    
'''    if parallel > 0:
        print('Computing {} alignments (parallel)'.format(algo))
        args = []
        for i in range(0, len(performances)):
            kwargs = {}
            if algo == 'ground-beat-interpol':
                kwargs['score_beat_annotation'] = score_beat_annotations[i]
                kwargs['performance_beat_annotation'] = performance_beat_annotations[i]
                args.append((alignment_algo, performances[i], perfdir, scoredir, outdir, kwargs))
        #args = [(alignment_algo, perf, perfdir, scoredir, outdir, kwargs) for perf in performances]
        with multiprocessing.Pool(parallel) as p: p.starmap(align_and_save, args)
'''
    print('Computing {} alignments'.format(algo))
    total = 0 #tracker for the total time
        
    for i in range(0, len(performances)):
        perf = performances[i]
        print('  ', perf, end=' ')
        kwargs = {} #useful for passing extra variables
            
        t0 = time.time()
        perf_path = os.path.join(perfdir, perf)
        score = os.path.join(scoredir,util.map_score(perf) + '.midi')
            
        if algo == 'ground-beat-interpol':
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


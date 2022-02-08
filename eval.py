import os, sys
import numpy as np

import lib.util as util
import lib.midi as midi

import music21

import pandas as pd


def match_onsets(score_notes, perf_notes, gt_alignment, thres=.100):
    """ Onset matching heuristic """

    matched_onsets = []
    pitches,score_onsets,score_offsets = zip(*score_notes)
    gt_onsets = np.interp(score_onsets,*zip(*gt_alignment))
    for pitch,gt_onset,score_onset in zip(pitches,gt_onsets,score_onsets):
        best_dist = 1 + thres
        for perf_pitch, perf_onset, _ in perf_notes:
            if np.abs(perf_onset - gt_onset) > thres:
                 continue # not a match (too far away)
            if perf_pitch != pitch:
                 continue # not a match (wrong pitch)

            dist = np.abs(perf_onset - gt_onset)
            if dist < thres and dist < best_dist: # found a possible match
                best_match = perf_onset
                best_dist = dist

        if best_dist < thres: # found a match
            matched_onsets.append((score_onset,best_match))
            
    return matched_onsets

def filter_performances():
    #function to filter the performances based on performance characteristics. Hope to find something in maestro or asap metadata files
    return

#Thresholds and colors must be parallel with length greater than 1
#Thresholds must be sorted.
def color_chooser(error_val, colors = [], thresholds = []):
    for i in range(1, len(thresholds)):
        if error_val > thresholds[i-1] and error_val < thresholds[i]:
            return colors[i-1]
    return colors[-1]   
            

#expects 4 thresholds.
def prepare_eval_score(scoredir, evaldir, color_thresholds=color_chooser):
    # using the results of evaluation, returns the musicxml (or midi) score with color annotations in parts with bad accuracy. 
    #for now, only midi will be used for ease.

    for file in sorted([f[:-len('.midi')] for f in os.listdir(scoredir) if f.endswith('.midi')]):
        m21_score = m21.converter.parse('{}.midi'.format(file))
        errors = np.loadtxt(os.path.join(evaldir, file + '.txt'))
        
        for element in m21_score.flat.getElementsByClass([m21.note.Note, m21.chord.Chord]).secondsMap:
            error_window = util.binary_search(element['offsetSeconds'], errors[:, 0], 0, len(errors[:, 0]))
            element['element'].style.color = color_chooser(errors[error_window] * 1000)
        
        
            
            
            
            
            
            
            
        
        
    
    # iterate over the beatmap things (so it's not instrument by instrument)
        # get timing of the note
        # accordingly, look up the error corresponding to this time in the score
        # add an attribute (?) in the musicxml score according to the error.
    return

def calculate_summary_metrics(error_alignment):
    #'misalignment_rate_50ms', 'misalignment_rate_250ms' 'variance_misaligned_50ms', 'variance_misaligned_250ms', '1stquartile', 'median', '3rdquartile', 'average_absolute_offset']
    
    #misalignment rate calculations
    misaligned_50ms = error_alignment[error_alignment[:, 1] > 0.050][:, 1]
    misaligned_250ms = error_alignment[error_alignment[:, 1] > 0.250][:, 1]
   
    return misaligned_50ms.mean(), misaligned_250ms.mean(), misaligned_50ms.var(), misaligned_250ms.var(), np.percentile(error_alignment[:, 1], 25), np.percentile(error_alignment[:, 1], 50), np.percentile(error_alignment[:, 1], 75), abs(error_alignment[:, 1]).mean()

# we should pass in the interpolated ground truths. 
# calculates all the metrics for a particular alignment algorithm
def calculate_bulk_metrics(candidatedir, gtdir, scoredir, perfdir):
    # Returns A2S metrics (there is no standard set).
    # Piecewise Precision Rate (PPR)
    algo = os.path.basename(os.path.normpath(candidatedir))
    metrics_basepath = 'eval/{}'.format(algo)
    os.makedirs(metrics_basepath, exist_ok=True)
    
    summary_file = 'eval/{}/{}'.format(algo, 'metric_summary.csv')
    df_columns = ['file_id','misalignment_rate_50ms', 'misalignment_rate_250ms', 'variance_misaligned_50ms', 'variance_misaligned_250ms', '1stquartile', 'median', '3rdquartile', 'average_absolute_offset']
    summary = []
    
    for file in sorted([f[:-len('.midi')] for f in os.listdir(perfdir) if f.endswith('.midi')]):
        gt_alignment = np.loadtxt(os.path.join(gtdir, file + '.txt'))
        ch_alignment = np.loadtxt(os.path.join(candidatedir, file + '.txt'))
        
        score_file = os.path.join(scoredir, util.map_score(file) + '.midi')
        _,score_start,score_end = midi.load_midi_events(score_file, strip_ends=True)
        
        # truncate to the range [score_start,score_end)
        #idx0 = np.argmin(score_start > gt_alignment[:,0])
        #idxS = np.argmin(score_end > gt_alignment[:,0])
        #gt_alignment = gt_alignment[idx0:idxS]
        
        #linearize the output alignments to match the same score times as those in the gt alignment.
        linearized = np.interp(gt_alignment[:,0], *zip(*ch_alignment))
        ch_alignment = gt_alignment.copy()
        ch_alignment[:,1] = linearized
        
        #time error
        error = list(gt_alignment[0:,1] - ch_alignment[0:,1])
        
        zipped_error = np.array(list(zip(ch_alignment[:, 0], error)))
        np.savetxt(os.path.join(metrics_basepath, file + '.txt'), zipped_error, fmt='%f\t', header='score\t\tperformance')
        
        #from the error, also you can set error thresholds (50ms, 250ms, etc), and calculate the percentages
        misalignment_rate_50ms, misalignment_rate_250ms, variance_misaligned_50ms, variance_misaligned_250ms, quartile_1st, median, quartile_3rd, absolute_mean_error = calculate_summary_metrics(zipped_error)
        
        new_row = [file, misalignment_rate_50ms, misalignment_rate_250ms, variance_misaligned_50ms, variance_misaligned_250ms, quartile_1st, median, quartile_3rd, absolute_mean_error] 
        summary.append(new_row)
            
    summary_pd = pd.DataFrame(summary, columns = df_columns)
    summary_pd.to_csv(summary_file, index=False)
    
    return

def evaluate(candidatedir, gtdir, scoredir, perfdir):
    mad, old_mad, rmse, old_rmse, missedpct = [[] for _ in range(5)]
    outliers = 0
    epsilon = 1e-4
    print("Performance\tTimeErr\tTimeDev\tNoteErr\tNoteDev\t%Match")
    for file in sorted([f[:-len('.midi')] for f in os.listdir(perfdir) if f.endswith('.midi')]):
        gt_alignment = np.loadtxt(os.path.join(gtdir, file + '.txt'))
        ch_alignment = np.loadtxt(os.path.join(candidatedir, file + '.txt'))
    
        score_file = os.path.join(scoredir, util.map_score(file) + '.midi')
        _,score_start,score_end = midi.load_midi_events(score_file, strip_ends=True)
    
        # truncate to the range [score_start,score_end)
        idx0 = np.argmin(score_start > gt_alignment[:,0])
        idxS = np.argmin(score_end > gt_alignment[:,0] + epsilon)
        gt_alignment = gt_alignment[idx0:idxS]
    
        #
        # compute our metrics
        #
    
        # linearized timings
        linearized = np.interp(gt_alignment[:,0], *zip(*ch_alignment))
        ch_alignment = gt_alignment.copy()
        ch_alignment[:,1] = linearized
    
        S = gt_alignment[-1,0] - gt_alignment[0,0]
        ds = gt_alignment[1:,0] - gt_alignment[:-1,0]
        error = list(gt_alignment[0:,1] - ch_alignment[0:,1])

        dev = [(1/2)*(e1+e2) if samesign
            else (1/2)*(e1**2+e2**2)/(e1+e2)
            for (e1,e2,samesign) in zip(np.abs(error[:-1]), np.abs(error[1:]), np.sign(error[:-1])==np.sign(error[1:]))]

        se = [(1/3)*(e1**2+e1*e2+e2**2) for (e1,e2) in zip(error[:-1],error[1:])]

        thismad = (1./S)*np.dot(dev,ds)
        thisrmse = np.sqrt((1./S)*np.dot(se,ds))
    
        #
        # compute old metrics
        #
    
        score_notes,_ = midi.load_midi(os.path.join(scoredir, util.map_score(file) + '.midi'))
        perf_notes,_ = midi.load_midi(os.path.join(perfdir,file + '.midi'))
        matched_onsets = match_onsets(score_notes, perf_notes, gt_alignment)
        missedpct.append(100*len(matched_onsets)/len(score_notes))

        onsets, gt_onsets = zip(*matched_onsets)
        ch_aligned_onsets = np.interp(onsets,*zip(*ch_alignment))
        dev = gt_onsets - ch_aligned_onsets

        thisold_mad = (1./len(dev))*np.sum(np.abs(dev))
        thisold_rmse = np.sqrt((1./len(dev))*np.sum(np.power(dev,2)))

        # throw out outliers with error > 300ms
        if thismad < .300:
            mad.append(thismad)
            rmse.append(thisrmse)
            old_mad.append(thisold_mad)
            old_rmse.append(thisold_rmse)
        else:
            outliers += 1   

        print('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(file, thismad, thisrmse, thisold_mad, thisold_rmse, missedpct[-1]))

    print('=' * 100)
    print('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format('bottomline', np.mean(mad), np.mean(rmse), np.mean(old_mad), np.mean(old_rmse), np.mean(missedpct)))
    print('(removed {} outliers)'.format(outliers))
   

if __name__ == "__main__":
    algo = sys.argv[1]
    scoredir = sys.argv[2]
    perfdir = sys.argv[3]

    candidatedir = os.path.join('align',algo)
    gtdir = os.path.join('align','ground')
    #evaluate(candidatedir, gtdir, scoredir, perfdir)


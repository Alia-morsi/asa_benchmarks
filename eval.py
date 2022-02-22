import os, sys
import numpy as np

import lib.util as util
import lib.midi as midi

import music21 as m21

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
def color_chooser(error_val, thresholds = [], colors = []):
    for i in range(1, len(thresholds)):
        if error_val > thresholds[i-1] and error_val < thresholds[i]:
            return colors[i-1]
        elif error_val < thresholds[i-1]:
            return colors[i-1]
    return colors[-1]   
            

#expects 4 thresholds.
def prepare_eval_score(scoredir, evaldir, thresholds, colors, color_thresholds=color_chooser):
    # using the results of evaluation, returns the musicxml (or midi) score with color annotations in parts with bad accuracy. 
    #for now, only midi will be used for ease.

    for file in sorted([f[:-len('.midi')] for f in os.listdir(scoredir) if f.endswith('.midi')]):
        m21_score = m21.converter.parse(os.path.join(scoredir, '{}.midi'.format(file)))
        errors = np.loadtxt(os.path.join(evaldir, file + '.txt'))
        
        for element in m21_score.flat.getElementsByClass([m21.note.Note, m21.chord.Chord]).secondsMap:
            error_window = np.searchsorted(errors[:, 0], element['offsetSeconds'])
            if error_window >= len(errors):
                continue
            element['element'].style.color = color_chooser(abs(errors[error_window][1]), thresholds, colors)
            
        m21_score.write('musicxml', os.path.join(evaldir, '{}.xml'.format(file)))
    return
  

def calculate_summary_metrics(error_alignment):
    
    #misalignment rate calculations
    misaligned_50ms = error_alignment[abs(error_alignment)[:, 1] > 0.050][:, 1]
    misaligned_250ms = error_alignment[abs(error_alignment)[:, 1] > 0.250][:, 1]
    
    misalignment_rate_50ms = len(misaligned_50ms) * 100 / len(error_alignment)
    misalignment_rate_250ms = len(misaligned_250ms) * 100 / len(error_alignment)
   
    return misalignment_rate_50ms, misalignment_rate_250ms, abs(misaligned_50ms).mean(), abs(misaligned_250ms).mean(), misaligned_50ms.var(), misaligned_250ms.var(), np.percentile(abs(error_alignment[:, 1]), 25), np.percentile(abs(error_alignment[:, 1]), 50), np.percentile(abs(error_alignment[:, 1]), 75), abs(error_alignment[:, 1]).mean()

# we should pass in the interpolated ground truths. 
# calculates all the metrics for a particular alignment algorithm
def calculate_bulk_metrics(candidatedir, gtdir, scoredir, perfdir):
    # Returns A2S metrics (there is no standard set).
    # Piecewise Precision Rate (PPR)
    algo = os.path.basename(os.path.normpath(candidatedir))
    metrics_basepath = 'eval/{}'.format(algo)
    os.makedirs(metrics_basepath, exist_ok=True)
    
    summary_file = 'eval/{}/{}'.format(algo, 'metric_summary.csv')
    threshold_file = 'eval/{}/{}'.format(algo, 'threshold_summary.csv')
    
    df_columns = ['file_id', 'misalignment_rate_50ms', 'misalignment_rate_250ms', 'misalignment_mean_50ms', 'misalignment_mean_250ms', 'variance_misaligned_50ms', 'variance_misaligned_250ms', '1stquartile', 'median', '3rdquartile', 'average_absolute_offset']
    summary = []
    
    thresholds = list(range(0, 1010, 10))
    aligned_notes = np.zeros(len(thresholds))
    total_notes = np.zeros(len(thresholds))
    
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
        
        #calculate AR at different thresholds:
        for i in range(0, len(thresholds)):
            t = thresholds[i]/1000.0
            aligned_notes[i] += len([i for i in error if abs(i) < t])
            total_notes[i] += len(error)
            
        
        zipped_error = np.array(list(zip(ch_alignment[:, 0], error)))
        np.savetxt(os.path.join(metrics_basepath, file + '.txt'), zipped_error, fmt='%f\t', header='score\t\tperformance')
        
        #from the error, also you can set error thresholds (50ms, 250ms, etc), and calculate the percentages
        misalignment_rate_50ms, misalignment_rate_250ms, misalignment_50ms_mean, misalignment_250ms_mean, variance_misaligned_50ms, variance_misaligned_250ms, quartile_1st, median, quartile_3rd, absolute_mean_error = calculate_summary_metrics(zipped_error)
        
        new_row = [file, misalignment_rate_50ms, misalignment_rate_250ms, misalignment_50ms_mean, misalignment_250ms_mean, variance_misaligned_50ms, variance_misaligned_250ms, quartile_1st, median, quartile_3rd, absolute_mean_error] 
        summary.append(new_row)
            
    summary_df = pd.DataFrame(summary, columns = df_columns)
    summary_df.to_csv(summary_file, index=False)
    
    #alignment rate per each threshold
    alignment_rate = (aligned_notes/total_notes) * 100
    
    #Alignment Rate by Threshold
    ar_df = pd.DataFrame(np.stack([thresholds, alignment_rate], axis=1), columns=['threshold', 'alignment_rate'])
    ar_df['algo'] = [algo] * len(thresholds)
    ar_df.to_csv(threshold_file, index=False)
    
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
    
    return


def get_score_performance_dfdt(gt_annotation_file):
    gt_alignment = np.loadtxt(gt_annotation_file)
    #not sure how many times it will crash due to zeroes.
    derivatives = [abs(y2-y1)/abs(x2-x1) for x1, y1, x2, y2 in zip(example[:-1, 0], example[1:, 0], example[:-1, 1], example[1:, 1])]
    return np.average(derivatives)


def metadata_for_qa(scoredir, perfdir, gtdir):
    #it might also be useful to search through which
    #alignment_dirs = [os.path.join(align_base, algo) for algo in algos]
    #evaluation_dirs = [os.path.join(eval_base, algo) for algo in algos]
    
    gt_files = sorted([f for f in os.listdir(gtdir) if f.endswith('.txt')])
    #to get the base filename, all we have to do is gt_alignment[:-len('.txt')]

    #large time offset between score and midi
    score_performance_dfdt = [] #this can later be replaced by avg of derivative but for now fakes.
    score_performance_d2fdt2 = [] #fakes for now
    for gt_file in gt_files:
        gt_annotation = np.loadtxt(os.path.join(gtdir, gt_file))
        score_performance_dfdt.append(abs(gt_annotation[-1, 1] - gt_annotation[0, 1])/
                                     abs(gt_annotation[-1, 0] - gt_annotation[-1, 1]))
        
    files = [file[:-len('.txt')] for file in gt_files]
    extra_metadata_df = pd.DataFrame(np.stack([files, score_performance_dfdt], axis=1), columns = ['file_id', 'score_performance_dfdt'])
    
    score_bpm = []
    performance_bpm = []
    for file in files:
        score_m21 = m21.converter.parse(os.path.join(scoredir, '{}.midi'.format(file)))
        performance_m21 = m21.converter.parse(os.path.join(perfdir, '{}.midi'.format(file)))
        
        score_tempo = list(score_m21.flat.getElementsByClass(m21.tempo.MetronomeMark))
        performance_tempo = list(performance_m21.flat.getElementsByClass(m21.tempo.MetronomeMark))
        
        #See if there is another way to get this info from mido directly without m21, because
        # it seems that they all just get the default bpm, and then the comparison makes no sense.
        score_bpm.append(score_tempo[0].number if len(score_tempo)>0 else -1)
        performance_bpm.append(performance_tempo[0].number if len(performance_tempo)>0 else -1)
        
    extra_metadata_df['score_bpm'] = score_bpm
    extra_metadata_df['performance_bpm'] = performance_bpm
    
    extra_metadata_df.to_csv('extra_metadata.csv', index=False)

    return

def metadata_for_visualization(candidatedir, gtdir, scoredir, perfdir):
    # list all the files in candidate dir
    
    for file in files:
        # load gt
        # load midi score
        # load score annotation
        # load midi performance
        # load performance annotation
        print('test')
        
        # variance of the time rate of gt: load gt, interpolate based on columns, get the mean of the second derivative. https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.interpolate.UnivariateSpline.derivative.html#scipy.interpolate.UnivariateSpline.derivative
        # 
    
    return
   

if __name__ == "__main__":
    algo = sys.argv[1]
    scoredir = sys.argv[2]
    perfdir = sys.argv[3]

    candidatedir = os.path.join('align',algo)
    gtdir = os.path.join('align','ground')
    #evaluate(candidatedir, gtdir, scoredir, perfdir)


## This is the accompanying code for the paper: Bottlenecks and Solutions for Audio to Score Alignment Research
It is a repository that allows the easy application of several DTW variants for audio to score alignment on an interpolated version of the asap dataset, as explained in the publication. 
 
The purpose of this is to create a foundation for a system which can be used for Audio to Score alignment benchmarking. At its current state it is by no means complete, but it is a call for interested researchers to gather and discuss what should be done. For this to be a framework for benchmarking, we would need:
1) to extend the datasets used. Include those in table 1 of the publication, and synthesize/adapt others when possible
2) to include more audio to score alignment implementations, including HMM based ones and others.
3) to expand the evaluation metrics, to cover a variety of scenarios, as argued in the initial publication.

The paper discusses the use of just one of such datasets (the ASAP dataset), for Audio to Score alignment, and only applies DTW algorithms.

## Setting up
Most of the requirements of this repository would be included in the environment.yml of the docker subdirectory. However, there are some things that (for now) need to be installed manually and copied to the correct places.

### pitchclass ctc models:
The folders libdl, libfmp, and models_pretrained need to be downloaded to the asa_benchmarks/lib/pitchclass_mctc directory. The file wrappers.py in that same directory was written by us to facilitate calling the pitchclass-mctc chroma estimation functions within the standard alignment pipeline. 
 
https://github.com/christofw/pitchclass_mctc (this codebase is on commit 04b4a96). 

### nnls-chroma vamp plugin:
To download the nnls vamp plugin, please go to http://www.isophonics.net/nnls-chroma and download the linux binary folder. extract and rename to vamp, and put the vamp folder (which has the files CITATION, COPYING, nnls-chroma.cat, nnls-chroma.n3, nnls-chroma.so, and README) in the docker folder. 

### ASAP Dataset:

## Getting the ASAP dataset
However, first the ASAP files need to be restructured a bit.

```
from utils import restructure_asap_files

restructure_asap_files('/Users/aliamorsi/Desktop/phD/a2s_with_dtw_survey/pitchclass_mctc/data/asap-preludes', 'metadata.csv')

```

## Interpolating Ground Truths from ASAP
We have added the ability to both generate ground truth annotations and to sonify them + play them alongside the actual performance wav.

Then, run the following to create the ground truth interpolations

```
python3 align.py ground-beat-interpol data/score data/perf 
```

To sonify them, run:

```
from extract import sonify_interpolated_gt

sonify_interpolated_gt()
```

and by default, the sonified files will be created in eval/sonic.
These files are stereo, with the sonified performance_aligned_gt on one channel and the performance wav on the other.

Open them with audacity, split to have each channel on a separate track, and play with the volume so that they sound even. If playing them together does not seem odd, then there is no problem with the ground truth.

## Restructured Data
After restructuring, the ASAP dataset looks as follows.

ASAP: 
Each dataset folder has a perf and a score directory:
perf/
<performance-1>.midi
<performance-1>.txt
<performance-1>.wav
.....
<performance-n>.midi
<performance-n>.txt
<performance-n>.wav

Each performance should have a midi file (corresponding to the performance), a txt file with the downbeat annotations, and a wav file of the audio itself. 

score/
<performance-1>.midi
<performance-1>.txt
....
<performance-n>.midi
<performance-n>.txt

Each performance has the midi score, and a txt file with the downbeat annotations. 

Whether in the perf or score directory, the annotation files are in the form: (put form of asap, and link)

Other datasets would, of course, be organized differently. 
(the difference between the restructure file and the interpolate file)

## Running the Code:
align.py is the entry point. See help (python align.py -h) to see the parameters. 
options:
--data: switch to indicate if you'd like to run a datset adaptation function, or an alignment function.
--action: expects a string parameter. if --data switch is on, then it should be ground_truth_interpol. Otherwise, it should be one of the alignment functions supported (chroma, cqt, dce, pitchclass-mctc)
--dataset: the name of the dataset. 

The output of an alignment is put in align/<dataset_name>/<algorithm_name>. The alignment files are text files mapping the time in the score (left column) to the time in the performance (right column). 

Whether for the interpolation (or other preprocessing to be added) or alignment, the file structure output from the restructure functionality needs to be put.

## Interpolating Ground Truths from ASAP
We have added the ability to both generate ground truth annotations and to sonify them + play them alongside the actual performance wav.

Then, run the following to create the ground truth interpolations

```
python3 align.py ground-beat-interpol data/score data/perf
```

To sonify them, run:

```
from extract import sonify_interpolated_gt

sonify_interpolated_gt()
```

and by default, the sonified files will be created in eval/sonic.
These files are stereo, with the sonified performance_aligned_gt on one channel and the performance wav on the other.

Open them with audacity, split to have each channel on a separate track, and play with the volume so that they sound even. If playing them together does not seem odd, then there is no problem with the ground truth.


## Extending the codebase
To add a new dataset

To add a new algorithm

To include training as part of this codebase

- if you are to add another dtw implementation, make sure that you transpose the audio representations to the format expected by the distance measure used.
- On the conversion from and to the actual times,

## Running with Docker
To run with Docker, please see docker/README for instructions on how to build a docker image from the resources in this repository, and how to mount folders on your local machine to the docker container (to avoid the image getting extra large, and so that the data persists after calculation)

On completing the said instructions, you would have a docker container with a copy of asa_benchmarks, in the root directory, where the data, align, and eval folders are symlinks to their corresponding folders in /mnt of the container (which are in turn folders in your local asa_benchmarks). Hence, you can run everything in the docker container, and the data would persist in their local locations in asa_benchmarks.
 
## Computing Alignments

To compute an alignment with a given dataset, 
```
python3 align.py {spectra,chroma,cqt} data/score data/perf N
```

The alignments generated by the alignment script are stored in align/{ground,spectra,chroma,cqt} as
plaintext files with two columns: the first column indicates time in the score, and the second
column indicates time in the performance.

To evaluate the results of a particular alignment algorithm:

(The evaluate and the generation of result files should be one step..)

```
python3 eval.py {spectra,chroma,cqt,ctc-chroma} data/score data/perf
```

To generate result files from the alignment process, run

```
from eval import calculate_bulk_metrics

calculate_bulk_metrics('align/ctc-chroma/', 'align/ground-beat-interpol', 'data/score', 'data/perf')
```

To create the extra metadata which we will use to filter things later

```
from eval import metadata_for_qa

metadata_for_qa('data/score', 'data/perf', 'align/ground-beat-interpol')
```

## Visualizations 

TODO: Add reference to the musicxml visualizations, and keep the visualizations by Thickstun + relate them to the missing notes/extra notes evaluation by Nakamura. 

## More Candidate Datasets
Examples of relevant datasets:
TO COMPLETE: Relevant datasets for Audio to Score alignment

Saarland                Granularity  Stength
RWC alignment subset
Phoenix
Bach10
ASAP (vanilla)
Interpolated ASAP

Granularity Strength Ideas
beat level
note level

## References

To reference this work, please cite
```bib
@article{morsi_serra_bottlenecks, 
author = {Alia Morsi and Xavier Serra}, 
title = {Bottlenecks and Solutions for Audio to Score Alignment Research}, 
journal = {Proceedings of the 23rd International Society for Music Information Conference (ISMIR)},
year = {2022}
}
```

Although it has diverged, this repo started out as a fork of https://github.com/jthickstun/alignment-eval



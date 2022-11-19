import librosa
import torch
import os
import numpy as np

#Custom Imports
import lib.pitchclass_mctc.libdl.data_preprocessing
from lib.pitchclass_mctc.libdl.data_loaders import dataset_context, dataset_context_segm
from lib.pitchclass_mctc.libdl.nn_models import basic_cnn_segm_sigmoid, basic_cnn_segm_blank_logsoftmax
from lib.pitchclass_mctc.libdl.data_preprocessing import compute_hopsize_cqt, compute_hcqt, compute_efficient_hcqt, compute_annotation_array_nooverlap

pretrained_models_base = 'lib/pitchclass_mctc/models_pretrained'


#Design wise I'm not really sure if there is any merit to having the dictionaries saved as class variables or
#keeping them as global vars..

num_octaves_inp = 6
num_output_bins, min_pitch = 12, 60

ctc_model_params = {
                'n_chan_input': 6,
                'n_chan_layers': [20,20,10,1],
                'n_ch_out': 2,
                'n_bins_in': num_octaves_inp*12*3,
                'n_bins_out': num_output_bins,
                'a_lrelu': 0.3,
                'p_dropout': 0.2
                }

ctc_fn_model = 'exp131b_trainmaestromunet_testmix_mctcwe_pitchclass_basiccnn_normtargl_SGD.pt'

#HCQT Parameters
hcqt_parameters = { 'bins_per_semitone' : 3,
                     'num_octaves' : 6,
                     'num_harmonics' : 5,
                     'num_subharmonics' : 1
                  }
hcqt_parameters['n_bins'] = hcqt_parameters['bins_per_semitone']*12*hcqt_parameters['num_octaves']

# Test Parameters
test_params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1
              }
device = 'cpu'

test_dataset_params = {'context': 75,
                       'seglength': 10,
                       'stride': 10,
                       'compression': 10
                      }

half_context = test_dataset_params['context']//2


def frame_to_sec(sampling_rate, hopsize, input_array):
    return input_array*(hopsize/sampling_rate)

def sec_to_frame(sampling_rate, hopsize, input_array):
    return input_array*(sampling_rate/hopsize)

class PitchClassCTC:
    def __init__(self, fn_model = ctc_fn_model, model_params = ctc_model_params, models_path=pretrained_models_base):
        self.model_params = model_params
        self.models_path = models_path
        self.fn_model = fn_model
        
        #loading and initializing the models
        self.model = basic_cnn_segm_blank_logsoftmax(n_chan_input = ctc_model_params['n_chan_input'], 
                                n_chan_layers = ctc_model_params['n_chan_layers'], 
                                n_ch_out = ctc_model_params['n_ch_out'], 
                                n_bins_in = ctc_model_params['n_bins_in'], 
                                n_bins_out = ctc_model_params['n_bins_out'], 
                                a_lrelu = ctc_model_params['a_lrelu'], 
                                p_dropout = ctc_model_params['p_dropout'])
        
        path_trained_model = os.path.join(self.models_path, self.fn_model)
        
        self.model.load_state_dict(torch.load(path_trained_model, map_location=torch.device('cpu')))
        
    def __call__(self, audio_in, fs_in):
          
        #calculate hcqt
        f_hcqt, fs_hcqt, hopsize_cqt = compute_efficient_hcqt(
            audio_in, 
            fs=fs_in, 
            fmin=librosa.note_to_hz('C1'), 
            fs_hcqt_target=10, 
            bins_per_octave=hcqt_parameters['bins_per_semitone']*12,
            num_octaves=hcqt_parameters['num_octaves'], 
            num_harmonics=hcqt_parameters['num_harmonics'], 
            num_subharmonics=hcqt_parameters['num_subharmonics'])
        
        
        inputs = np.transpose(f_hcqt, (2, 1, 0))
        targets = np.zeros(inputs.shape[1:]) # need dummy targets to use dataset object
        inputs_context = torch.from_numpy(np.pad(inputs, ((0, 0), (half_context, half_context+1), (0, 0))))
        targets_context = torch.from_numpy(np.pad(targets, ((half_context, half_context+1), (0, 0))))

        test_set = dataset_context_segm(inputs_context, targets_context, test_dataset_params)
        test_generator = torch.utils.data.DataLoader(test_set, **test_params)
        
        pred_tot = np.zeros((0, num_output_bins))
    
        for test_batch, test_labels in test_generator:
            # Model computations
            y_pred = self.model(test_batch)
            pred_log = torch.squeeze(y_pred.to('cpu')).detach().numpy()
            pred_tot = np.append(pred_tot, pred_log[1, :, 1:], axis=0)
            predictions = np.exp(pred_tot)
            
        #we do this transposition so that it fits with the expected input of librosa.dtw
        return hopsize_cqt, fs_in, predictions.transpose([1, 0])
        
    
    

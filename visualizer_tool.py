import numpy as np
import holoviews as hv
hv.extension('bokeh')
import panel as pn
import torch
import librosa
import torchaudio
from functools import partial
from IPython.display import Audio
import datetime
from bokeh.models import HoverTool

def float_sec2minute_sec(a):
    
    min=int(a//60)
    sec=int(a%60)
    if sec == 0:
        sec = '00'
    a_str = '{}:{}'.format(min,sec)
    return a_str

def make_data(x,y,z,xdict,label=True):
    assert len(z.shape) == 2

    I = z.shape[0]
    J = z.shape[1]
    jt = J/300
    data = []
    for i in range(I):
        for j in range(J):
            if label:
                if z[i,j]>0:
                    zij = 1
                else:
                    zij=0
            else:
                zij = z[i,j]
            ijk = (y[j]/jt,xdict[int(x[i])],zij)
            data.append(ijk)
    return data

def norm(data,idx=True):

    if idx:
        maxdata = np.max(data,axis=1)[0]
        mindata = np.min(data,axis=1)[0]
        norm_data = (data - mindata)/(maxdata-mindata)
        return norm_data
    else:
        maxdata = np.max(data)
        mindata = np.min(data)
        norm_data = (data - mindata)/(maxdata-mindata)
        return norm_data

def create_visualizer_gui(num_path, categories,dur=2):
    
    sample = torch.load('data/data{}.pt'.format(num_path))
    sig,fs = torchaudio.load('audio_signals/audio{}.wav'.format(num_path))
    


    stai,mbe,labels  = sample['stai'].to('cpu'),sample['mbe'].to('cpu'), sample['label'].to('cpu')

    labels = labels*len(categories)
    idx = ['ACI', 'BIO', 'NDSI', 'SH', 'TH', 'M']

    n_windows, n_labels = labels.shape
    _, n_mels = mbe[0].shape
    _, n_stai = stai[0].shape

    stai = norm(stai.numpy())
    mbe = norm(mbe.numpy(),idx=False)
    
    num_windows = np.linspace(0,n_windows-1,n_windows)
    num_labels = np.linspace(0,n_labels-1,n_labels,dtype=int)
    num_stai = range(n_stai)
    num_mbe = np.linspace(0,n_mels-1,n_mels,dtype=int)
    

    idx_data = make_data(num_stai,num_windows,stai[0].T,idx,label=False)
    label_data = make_data(num_labels,num_windows,labels.numpy().T,categories)
    mbe_data = make_data(num_mbe,num_windows,mbe[0].T,num_mbe+1,label=False)

    
    

    stai_plot = hv.HeatMap(idx_data, 
                           kdims=['time', 'IDXS'],vdims=['value'])
    mbe_plot = hv.HeatMap(mbe_data,
                            kdims=['time','Mel'],vdims=['value'])
    label_plot = hv.HeatMap(label_data,
                            kdims=['time', 'class'],vdims=['presence'])                      

    stai_plot.opts(height=200, width=950, cmap='Inferno',
                   tools=['hover'],show_grid=True,colorbar=True,xlim=(0,300))
    mbe_plot.opts(height=200, width=950, cmap='Spectral_r',
                  tools=['hover'],show_grid=False,colorbar=True,xlim=(0,300),ylim=(0,n_mels))
    
    label_plot.opts(height=450, width=950, cmap='Blues',
                    tools=['hover'],show_grid=True,xlim=(0,300))


    bg_plot = hv.Layout([stai_plot,mbe_plot, label_plot]).cols(1)

    frame_slider = pn.widgets.IntSlider(name="Time", value=0, start=0, end=300-dur)
    length = dur*fs

    @pn.depends(frame=frame_slider)
    def cross_hair(frame):
        return hv.VLine(frame).opts(color='red') * hv.VLine(frame+dur).opts(color='red')

    cross_hair_dmap = hv.DynamicMap(cross_hair)
    plots = (bg_plot * cross_hair_dmap).cols(1)

    def prepare_wav_for_panel(waveform):
        max_val = np.amax(np.abs(waveform))
        return np.int16(waveform * 32767 / max_val)


    @pn.depends(frame=frame_slider)
    def listen(frame):
        idx_wav = int(frame*fs)
        return pn.pane.Audio(prepare_wav_for_panel(sig[0, idx_wav:idx_wav+length].numpy()), 
                             sample_rate=fs, name='Audio', autoplay=True)

    app = pn.Column(
        pn.Spacer(height=10),
        frame_slider,
        plots,
        listen,
    )
    return app
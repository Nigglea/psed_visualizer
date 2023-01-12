import numpy as np
import holoviews as hv
hv.extension('bokeh')
import panel as pn
import torch
import librosa
import torchaudio
from functools import partial

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

def create_sed_gui(num_path, categories,dur=2):
    
    sample = torch.load('mbe/mbe{}.pt'.format(num_path),map_location=torch.device('cpu'))
    sig,fs = torchaudio.load('audio_signals/audio{}.wav'.format(num_path))

    logmel,labels  = sample[list(sample.keys())[0]].to('cpu'),sample[list(sample.keys())[1]].to('cpu')
    get_rms = partial(librosa.feature.rms, frame_length=2048, hop_length=1024)
    rms = 20*np.log10(get_rms((sig[0]-torch.mean(sig))))


    n_windows, n_labels = labels.shape
    _, n_mels = logmel[0].shape

    labels = labels*len(categories)
    num_windows = np.linspace(0,n_windows-1,n_windows)
    num_labels = np.linspace(0,n_labels-1,n_labels)
    num_mels = np.arange(n_mels)
    hz = np.linspace(0,fs//2,n_mels)
    mel = 2410*np.log10(1+hz/625)
    
    label_ticks = [(int(tick), label) for tick, label in zip(np.arange(len(categories)), categories)]
    mel_ticks = [(int(m),n_mels-int(m)) for m in np.arange(0,n_mels-1,4)]
    nt = 5*3+1
    win_ticks = np.linspace(0,n_windows+861,nt+1)
    tim_ticks = np.linspace(0,300+20,nt+1,dtype=np.float32)
    time_win_ticks = [(int(win_ticks[i]),float_sec2minute_sec(tim_ticks[i])) for i in range(len(win_ticks)-1)]
    
    time = np.linspace(0,300,len(rms[0]))
    RMS = [(time[i],rms[0][i]) for i in range(len(rms[0]))]
    
    
    mel_data = make_data(num_mels,num_windows,logmel[0].numpy().T,num_mels,label=False)
 
    label_data = make_data(num_labels,num_windows,labels.numpy().T,categories)

    
    
    
    rms_plot = hv.Curve(RMS, 'time', 'rms').opts(height=180, width=950,xlim=(0,300))
    
    logmel_plot = hv.HeatMap(mel_data, 
                               kdims=['time', 'Mel'],vdims=['value']).opts(height=450, width=950, cmap='Spectral_r', xlim=(0,300),
                                                            ylim=(min(num_mels),max(num_mels)),tools=['hover'],colorbar=True)
    
    label_plot = hv.HeatMap(label_data,
                            kdims=['time', 'class'], vdims=['presence']).opts(height=450, width=950, cmap='Blues', 
                                                            xlim=(0,300),tools=['hover'],show_grid=True)


    bg_plot = hv.Layout([rms_plot, logmel_plot, label_plot]).cols(1)
    

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
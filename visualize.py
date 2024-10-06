import os
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset
from obspy import read


for file in os.listdir('./data/lunar/test/data/S12_GradeB/'):
    if file.endswith('.mseed'):
        tr = read(f'./data/lunar/test/data/S12_GradeB/{file}')[0]
        # Sampling frequency of our trace
        # df = tr.stats.sampling_rate

        # How long should the short-term and long-term window be, in seconds?
        # sta_len = 120
        # lta_len = 600

        tr_data = tr.data
        tr_times = tr.times()
        plt.plot(tr_times,tr_data)
        plt.title(file)
        plt.show()
        
        # Run Obspy's STA/LTA to obtain a characteristic function
        # This function basically calculates the ratio of amplitude between the short-term 
        # and long-term windows, moving consecutively in time across the data
        # cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))

        # # Plot characteristic function
        # fig = plt.figure(figsize=(10, 10))
        # ax = plt.subplot(2, 1, 1)
        # # Play around with the on and off triggers, based on values in the characteristic function
        # thr_on = 4
        # thr_off = 1.5
        # on_off = np.array(trigger_onset(cft, thr_on, thr_off)) 
        # for i in np.arange(0,len(on_off)):
        #     triggers = on_off[i]
        #     ax.axvline(x = tr_times[triggers[0]], color='red', label='Trig. On')
        # ax.plot(tr_times,tr_data)
        # ax.set_xlim([min(tr_times),max(tr_times)])
        # ax.legend()
        # ax.set_title('STA/LTA')
        # ax2 = plt.subplot(2, 1, 2)
        # ax2.imshow(plt.imread(f'./data/lunar/training/plots/{file[:-6]}.png'))
        # ax2.set_title('Spectrogram')
        # plt.show()
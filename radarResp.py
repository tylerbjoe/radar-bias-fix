class Radar_Resp():

    ## <-------------------  Begin __init__() method -------------------> ##

    def __init__(self, num_sensors, n, highsides, fs):
        self.num_sensors = num_sensors
        self.n = n
        self.highsides = [f'HS {i}' for i in highsides]
        self.fs = fs

    ## <-------------------  End __init__() method -------------------> ##

    ## <-------------------  Begin window_all_signals() method -------------------> ##

    def window_all_signals(self, resampled_ppg_dict):
        """
        Method for windowing ppg signals to get them ready to process. Makes 20 second windows
        so that the latter 10 seconds can be used, and the first 10 seconds discarded after 
        filtering to avoid ringing.

        :PARAMS:
        resampled_ppg_dict: [multi-nested dict] Nested dictionary of resampled ppg signals 
                            from all sensors, colors, and highsides. Nesting is as folows:
                            resampled_ppg_dict[sensor][color][hs] where sensor is 'hydros X',
                            color is in ['green', 'ir', 'red'], and hs is 'HS X' for X being
                            the sensor or highside number.
        resampled_time: [1D np.array] The new time array for the whole collection, in seconds.         
        
        :RETURNS:
        X: [nested dict] The windowed ppg signals. Nested as X[hs][color][window index]
        """

        # WINDOW THE SIGNALS TO PROCESS IN 10 SECOND CHUNKS
        X = {}  # initialize dictionary of windowed signals 

        SENSOR_KEY = f'hydros {self.n}'

        for hs in self.highsides:

            # SELECT A SINGLE CHANNEL'S RED AND IR SIGNALS
            red = resampled_ppg_dict[SENSOR_KEY]['red'][hs]
            ir = resampled_ppg_dict[SENSOR_KEY]['ir'][hs]
            green = resampled_ppg_dict[SENSOR_KEY]['green'][hs]

            raw_ppg = {'red': red,
                    'ir': ir,
                    'green' : green}

            W = window_signal(X=raw_ppg, win_len=20, dt=1, fs=self.fs, ref='red')

        num_windows = X['HS 1']['red'].shape[0]

        self.num_windows = num_windows

        return X
        
    ## <-------------------  End window_all_signals() method -------------------> ##

    ## <-------------------  Begin __segment_and_cluster_windows() method -------------------> ##

    def __segment_and_cluster_window(self, X, i):
        """ 
        Method for performing first stage of Bayesian SpO2 algorithm, which involves filtering
        the ppg signal, segmenting it into pieces, clustering those segments, and identifying which
        cluster to use in the inference stage.

        :PARAMS:
        X: [nested dict] The windowed ppg signals. Nested as X[hs][color][window index].
        i: [int] The window index which selects the window on which to perform the operation.

        :RETURNS:
        red_segs: [2D np.array] The red ppg segments from the cluster selected for inference.
                The first index is the segment index, and the second index is over points in each segment.
                Ex. red_segs[k, :] is the kth signal segment. 
        ir_segs: [2D np.array] The ir ppg segments from the cluster selected for inference.
                The first index is the segment index, and the second index is over points in each segment.
                Ex. red_segs[k, :] is the kth signal segment. 
        """
            
        # ADJUST AND SEGMENT ALL 4 HS
        R = do_shift_and_combine(X, x_name='red', win_ind=i, fs=self.fs, highsides=self.highsides)
        IR = do_shift_and_combine(X, x_name='ir', win_ind=i, fs=self.fs, highsides=self.highsides)
        
        # CATCH NO PARTITIONS ERROR
        if R[0,0] == -1 or IR[0,0] == -1:
            return np.array([[-1]]), np.array([[-1]])

        # CLUSTER
        num_clusters = 3
        red_labels = do_spectral_clustering(R, num_clusters)
        ir_labels = do_spectral_clustering(IR, num_clusters)

        # DETERMINE THE LABEL CORRESPONDING TO THE PEAK CLUSTER IN EACH SIGNAL WINDOW
        red_seg_label = identify_peaks(R, red_labels, num_clusters, mode='mid')
        ir_seg_label = identify_peaks(IR, ir_labels, num_clusters, mode='mid')  

        # TAKE THE PEAKS AND TIME SERIES FOR THE PEAKS IN BOTH SIGNALS
        red_seg_inds = np.squeeze(
                            np.array(
                                        np.where(red_labels==red_seg_label)
                                        )
                            )
        ir_seg_inds = np.squeeze(
                            np.array(
                                        np.where(ir_labels==ir_seg_label)
                                        )
                            )

        red_segs = np.array([R[j, :] for j in red_seg_inds])
        ir_segs = np.array([IR[j, :] for j in ir_seg_inds])


        return red_segs, ir_segs
    
    ## <-------------------  End __segment_and_cluster_windows() method -------------------> ##

    ## <-------------------  Begin __do_bayesian_inference() method -------------------> ##

    def __do_bayesian_inference(self, bi, red_segs, ir_segs, red_prior, ir_prior):
        """
        Method for performing Bayesian inference in a window. The method models the cluster of
        segments red_segs and ir_segs as samples from a normal distribution with a mean
        that is a true AC component. The method performs exact inference to determine the 
        parameters mu and sigma of this distribution for each window (the "evidence" is 
        explicitly calculated) with no need for sampling or variational methods.

        :PARAMS:
        bi: [class instance] Instance of the Bayesian_Inference class. This contains useful utilities.
        red_segs: [2D np.array] The red ppg segments from the cluster selected for inference.
                The first index is the segment index, and the second index is over points in each segment.
                Ex. red_segs[k, :] is the kth signal segment. 
        ir_segs: [2D np.array] The ir ppg segments from the cluster selected for inference.
                The first index is the segment index, and the second index is over points in each segment.
                Ex. red_segs[k, :] is the kth signal segment.
        red_prior: [2D np.array] Normalized prior probability distribution. The probability distribution that
                    encodes belief about the model paramters before seeing evidence from the current window.
                    This is either a uniform distribution at the 0th window, or it is the relaxed posterior from
                    the previous window.
        red_prior: [2D np.array] Normalized prior probability distribution. The probability distribution that
                    encodes belief about the model paramters before seeing evidence from the current window.
                    This is either a uniform distribution at the 0th window, or it is the relaxed posterior from
                    the previous window.


        :RETURNS:
        ACs: [list of floats] A list whose two elements are the AC component estimates for the red and IR
            signals in the window after inference, respectively.
        Posts: [list of 2D np.arrays] A list whose two elements are the relaxed posterior distributions over the 
                possible parameters of models that generate the sigal segments.
        """

        # INFER RED
        red_data = np.array([get_AC(x) for x in red_segs])
        red_posterior = bi.update_inference(prior=red_prior, data=red_data)
        red_mu, red_sigma, _ = bi.get_distribution_max(red_posterior)
        relaxed_red_posterior = bi.relax_posterior(red_posterior, filter_sigma=0.25)

        # INFER IR
        ir_data = np.array([get_AC(x) for x in ir_segs])
        ir_posterior = bi.update_inference(prior=ir_prior, data=ir_data)
        ir_mu, ir_sigma, _ = bi.get_distribution_max(ir_posterior)
        relaxed_ir_posterior = bi.relax_posterior(ir_posterior, filter_sigma=0.25)

        # PACKAGE ESTIMATES AND POSTERIORS
        ACs = [red_mu, ir_mu]
        Posts = [relaxed_red_posterior, relaxed_ir_posterior]

        # RETURN INFERRED OUTPUTS
        return ACs, Posts
    
    ## <-------------------  End __do_bayesian_inference() method -------------------> ##

    ## <-------------------  Begin __get_inferred_ratios() method -------------------> ##

    def __get_inferred_ratio(self, red_AC, ir_AC):
        """
        Method for calculating the ratio for each window. Takes the ratio of the most likely
        distribution means from the inference. 

        :PARAMS:
        red_AC: [float] The maximum likelihood value for the red AC component after seeing 
                evidence from the current window.
        ir_AC: [float] The maximum likelihood value for the IR AC component after seeing 
                evidence from the current window.

        :RETURNS:
        r: [float] Estimate of the optical ratio, which is mapped to an SpO2 value.
        """
        r = red_AC / ir_AC
        return r

    ## <-------------------  End __get_inferred_ratios() method -------------------> ##

    ## <-------------------  Begin get_ratio() method -------------------> ##

    def get_ratio_and_posteriors(self, bi, X, i, red_prior, ir_prior):
        """
        Method for implementing full inference algorithm pipeline. Makes use 
        of private methods in the class. This represents the most abstracted way
        to implement the algorithm.

        :PARAMS:
        bi: 

        :RETURNS:


        """

        red_peaks, ir_peaks = self.__segment_and_cluster_window(X, i)

        # CATCH NO PARTITIONS ERROR
        if red_peaks[0,0] == -1 or ir_peaks[0,0] == -1:
            return -1, -1

        ACs, Posts = self.__do_bayesian_inference(bi, red_peaks, ir_peaks, red_prior, ir_prior)

        red_AC, ir_AC = ACs[0], ACs[1]

        ratio = self.__get_inferred_ratio(red_AC, ir_AC)

        return ratio, Posts

    ## <-------------------  End get_ratio() method -------------------> ##

    ## <-------------------  Begin screen_ratio() method -------------------> ##

    def screen_ratio_and_posts(self, r, Posts, r_prev, Posts_prev):
        """ 
        Method for screening errors, which appear as ratio values of r = -1. 
        In the case of this error, this method replaces the current ratio estiamte
        with the previous estimate, and it returns the previous posteriors.
        
        :PARAMS:
        r: [float] The ratio estimate from the current window.
        Posts: [list of 2D np.arrays] The posterior distributions from the current window
                for the red and IR signal models after inference.
        r_prev: [float] The ratio estimate from the previous window.
        Posts_prev: [list of 2D np.arrays] The posterior distributions from the previous window
                for the red and IR signal models after inference. Equivalent to the current window prior.

        :RETURNS:
        ratio: [float] The selected ratio for current window. Either the current window estimate, or the
                estimate from the previous window.
        posteriors: [list of 2D np.arrays] The selected posterior distributions. Either the distributions after
                    inference in the current window, or the distributions from the previous window.
        """

        # CATCH NO PARTITIONS ERROR
        if r == -1:
            return r_prev, Posts_prev
        
        else:
            return r, Posts
        
    ## <-------------------  End screen_ratio() method -------------------> ##
    
    ## <-------------------  Begin trim_ratio_list() method -------------------> ##

    def trim_ratio_list(self, ratios):
        """
        Method for removing failed estimates from leading edge of list. The 
        screen_ratio_and_posts() method requires a previous ratio estimate, so a
        placeholder of -1 is used. If successive windows also fail to produce an estimate,
        this placeholder will be repeated. This method removes the placeholder and any duplicates
        from the list of ratios assembled by iterating through all the windows.

        :PARAMS:
        ratios: [list of ints and floats] A list of the ratio estimates after iterating 
                through all windows.

        :RETURNS:
        ratios: [list of floats] The corrected list of ratio estimates.
        """

        # REMOVE -1 FROM RATIOS
        i = 0
        while i < len(ratios)-1:
            if ratios[i] == -1:
                i+=1
                continue
            else:
                break
            
        ratios = np.array(ratios[i:])
        return ratios

    ## <-------------------  End trim_ratio_list() method -------------------> ##

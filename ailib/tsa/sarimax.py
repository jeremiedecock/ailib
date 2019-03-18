def least_squares(X, y):
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, y)
    invXX = np.linalg.inv(XX)

    theta = np.dot(invXX, Xy)
    
    return theta


class SARIMAX:
    """
    TODO
    
    Parameters
    ----------
    ar_params : array_like, 1d
        Coefficient for autoregressive lag polynomial, including zero lag
    ma_params : array_like, 1d
        Coefficient for moving-average lag polynomial, including zero lag
    d : int
        TODO
    const : float
        TODO
    rng : callable object (function or functor)
        The random number generator used in the process. A normal
        distribution is used by default (np.random.normal()). `None` value
        disables randomness in the SARIMAX process.
    """
    def __init__(self, ar_params=(1.,), ma_params=(), d=0, const=0., rng=np.random.normal):
        #self.ar_params = [float(param) for param in ar_params]
        #self.ma_params = [float(param) for param in ma_params]
        self.ar_params = np.array(ar_params)
        self.ma_params = np.array(ma_params)
        self.d = d
        self.const = const
        self.rng = rng
        
    @property
    def p(self):
        """
        Autoregressive order (the "p" value of the SARIMAX(p,d,q)(P,D,Q)).
        """
        return len(self.ar_params)
    
    @property
    def q(self):
        """
        Moving average order (the "q" value of the SARIMAX(p,d,q)(P,D,Q)).
        """
        return len(self.ma_params)
    
    def sample(self, num_samples=1, past=None, past_random_values=None, exog=None, return_dataframe=False):
        """
        Generate a random sample of a SARIMAX process.
        
        TODO: the following is ARMA not SARIMAX... TODO: how to deal with the eXogeneous part ???
        .. math::
 
            X_t = c + \phi_1 X_{t-1} + \ldots + \phi_p X_{t-p} + \theta_1 \epsilon_{t-1}
                  + \ldots + \theta_q \epsilon_{t-q} + \epsilon_t
        
        Parameters
        ----------
        num_samples : int
            Length of simulated time series.
        past : Array like
            The past samples of the time series.
        
        Returns
        -------
        float or list of float
            A sample of SARIMAX process (of length `num_sample`). Return a float if `num_sample` equals 1.
            Else return a list of floats.
        """
        if past is None:
            if self.rng is not None:
                past = (0.,)
            else:
                past = (1.,)
        
        if past_random_values is None:
            past_random_values = (0.,)

        assert num_samples > 0
        assert len(past) >= 1
        if exog is not None:
            assert num_samples <= len(exog)
        
        # Detrend if d>1
        #past = np.diff(past, n=self.d)
        
        if num_samples == 1:
            
            ar_zeros_padding_len = self.p - len(past)
            ma_zeros_padding_len = self.q - len(past_random_values)

            ar_past_array = np.array(past[::-1])
            ma_past_array = np.array(past_random_values[::-1])

            if ar_zeros_padding_len > 0:
                ar_past_array = np.concatenate((past, np.zeros(shape=ar_zeros_padding_len)))
            else:
                ar_past_array = ar_past_array[:self.p]
            
            if ma_zeros_padding_len > 0:
                ma_past_array = np.concatenate((past_random_values, np.zeros(shape=ma_zeros_padding_len)))
            else:
                ma_past_array = ma_past_array[:self.q]
            
            ar_t = np.dot(self.ar_params, ar_past_array)    # phi_1 X_{t-1} + ... + phi_p X_{t-p}
            ma_t = np.dot(self.ma_params, ma_past_array)    # theta_1 epsilon_{t-1} + ... + theta_q epsilon_{t-q}
            epsilon_t = self.rng() if self.rng is not None else 0.
            y_t = self.const + ar_t + ma_t + epsilon_t
            
            if return_dataframe:
                values = [[y_t,
                           ar_t,
                           ma_t,
                           epsilon_t]]
                samples = pd.DataFrame(values, columns=('x', 'ar', 'ma', 'epsilon'))
            else:
                samples = y_t
                
        else:
            
            # Not very efficient but quite simple
            
            y_hist = [0.]
            ar_hist = [0.]
            ma_hist = [0.]
            epsilon_hist = [0.]
            
            for sample in range(num_samples):
                df_sample = self.sample(num_samples=1,
                                        past=y_hist,
                                        past_random_values=epsilon_hist,
                                        return_dataframe=True)
                
                y_hist.append(df_sample.loc[0, 'x'])
                ar_hist.append(df_sample.loc[0, 'ar'])
                ma_hist.append(df_sample.loc[0, 'ma'])
                epsilon_hist.append(df_sample.loc[0, 'epsilon'])
            
            if return_dataframe:
                values = np.array([y_hist,
                                   ar_hist,
                                   ma_hist,
                                   epsilon_hist]).T
                samples = pd.DataFrame(values, columns=('x', 'ar', 'ma', 'epsilon'))
            else:
                samples = y_hist
                
        for d_index in range(self.d):
            samples = np.cumsum(samples)
            
        return samples
    
    def fit(self, endog, exog=None):
        """
        TODO
        """
        #X = np.full(shape=(len(endog), self.p + self.q + 1), fill_value=np.nan)
        X = np.full(shape=(len(endog), self.p + 1), fill_value=np.nan)
        
        y = np.array(endog).reshape([-1, 1])
        
        X[:,0] = endog
        for p in range(self.p):
            X[:-1,p+1] = X[1:,p]
        
        # TODO: How to do for AR since we don't have epsilon in endog ???
        # See https://stats.stackexchange.com/questions/48026/how-is-the-ma-part-of-arma-solved-for
        
        y = y[:-self.p]
        X = X[:-self.p,1:]
        
        theta = least_squares(X, y)
        
        return theta
    
    def predict(self):
        """
        TODO
        """
        # TODO: equals to self.sample()
        pass
    
    def forecast(self, num_time_steps, exog=None):
        """
        TODO
        """
        # TODO: equals to self.sample()
        pass

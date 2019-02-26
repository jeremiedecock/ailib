#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2019 Jeremie DECOCK (http://www.jdhp.org)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
This module contains a function to evaluate TSA models.
"""

__all__ = ['rolling_forecast']

import numpy as np
import pandas as pd
import sys

def rolling_forecast(data,
                     endog,
                     exog,
                     model,
                     periods_length,
                     num_periods_learned,
                     num_periods_predicted=1,
                     one_eval_per_period=False,
                     error_function=None,
                     error_label="Error",
                     output_file_path="sarimax_results.json"):
    """TODO
    
    Parameters
    ----------
    data : Pandas DataFrame
        TODO

    Returns
    -------
    Pandas DataFrame
        TODO
    """

    df = data
    
    forcast_list = [[] for l in range(periods_length)]
    error_list = []

    step = periods_length if one_eval_per_period else 1

    for start_learn_index in range(0,
                                   len(df) - num_periods_learned - (num_periods_predicted * periods_length),
                                   step):
        
        end_learn_index = start_learn_index + num_periods_learned * periods_length - 1
        
        start_predict_index = end_learn_index+1
        end_predict_index = start_predict_index + num_periods_predicted * periods_length - 1
        
        print("Learn from index {} to index {} and predict from index {} to index {}".format(start_learn_index, end_learn_index, start_predict_index, end_predict_index))

        try:
            model_fit = model.fit()

            model_forecast = model_fit.forecast(periods_length * num_periods_predicted,
                                                exog=df.loc[start_predict_index:end_predict_index, exog].values.reshape(-1, 1))

            forcast_list[start_learn_index % periods_length] += model_forecast.values.tolist()

            if error_function is not None:
                error = error_function(df.loc[start_predict_index:end_predict_index, endog], model_forecast)
                error_list.append(error)
                print('{} = {:0.3f}'.format(error_label, error))
            else:
                error_list.append(float("nan"))

        except Exception as e:

            forcast_list[start_learn_index % periods_length] += [float("nan") for i in range(periods_length * num_periods_predicted)]
            error_list.append(float("nan"))
            print(e, file=sys.stderr)

        # Save results in a JSON file #####################
        
        data = {
            "predictions": forcast_list,
            "error": error_list,
            "time": datetime.datetime.now().isoformat()
        }

        with open(output_file_path, "w") as fd:
            json.dump(data, fd, sort_keys=True, indent=4)  # pretty print format

    return forcast_list, error_list

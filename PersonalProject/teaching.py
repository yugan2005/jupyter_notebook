# coding=utf-8

import pandas as pd
import numpy as np
import math
import plotly.plotly as py
import plotly.offline as po
import plotly.graph_objs as go
import plotly.tools as tls
from datetime import datetime
import pandas_datareader.data as web

def MACD(df, slow_period = 26, fast_period = 12, dif_period = 9):
    slow_EMA = df.ewm(span=slow_period, min_periods=0, adjust=False).mean()
    fast_EMA = df.ewm(span=fast_period, min_periods=0, adjust=False).mean()
    dif = fast_EMA - slow_EMA
    dea = dif.ewm(span=dif_period, min_periods=0, adjust=False).mean()
    macd = (dif-dea)*2
    return macd.to_frame(name='MACD')

def KDJ(df, N=9, M1=3, M2=3):
    RSV = ((df.Close - df.Low.rolling(N, min_periods=0).min())
           /(df.High.rolling(N, min_periods=0).max() - df.Low.rolling(N, min_periods=0).min())*100)
    RSV_s = pd.Series(RSV, index=df.index)
    K = RSV_s.ewm(alpha=1/float(3), min_periods=0, adjust=False).mean()
    D = K.ewm(alpha=1/float(3), min_periods=0, adjust=False).mean()
    J = 3*K - 2*D
    return K, D, J

def get_dwm_ochlm(scode):
    daily = web.DataReader(scode, 'yahoo', start='1990-01-01')
    daily = daily[daily.Volume != 0]
    daily = daily[daily.Volume != daily.Volume.shift()]
    days = daily.reset_index()[['Date']].set_index('Date', drop=False)
    month_last = days.to_period('M').groupby(level=0).last()
    week_last = days.to_period('W').groupby(level=0).last()
    monthly = daily.resample('M').agg({'Open':'first', 'Close':'last', 'High':'max', 'Low':'min'}).dropna()
    monthly.index = month_last['Date']
    weekly = daily.resample('W').agg({'Open':'first', 'Close':'last', 'High':'max', 'Low':'min'}).dropna()
    weekly.index = week_last['Date']

    monthly.loc[:, 'MACD'] = MACD(monthly.Close)['MACD']
    weekly.loc[:, 'MACD'] = MACD(weekly.Close)['MACD']
    daily.loc[:, 'MACD'] = MACD(daily.Close)['MACD']

    monthly.loc[:, 'K'], monthly.loc[:, 'D'], monthly.loc[:, 'J'] = KDJ(monthly)
    weekly.loc[:, 'K'], weekly.loc[:, 'D'], weekly.loc[:, 'J'] = KDJ(weekly)
    daily.loc[:, 'K'], daily.loc[:, 'D'], daily.loc[:, 'J'] = KDJ(daily)

    daily.loc[:, 'Date']=daily.index
    weekly.loc[:, 'Date']=weekly.index
    monthly.loc[:, 'Date']=monthly.index
    month_to_daily = monthly.reindex(daily.index, method='ffill')
    week_to_daily = weekly.reindex(daily.index, method='ffill')

    dwm_ochlm = (daily[['Date', 'High', 'Close', 'Open', 'Low', 'MACD', 'K', 'D', 'J']]
                 .join(week_to_daily, lsuffix='_d', rsuffix='_w'))
    dwm_ochlm = (dwm_ochlm.join(month_to_daily)
                 .rename(columns=
                         {'High':'High_m', 'Close':'Close_m', 'Open':'Open_m',
                          'Low':'Low_m', 'MACD':'MACD_m', 'Date':'Date_m',
                          'K':'K_m', 'D':'D_m', 'J':'J_m'}))

    dwm_ochlm = dwm_ochlm.dropna()

    dwm_ochlm.loc[:, 'grp_d'] = (dwm_ochlm.MACD_d * dwm_ochlm.MACD_d.shift() <0).cumsum()
    dwm_ochlm.loc[:, 'm_neg_d'] = (dwm_ochlm.MACD_d <0)
    dwm_ochlm.loc[:, 'grp_w'] = (dwm_ochlm.MACD_w * dwm_ochlm.MACD_w.shift() <0).cumsum()
    dwm_ochlm.loc[:, 'm_neg_w'] = (dwm_ochlm.MACD_w <0)
    dwm_ochlm.loc[:, 'grp_m'] = (dwm_ochlm.MACD_m * dwm_ochlm.MACD_m.shift() <0).cumsum()
    dwm_ochlm.loc[:, 'm_neg_m'] = (dwm_ochlm.MACD_m <0)
    return dwm_ochlm

def get_plot(scode, endtime):
    dwm_ochlm = get_dwm_ochlm(scode)
    y_max = dwm_ochlm.High_d.max()
    y_min = dwm_ochlm.Low_d.min()
    m_region_date_w = dwm_ochlm[['Date_d', 'grp_w']].groupby('grp_w').first()
    m_region_date_m = dwm_ochlm[['Date_d', 'grp_m']].groupby('grp_m').first()
    trace_vlines = []
    for i in range(len(m_region_date_w)):
        vline_w = go.Scatter(x=[m_region_date_w.Date_d.iloc[i], m_region_date_w.Date_d.iloc[i]],
                             y=[y_min, y_max],
                             mode='lines',
                             line=go.Line(color='green', width=0.5),
                             showlegend=False,
                             name = '')
        trace_vlines.append(vline_w)

    for i in range(len(m_region_date_m)):
        vline_m = go.Scatter(x=[m_region_date_m.Date_d.iloc[i], m_region_date_m.Date_d.iloc[i]],
                             y=[y_min, y_max],
                             mode='lines',
                             line=go.Line(color='red', width=0.5),
                             showlegend=False, 
                             name = '')
        trace_vlines.append(vline_m)

    trace_all = trace_vlines

    trace_h_w = go.Scatter(x=dwm_ochlm.Date_w, y=dwm_ochlm.High_w,
                           line=go.Line(width=0.5, dash = 'dot', color='orange'),
                           name = '周线最高值',
                           legendgroup='周线')
    trace_l_w = go.Scatter(x=dwm_ochlm.Date_w, y=dwm_ochlm.Low_w,
                           line=go.Line(width=0.5, dash = 'dash', color='orange'),
                           name = '周线最低值',
                           legendgroup='周线')
    trace_c_w = go.Scatter(x=dwm_ochlm.Date_w, y=dwm_ochlm.Close_w,
                           line=go.Line(color='orange'),
                           name = '周线收盘',
                           legendgroup='周线')
    trace_h_d = go.Scatter(x=dwm_ochlm.Date_d, y=dwm_ochlm.High_d,
                           line=go.Line(width=0.5, dash = 'dot', color='orchid'),
                           name = '日线最高值',
                           legendgroup='日线')
    trace_l_d = go.Scatter(x=dwm_ochlm.Date_d, y=dwm_ochlm.Low_d,
                           line=go.Line(width=0.5, dash = 'dash', color='orchid'),
                           name = '日线最低值',
                           legendgroup='日线')
    trace_c_d = go.Scatter(x=dwm_ochlm.Date_d, y=dwm_ochlm.Close_d,
                           line=go.Line(color='orchid'),
                           name = '日线收盘',
                           legendgroup='日线')
    trace_h_m = go.Scatter(x=dwm_ochlm.Date_m, y=dwm_ochlm.High_m,
                           line=go.Line(width=0.5, dash = 'dot', color='steelblue'),
                           name = '月线最高值',
                           legendgroup='月线')
    trace_l_m = go.Scatter(x=dwm_ochlm.Date_m, y=dwm_ochlm.Low_m,
                           line=go.Line(width=0.5, dash = 'dash', color='steelblue'),
                           name = '月线最低值',
                           legendgroup='月线')
    trace_c_m = go.Scatter(x=dwm_ochlm.Date_m, y=dwm_ochlm.Close_m,
                           line=go.Line(color='steelblue'),
                           name = '月线收盘',
                           legendgroup='月线')

    trace_all += [trace_h_w, trace_l_w, trace_c_w, trace_h_d, trace_l_d, trace_c_d,trace_h_m, trace_l_m, trace_c_m]

    trace_macd_w = go.Scatter(x=dwm_ochlm.Date_w, y=dwm_ochlm.MACD_w,
                              line=go.Line(color='orange'),
                              name = '周线MACD',
                              yaxis='y2')
    trace_macd_d = go.Scatter(x=dwm_ochlm.Date_d, y=dwm_ochlm.MACD_d,
                              line=go.Line(color='orchid'),
                              name = '日线MACD',
                              yaxis='y2')
    trace_macd_m = go.Scatter(x=dwm_ochlm.Date_m, y=dwm_ochlm.MACD_m,
                              line=go.Line(color='steelblue'),
                              name = '月线MACD',
                              yaxis='y2')

    trace_all += [trace_macd_w, trace_macd_d, trace_macd_m]

    trace_k_w = go.Scatter(x=dwm_ochlm.Date_w, y=dwm_ochlm.K_w,
                           line=go.Line(width=0.5, dash = 'dot', color='orange'),
                           name = '周线K',
                           yaxis='y3',
                           legendgroup='周线KDJ')

    trace_d_w = go.Scatter(x=dwm_ochlm.Date_w, y=dwm_ochlm.D_w,
                           line=go.Line(width=0.5, dash = 'dash', color='orange'),
                           name = '周线D',
                           yaxis='y3',
                           legendgroup='周线KDJ')
    trace_j_w = go.Scatter(x=dwm_ochlm.Date_w, y=dwm_ochlm.J_w,
                           line=go.Line(color='orange'),
                           name = '周线J',
                           yaxis='y3',
                           legendgroup='周线KDJ')

    trace_k_d = go.Scatter(x=dwm_ochlm.Date_d, y=dwm_ochlm.K_d,
                           line=go.Line(width=0.5, dash = 'dot', color='orchid'),
                           name = '日线K',
                           yaxis='y3',
                           legendgroup='日线KDJ')
    trace_d_d = go.Scatter(x=dwm_ochlm.Date_d, y=dwm_ochlm.D_d,
                           line=go.Line(width=0.5, dash = 'dash', color='orchid'),
                           name = '日线D',
                           yaxis='y3',
                           legendgroup='日线KDJ')
    trace_j_d = go.Scatter(x=dwm_ochlm.Date_d, y=dwm_ochlm.J_d,
                           line=go.Line(color='orchid'),
                           name = '日线J',
                           yaxis='y3',
                           legendgroup='日线KDJ')
    trace_k_m = go.Scatter(x=dwm_ochlm.Date_m, y=dwm_ochlm.K_m,
                           line=go.Line(width=0.5, dash = 'dot', color='steelblue'),
                           name = '月线K',
                           yaxis='y3',
                           legendgroup='月线KDJ')
    trace_d_m = go.Scatter(x=dwm_ochlm.Date_m, y=dwm_ochlm.D_m,
                           line=go.Line(width=0.5, dash = 'dash', color='steelblue'),
                           name = '月线D',
                           yaxis='y3',
                           legendgroup='月线KDJ')
    trace_j_m = go.Scatter(x=dwm_ochlm.Date_m, y=dwm_ochlm.J_m,
                           line=go.Line(color='steelblue'),
                           name = '月线J',  
                           yaxis='y3',
                           legendgroup='月线KDJ')

    trace_all += [trace_k_w, trace_d_w, trace_j_w, trace_k_d, trace_d_d, trace_j_d, trace_k_m, trace_d_m, trace_j_m]
    x_show_range=[dwm_ochlm.Date_d[0].value/1e6, pd.Timestamp(endtime).value/1e6]
    y_show_range =[dwm_ochlm.loc[dwm_ochlm.Date_d < pd.Timestamp(endtime), 'Low_d'].min(),
                   dwm_ochlm.loc[dwm_ochlm.Date_d < pd.Timestamp(endtime), 'High_d'].max()*1.05]
    layout = go.Layout(yaxis=dict(domain=[0, 0.6],
                                  range=y_show_range),
                       yaxis2=dict(domain=[0.6, 0.8]),
                       yaxis3=dict(domain=[0.8, 1]),
                       height=1000,
                       legend=dict(traceorder='reversed'),
                       xaxis=dict(range=x_show_range))

    fig = go.Figure(data = trace_all, layout=layout)
    po.iplot(fig, show_link=False)
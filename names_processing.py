import numpy as np
import pandas as pd
import csv
import os
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib import
import colorsys
# from wordcloud import WordCloud

# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

import datetime as dt
import pickle

import requests
from bs4 import BeautifulSoup
# from requests_futures.sessions import FuturesSession

pd.set_option('display.max_rows',150)

currentyear = 2022  # the last year of data available in the "names" folder
maxyear = currentyear + 30


#Set figure options:

#Yes, I want this as a universal default across all notebooks.
# matplotlib.rcParams['font.sans-serif'] = ['Helvetica']

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
# plt.rc('legend', fontsize=12)    # legend fontsize
# plt.rc('figure', titlesize=10)  # fontsize of the figure title



#Load in preprocessed data. The "namelife_S_full" files are by
#far the largest, at 3-4 gigs.

alive_F = pd.read_pickle('./data/life_F_df.pkl')
alive_M = pd.read_pickle('./data/life_M_df.pkl')
alive_F_p = pd.read_pickle('./data/life_F_p_df.pkl')
alive_M_p = pd.read_pickle('./data/life_M_p_df.pkl')

#alive_M_t = alive_M.values.astype('float64').T
#alive_F_t = alive_F.values.astype('float64').T

totalnames_table = pd.read_pickle('./data/Total_soc_cards.pkl')
names_df_trim = pd.read_pickle('./data/names_df_trim.pkl')

namelife_F_full = np.load('./data/namelife_F_full.npy')
namelife_F_base = np.load('./data/namelife_F_base.npy')
namelife_F_name = pd.read_pickle('./data/namelife_F_name.pkl')
namebirth_F = np.load('./data/namebirth_F.npy')
namelife_M_full = np.load('./data/namelife_M_full.npy')
namelife_M_base = np.load('./data/namelife_M_base.npy')
namelife_M_name = pd.read_pickle('./data/namelife_M_name.pkl')
namebirth_M = np.load('./data/namebirth_M.npy')



#OPTIONAL VISUALIZATION.

#Set variable here to eyeball
showplot = True

if showplot:
    #Example name:
    tgt_name = 'Lakynn'
    tgt_sex = 'F'

    if tgt_sex == 'M':
        tempy = namelife_M_base[np.where(namelife_M_name==tgt_name),:][0,0,:]
    else:
        tempy = namelife_F_base[np.where(namelife_F_name==tgt_name),:][0,0,:]
    plt.plot(np.arange(1880, maxyear), tempy)
    tempdf = names_df_trim[(names_df_trim['Name']==tgt_name) & (names_df_trim['Sex']==tgt_sex)]
    plt.plot(tempdf['Year'], tempdf['Number'])
    plt.plot([2019, 2019],[0,1.1*np.max(tempy)],color=[0,0,0,0.1])
    plt.show()


#OPTIONAL VISUALIZATION: THIS ONE IS HANDY

if 1:
    #Example life of name:

    nameind = 2015

    fig, ax = plt.subplots(1,1)
    im = plt.imshow(namelife_F_full[nameind,:,:])
    ax.set_title('(Living "' + str(namelife_F_name[nameind]) + '"s in X year, with Y birth year)')
    ax.set_xlabel('Year')
    ax.set_xticks(np.arange(0,maxyear-1880,10))
    ax.set_xticklabels(np.arange(1880,maxyear,10), rotation=90)

    plt.ylabel('Birth year (1880-2019)')
    plt.colorbar(im)
    plt.show()

    ax = plt.subplot(1,1,1)
    ax.plot(namebirth_F[nameind,:])
    ax2 = ax.twinx()
    ax2.plot(namelife_F_base[nameind,:], color='red')
    # ax.plot([139, 139],[0,4.5e10],color=[0,0,0,0.1])
    #ax.set_xlim([0,138])
    year_x = [0,20,40,60,80,100,120,140,160,180]
    ax.set_xticks(year_x)
    year_tick = (np.array(year_x)+1880).astype('str')
    ax.set_xticklabels(year_tick)
    ax.set_ylabel('Born per year (blue)')
    ax2.set_ylabel('Alive per year (red)')
    ax.set_title(namelife_F_name[nameind])
    plt.show()

# Takes well under 1 minute now.

# Average life-weighted year-of birth for each name for each year:
# DOES NOT CHECK WHETHER ANYONE IS STILL ALIVE IN X YEAR BEFORE
# MATH, SO EXPECT DIVIDE BY ZERO ERRORS. NaNs are totally fine for
# showing lack of alive people, but not pretty coding.

# It is possible to do this with matrix multiplication. It would
# take longer to get running right than just running this.

agevec = np.arange(1880, maxyear).T

aliveshape = namelife_M_full.shape
namelife_M_yob = np.zeros(namelife_M_base.shape)
namelife_sum = np.sum(namelife_M_full, axis=1)
for rowind in np.arange(0, aliveshape[0]):
    tempsum = namelife_sum[rowind, :]
    namelife_M_yob[rowind, :] = np.matmul(agevec, namelife_M_full[rowind, :, :]) / namelife_sum[rowind, :]

aliveshape = namelife_F_full.shape
namelife_F_yob = np.zeros(namelife_F_base.shape)
namelife_sum = np.sum(namelife_F_full, axis=1)
for rowind in np.arange(0, aliveshape[0]):
    namelife_F_yob[rowind, :] = np.matmul(agevec, namelife_F_full[rowind, :, :]) / namelife_sum[rowind, :]


if showplot:
    plt.plot(np.arange(1880, maxyear), namelife_M_base[0,:])
    plt.plot(np.arange(1880, maxyear),
             -1*(namelife_M_yob[0,:] - np.arange(1880,maxyear)))
    plt.title(namelife_M_name[0])
    plt.legend(['Number alive','Avg age'])
    plt.show()


#

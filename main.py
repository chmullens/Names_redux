if __name__ == '__main__':
    print('(running main.py)')

import pandas as pd
import os
import time
import requests
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

# Generate the actuarial data files (defaults to actuarialdata directory)
import actuarial_calcs

actuarial_calcs  # (this just runs the script, expect performance warnings)


# Set figure options:

# Yes, I want this as a universal default for the project.
mpl.rcParams['font.sans-serif'] = ['Helvetica']

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
# plt.rc('legend', fontsize=12)    # legend fontsize
# plt.rc('figure', titlesize=10)  # fontsize of the figure title

# TODO (low priority): Automate currentyear (pull from data or move to config)
currentyear = 2023  # the last year of data available in the "names" folder
maxyear = currentyear + 30

# Grab the actuarial data:

# The data loaded below is processed in the "US_Lifespan" notebook.
# The four resulting files total less than 1MB, Basically, from a
# set of decade-precision life tables, I interpolated values for
# each year. Does an acceptable job, though it largely misses the
# impact of single-year changes that significantly impacted narrow
# age ranges, like WWI and WWII. Still gets the broad strokes. I do
# some further work below to estimate pre-1900 numbers.

# TODO: move filenames to config
alive_F = pd.read_pickle('actuarialdata/life_F_df.pkl')
alive_M = pd.read_pickle('actuarialdata/life_M_df.pkl')
alive_F_p = pd.read_pickle('actuarialdata/life_F_p_df.pkl')
alive_M_p = pd.read_pickle('actuarialdata/life_M_p_df.pkl')

# These approximations are fine for now, I'm not deeply interested in the
# actuarial outcomes of single-year groups.

# Get the yearly total of all names (including sub-5) from Social Security
# Admin, by year (useful for normalization)


# Gets the data raw from the website and pickles it.
# TODO: move reload option to config, move functionality to separate file?
reload_birthtotals = True

if reload_birthtotals:
    response = requests.get('https://www.ssa.gov/oact/babynames/numberUSbirths.html')
    soup = BeautifulSoup(response.content, 'html.parser')

    header = soup.find("tr")
    temptable = header.find_parent("table")
    totalnames_table = pd.read_html(temptable.prettify())[0]
    totalnames_table.rename(columns={'Year of  birth':'Year of birth'}, inplace=True)
    totalnames_table.set_index('Year of birth', inplace=True)

    totalnames_table.to_pickle('Total_soc_cards.pkl')
else:
    totalnames_table = pd.read_pickle('Total_soc_cards.pkl')

# Load the year-of-birth data, in "yobYYYY.txt" format csv files.
#
# Data source: https://www.ssa.gov/OACT/babynames/limits.html,
# has national zip file that unpacks to the files below.

yearfiles = os.listdir('./namedata/names/')
yearfiles.sort()

# Load the year-of-birth data, in "yobYYYY.txt" format csv files.
#
# Data source: https://www.ssa.gov/OACT/babynames/limits.html,
# has national zip file that unpacks to the files below.

yearfiles = os.listdir('./namedata/names/')
yearfiles.sort()

names_temp = []
for ftemp in yearfiles:
    if ftemp[-3:] == 'txt':
        dftemp = pd.read_csv('./namedata/names/' + ftemp,
                             names=['Name', 'Sex', 'Number'])
        dftemp['Year'] = int(ftemp[3:7])
        names_temp.append(dftemp)

names_df = pd.concat(names_temp)
names_df.info()
# These births will be called "nameset births" from here on out.

# Reset the indexes to unduplicate them, since each stack was indexed
# separately; keeping the old indices is handy though, so default
# 'drop = false' option is kept.
#
# Yes, the sorting matters. It makes plotting way easier later on, and
# I could definitely use it to speed up the name structure indexing.

names_df = names_df.reset_index().sort_values(['Name','Year'])

# Above is raw, not controlled for population or births. What was
# the total as a fraction of the year's name births? Currently only
# using the over-5-names as the core comparison. Could pull from
# the social security total above, here, in future improvements.

year_total = names_df.groupby('Year').sum()

# Join year_total w/ names_df to get what fraction of total births
# that year made up of the given name/sex pair:
names_df = pd.merge(left=names_df, right=totalnames_table,
                    left_on='Year', right_on='Year of birth',
                    how='left')
names_df['Fraction'] = names_df['Number']/names_df['Total']

# Not normalizing to number of male or female births, just total.
# Possible future improvement, since likelihood of registering
# appears to have had some sex bias early on.
names_df.drop(columns=['Male','Female','Total'], inplace=True)


# Calculate total number of births for each name:
totalbirths_byname = names_df[['Name','Sex','Number']].groupby(['Name','Sex']).sum()

trimvalue = 20
# Limit dataset to names that had more than 20 total
# people with that name:
keepnames = totalbirths_byname[totalbirths_byname['Number'] > trimvalue]

# Join on name/sex pairs to keep only ones included above
names_df_trim = pd.merge(left=names_df, right=keepnames, on=['Name','Sex'], how='inner')
# Handling number column ambiguity, not pretty but works fine
names_df_trim['Number'] = names_df_trim['Number_x']
names_df_trim.drop(columns=['Number_x', 'Number_y'], inplace=True)
val1 = len(names_df.groupby(['Name','Sex']).size())
val2 = len(names_df_trim.groupby(['Name','Sex']).size())
val1a = len(names_df)
val2a = len(names_df_trim)

# Display results
print('Pretrim -> posttrim, minimum {} births:\n'.format(trimvalue))
print('Names (total of M and F separately):')
print(val1, '->', val2)
print('Name/year records:')
print(val1a, '->', val2a)
print('\nRemoves {:.2f} of names while only removing {:.3f} of total birth records.'.format((val1-val2)/val1, (val1a-val2a)/val1a))

# Cache
names_df_trim.to_pickle('names_df_trim.pkl')

# Grab the individual names
names_list = names_df_trim['Name'].unique()

# Data analysis check:
# Calculate how many unique names per year (useful check for trim)

numnames = []
for year in names_df['Year'].sort_values().unique():
    nn_y = year
    nn_b = len(names_df[names_df['Year'] == year]['Name'].unique())
    nn_c = len(names_df_trim[names_df_trim['Year'] == year]['Name'].unique())
    numnames.append([nn_y, nn_b, nn_c])

numnames = pd.DataFrame(numnames, columns=['Year', 'N_names_tot', 'N_names_trim']).set_index('Year')


# NOTES:
# Names dataset is a pretty good representation of all births, starting about 1915.
#
# The names dataset we're working with, to review, is published by the Social Security
# Administration and includes all recorded baby names with 5 or more births during that
# year. This dataset does a reasonably good job tracking total births starting around
# 1915. By 1920, the names database was covering at least 75% of all births in the
# census estimates, up to 97.5% in 1960. Since then, the coverage has declined; given
# the striking increase in total number of unique names observed per year, it is likely
# that decline mostly reflects increasing numbers of excluded low-count names.
#
# Pre-1915 data has limited coverage, but we'll assume it reflects name distribution
# Most of the data is good enough to work with, but pre-1915, we'll be relying partly
# on extrapolation. For the purposes of this analysis, I'll assume that the baby names
# recorded from 1880 to 1915 accurately represent the overall distribution of names in
# the population. There are a variety of reasons we might expect that assumption to be
# flawed, such as bias from which states were reporting at the time, class differences,
# immigration recency affecting likelihood of registration, urban/rural differences,
# etc., but the distribution of the most popular baby names before and after the big
# registration spike in the 1910s is pretty similar, so there's a good chance those
# factors don't make a *huge* difference at least.
#
# If I were performing this analysis in 30 years, I'd probably just skip that earlier
# data, but if we skip it now, we have very few years with reasonably-complete birth
# cohort data.

# Living names:
#
# The goal: A dataset of the names of currently living US-born people for each year,
# that covers people born 1880 to the current day. The number of living people who
# are over 90 has historically been small, so we'll have a pretty complete population
# estimate of US-born names starting in 1970 or so.
#
# The actuarial table estimates start in 1900, and have decade-by-decade estimates.
# I'll use a few assumptions here:
#
# 1. Assumption: All years between the decade estimates (i.e. 1901-1909) can be
# approximated by linear interpolation. This should be pretty accurate, since the
# decade data includes single-year resolution of cohort life expectancy. This
# interpolation is already performed in the loaded tables.
# 2. Assumption: All birth cohorts from 1880-1899 follow the 1900 lifespan pattern,
# scaled to the expected values found in the 1900 period life table. This is likely
# to be inaccurate given how fast childhood mortality was changing after 1900, but
# will be partially corrected by normalizing to known values in the 1900 dataset,
# like number of individuals who are 20 in 1900.
# 3. Assumption: Names do not predict actuarial outcomes. Likely to be somewhat
# inaccurate, since names are associated with demographics, but it's both less
# difficult and less prone to error than attempting to categorize names by
# demographics.


# Reprocessing actuarial data:
#
# Assume 1880-1900 lifespans have the same trajectory as 1900, then
# normalize the starting births to match the number from that cohort
# who were alive in 1900. We can get that data from the period life
# tables; they predict how long someone will live based on how many
# people who are X age in the previous year are still alive.

# TODO: MOVE THIS TO actuarial_calcs.py, that's where it should live
alive_F_prescale = alive_F.copy() # Keep originals for reference
alive_M_prescale = alive_M.copy() # Keep originals for reference

# Add rows for 1880-1899:
for year in np.arange(1880, 1900):
    alive_F.loc[year,:] = alive_F.loc[1900,:]
    alive_F_prescale.loc[year,:] = alive_F_prescale.loc[1900,:]
    # Multiply that year's aliveness vector by a scale number that
    # makes the aliveness in 1900 line up with the "p" (period)
    # data for 1900, which is based on past survival.
    alive_F.loc[year,:] = alive_F.loc[year,:] * (alive_F_p.loc[1900,1900-year-1])/(alive_F.loc[1900,1900-year-1])
alive_F = alive_F.sort_index()
alive_F_prescale = alive_F_prescale.sort_index()

for year in np.arange(1880, 1900):
    alive_M.loc[year,:] = alive_M.loc[1900,:]
    alive_M_prescale.loc[year,:] = alive_M_prescale.loc[1900,:]
    alive_M.loc[year,:] = alive_M.loc[year,:] * (alive_M_p.loc[1900,1900-year-1])/(alive_M.loc[1900,1900-year-1])
alive_M = alive_M.sort_index()
alive_M_prescale = alive_M_prescale.sort_index()

# In short, the alive_X structures are the number of people born in [row]
# year that have/will survive for [col] years. I'm using this "cohort data"
# rather than period data, since I'm most interested in tracking individual
# lives across the timespan. THIS WILL MEAN THAT THE LIFE TABLE DOES NOT
# START AT 100,000 FOR THESE VALUES. That won't cause any analysis problems,
# but should be kept in mind because it's nonstandard for actuarial data.


# Define functions/vars to build the big matrix of aliveness:

# It may be possible to pull this off with joins or array ops,
# but there's a LOT of looking at multiple rows to solve for
# a single set of values. Could also be inefficient.

# Most likely, the way to get this ACTUALLY fast is to pre-sort
# by name and year, then use find operations to grab ranges of
# data at a time. Match 2 items, versus matching 140. But since
# I can save/load the data faster than I could process it out,
# this one-time 5-minute run (for M and F total) is fine.

# Variables:
#    nyears, total range of years of birth data (140, through 2019)
#    ndeath, total range of alive_X data for each year (120)
#    nyearsmax, last projected year (2050 - 1880)
nyears = len(numnames)
ndeath = alive_M.shape[1]  # once
nyearsmax = maxyear - 1880

# Rebuild the alive matrix to sit across the target years:
# For each year's row, start the alive data from that year
# at that year's index, instead of at zero.
alive_M_re = np.zeros([nyears, nyears + ndeath])
for n in np.arange(0, nyears):
    tgt_year = n + 1880
    alive_M_re[n, n:ndeath + n] = alive_M.loc[tgt_year, :]
alive_F_re = np.zeros([nyears, nyears + ndeath])
for n in np.arange(0, nyears):
    tgt_year = n + 1880
    alive_F_re[n, n:ndeath + n] = alive_F.loc[tgt_year, :]

baseyear_df = pd.DataFrame(index=np.arange(1880, currentyear+1))


def gen_name_alive(name, sex, namedf, base_alive):
    # global vars used:
    #    baseyear_df
    #    nyears
    #    ndeath
    #    nyearsmax

    tgt_ind = (namedf['Name'] == name) & (namedf['Sex'] == sex)

    name_chunk = namedf[tgt_ind]
    name_chunk = name_chunk.set_index('Year')
    currentbirths = baseyear_df.join(name_chunk['Number']).fillna(0).values.T

    alive_vec = np.matmul(currentbirths, base_alive[:, :nyearsmax])
    alive_arr = base_alive[:, :nyearsmax] * np.tile(currentbirths.T, [1, nyearsmax])
    return alive_vec, alive_arr, currentbirths


# Takes about 3 minutes.

# Why am I splitting up the name chunks? Partly processing,
# partly making it easier to tell apart "Emily", the female
# name, from "Emily", the male name. Yes, there are a lot of
# names where there is gender overlap. I want to count them
# separately, since they can represent separate trends.

# Could function this, but it uses and generates a pile of
# variables that I want to keep for both cases, and only
# happens twice.

# 'names_F', full dataset with only female names
# 'nameset_F', full list of female names
# 'tempyears', how many years in the dataset
names_F = names_df_trim[names_df_trim['Sex'] == 'F'].sort_values('Name')
nameset_F = names_F['Name'].unique()
tempyears = names_F['Year'].unique()

# Array dimensions: Number of names, number of initial years, max year
namelife_F_full = np.zeros([len(nameset_F), nyears, nyearsmax])
namelife_F_base = np.zeros([len(nameset_F), nyearsmax])
namebirth_F = np.zeros([len(nameset_F), nyears])
namelife_F_name = []

print('Number of names: ' + str(len(nameset_F)))

namecount = 0

starttime1 = time.time()
starttime2 = time.time()
print('\nBEGIN:')

binsize = 100
# Make the bin indexes:
bin_indexes = np.arange(0, len(nameset_F), binsize)
if (len(nameset_F) % binsize):
    bin_indexes = np.append(bin_indexes, len(nameset_F) + 1)

for n in range(len(bin_indexes) - 1):
    # Grab this bin's set of data:
    # print(n)
    # print(bin_indexes[n], bin_indexes[n+1])
    bininds = [bin_indexes[n], bin_indexes[n + 1]]
    nameset_sub = nameset_F[bininds[0]:bininds[1]]
    names_F_sub = names_F[names_F['Name'].isin(nameset_sub)]

    for nameind in np.arange(0, len(nameset_sub)):
        nametemp = nameset_sub[nameind]
        a_v, a_a, c_b = gen_name_alive(nametemp, 'F', names_F_sub, alive_F_re)

        namelife_F_base[namecount, :] = a_v
        namelife_F_full[namecount, :, :] = a_a
        namebirth_F[namecount, :] = c_b
        namelife_F_name.append(nametemp)

        namecount += 1

    print('bininds ' + str(bininds) + ', ' +
          str(round(time.time() - starttime1, 3)) + 's tot, ' +
          str(round(time.time() - starttime2, 3)) + 's per ' + str(binsize) + '-loop')
    starttime2 = time.time()

# Scale for base population of 100000 in actuarial data:
namelife_F_full = (namelife_F_full / 100000).astype('int32')
namelife_F_base = namelife_F_base / 100000
namelife_F_name = pd.Series(namelife_F_name)

# Takes under 2 minutes.

# 'names_M', full dataset with only male names
# 'nameset_M', full list of male names
# 'tempyears', how many years in the dataset
names_M = names_df_trim[names_df_trim['Sex'] == 'M'].sort_values('Name')
nameset_M = names_M['Name'].unique()
tempyears = names_M['Year'].unique()

# Array dimensions: Number of names, number of initial years, max year
namelife_M_full = np.zeros([len(nameset_M), nyears, nyearsmax])
namelife_M_base = np.zeros([len(nameset_M), nyearsmax])
namebirth_M = np.zeros([len(nameset_M), nyears])
namelife_M_name = []

print('Number of names: ' + str(len(nameset_M)))

namecount = 0

starttime1 = time.time()
starttime2 = time.time()
print('\nBEGIN:')

binsize = 100
# Make the bin indexes:
bin_indexes = np.arange(0, len(nameset_M), binsize)
if (len(nameset_M) % binsize):
    bin_indexes = np.append(bin_indexes, len(nameset_M) + 1)

for n in range(len(bin_indexes) - 1):
    # Grab this bin's set of data:
    bininds = [bin_indexes[n], bin_indexes[n + 1]]
    nameset_sub = nameset_M[bininds[0]:bininds[1]]
    names_M_sub = names_M[names_M['Name'].isin(nameset_sub)]

    for nameind in np.arange(0, len(nameset_sub)):
        nametemp = nameset_sub[nameind]
        a_v, a_a, c_b = gen_name_alive(nametemp, 'M', names_M_sub, alive_M_re)

        namelife_M_base[namecount, :] = a_v
        namelife_M_full[namecount, :, :] = a_a
        namebirth_M[namecount, :] = c_b
        namelife_M_name.append(nametemp)

        namecount += 1

    print('bininds ' + str(bininds) + ', ' +
          str(round(time.time() - starttime1, 3)) + 's tot, ' +
          str(round(time.time() - starttime2, 3)) + 's per ' + str(binsize) + '-loop')
    starttime2 = time.time()

# Scale for base population of 100000 in actuarial data:
namelife_M_full = (namelife_M_full / 100000).astype('int32')
namelife_M_base = namelife_M_base / 100000
namelife_M_name = pd.Series(namelife_M_name)


#Note that variables are also cached in several earlier locations.

np.save('namelife_F_full.npy',namelife_F_full)
np.save('namelife_F_base.npy',namelife_F_base)
np.save('namebirth_F.npy',namebirth_F)
namelife_F_name.to_pickle('namelife_F_name.pkl')

np.save('namelife_M_full.npy',namelife_M_full)
np.save('namelife_M_base.npy',namelife_M_base)
np.save('namebirth_M.npy',namebirth_M)
namelife_M_name.to_pickle('namelife_M_name.pkl')




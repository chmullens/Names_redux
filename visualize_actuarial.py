# OPTIONAL DATA VISUALIZATION

alive_M_t = alive_M.values.astype('float64').T
alive_F_t = alive_F.values.astype('float64').T

# Exploratory comparison:
fig = plt.figure(figsize=[8, 8])
ax = plt.subplot(111, aspect='equal')
cax = ax.imshow(alive_M_t, cmap='plasma')
plt.title('Number of US male births from X year alive at Y age', pad=15)
plt.ylabel('Age')
plt.xlabel('Year of birth')

# Get contours:
contours_alive = np.zeros([alive_M_t.shape[1], 11])
alive_num = []
for n in range(10):
    # Note: Below method for finding first index at which the comparison
    # is true is not necessarily the best, it may cause spotty outputs for
    # unsorted data for example, but allows multidimensional comparison
    # natively with use of the 'axis' variable. Where, nonzero, and
    # searchsorted use a 1D array and would have to be reindexed. Would be
    # good flatten/rebuild matrix practice at some point in the future.
    #
    # Good discussion here:
    # https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value

    # Find first index at which fewer than X people are alive:
    contours_alive[:, n] = np.argmax(alive_M_t < (100000 - n * 10000), axis=0)

    alive_num.append(100000 - n * 10000)

contours_alive[:, 10] = np.argmax(alive_M_t < 1, axis=0)
contours_alive[contours_alive[:, 10] == 0, 10] = np.max(contours_alive[:, 10])
alive_num.append(0)

plt.plot(contours_alive[:, 1:], color=[1, 1, 1], linewidth=2)

ax.plot([120, 0], [0, 120], 'k', linewidth=1)
txtemp = ax.text(80, 20, 'Living people, 2020', size=12, rotation=-45)

plt.ylim([-0.5, 119.5])
plt.xlim([.5, 120])

ax.set_xticks(np.arange(0, 121, 10))
ax.set_xticklabels((np.arange(0, 121, 10) + 1900).astype('str'))
ax.set_yticks(np.arange(0, 121, 10))

cbar = fig.colorbar(cax,
                    ticks=alive_num, fraction=0.046, pad=0.04,
                    label='No. of individuals alive out of 100k', )
cbar.ax.set_yticklabels(alive_num)  # vertically oriented colorbar
cbar_lims = cbar.ax.get_xlim()
cbar.ax.invert_yaxis()
cbar.ax.hlines(alive_num, cbar_lims[0], cbar_lims[1], colors='w', linewidth=1)
fig.set_facecolor('white')

plt.show()

# Redo for F:
fig = plt.figure(figsize=[8, 8])
ax = plt.subplot(111, aspect='equal')
cax = ax.imshow(alive_F_t, cmap='plasma')
plt.title('Number of US female births from X year alive at Y age', pad=15)
plt.ylabel('Age')
plt.xlabel('Year of birth')

# Get contours:
contours_alive = np.zeros([alive_F_t.shape[1], 11])
alive_num = []
for n in range(10):
    # Note: Below method for finding first index at which the comparison
    # is true is not necessarily the best, it may cause spotty outputs for
    # unsorted data for example, but allows multidimensional comparison
    # natively with use of the 'axis' variable. Where, nonzero, and
    # searchsorted use a 1D array and would have to be reindexed. Would be
    # good flatten/rebuild matrix practice at some point in the future.
    #
    # Good discussion here:
    # https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value

    # Find first index at which fewer than X people are alive:
    contours_alive[:, n] = np.argmax(alive_F_t < (100000 - n * 10000), axis=0)

    alive_num.append(100000 - n * 10000)

contours_alive[:, 10] = np.argmax(alive_F_t < 1, axis=0)
contours_alive[contours_alive[:, 10] == 0, 10] = np.max(contours_alive[:, 10])
alive_num.append(0)

plt.plot(contours_alive[:, 1:], color=[1, 1, 1], linewidth=2)

ax.plot([120, 0], [0, 120], 'k', linewidth=1)
txtemp = ax.text(80, 20, 'Living people, 2020', size=12, rotation=-45)

plt.ylim([-0.5, 119.5])
plt.xlim([.5, 120])

ax.set_xticks(np.arange(0, 121, 10))
ax.set_xticklabels((np.arange(0, 121, 10) + 1900).astype('str'))
ax.set_yticks(np.arange(0, 121, 10))

cbar = fig.colorbar(cax,
                    ticks=alive_num, fraction=0.046, pad=0.04,
                    label='No. of individuals alive out of 100k', )
cbar.ax.set_yticklabels(alive_num)  # vertically oriented colorbar
cbar.ax.invert_yaxis()
cbar_lims = cbar.ax.get_xlim()
cbar.ax.hlines(alive_num, cbar_lims[0], cbar_lims[1], colors='w', linewidth=1)
fig.set_facecolor('white')

plt.show()
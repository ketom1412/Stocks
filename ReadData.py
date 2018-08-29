import pandas as pd 
import matplotlib.pyplot as plt

perf = pd.read_pickle('dma.pickle') # read in perf DataFrame
print(perf)
perf.head()

# %pylab inline
# figsize(12, 12)

ax1 = plt.subplot(211)
perf.portfolio_value.plot(ax=ax1)
ax1.set_ylabel('Portfolio Value')
ax2 = plt.subplot(212, sharex=ax1)
perf.AAPL.plot(ax=ax2)
ax2.set_ylabel('AAPL Stock Price')
plt.show()
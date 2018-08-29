QUANDL_API_KEY=aizTht3-SpeeCyXqeYGZ zipline ingest -b quandl
zipline run -f dual_moving_average.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle
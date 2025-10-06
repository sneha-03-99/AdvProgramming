import datetime as dt
from datetime import date,timedelta 
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick # optional may be helpful for plotting percentage
import numpy as np
import pandas as pd
import seaborn as sb # optional to set plot theme
from IPython.display import display
sb.set_theme() # optional to set plot theme

DEFAULT_START = dt.date.isoformat(dt.date.today() - dt.timedelta(365))
DEFAULT_END = dt.date.isoformat(dt.date.today())

class Stock:
    def __init__(self, symbol, start=DEFAULT_START, end=DEFAULT_END):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data = self.get_data()


    def get_data(self):
        """method that downloads data and stores in a DataFrame
           uncomment the code below wich should be the final two lines 
           of your method"""
        data= yf.download(self.symbol, start= self.start, end=self.end)
        data.index= pd.to_datetime(data.index)
        data.dropna(inplace=True)
        self.calc_returns(data)
        return data
        pass

    
    def calc_returns(self, data):
        """method that adds change and return columns to data"""
        data["change"] = data["Close"].diff()
        data["instant_return"] = np.log(data["Close"]).diff().round(4)
        pass

    
    def plot_return_dist(self):
        """method that plots instantaneous returns as histogram"""
        plt.figure(figsize=(8, 5))
        plt.hist(self.data["instant_return"].dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.title(f"Distribution of Instantaneous Returns for {self.symbol}", fontsize=14)
        plt.xlabel("Instantaneous Return", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()
        pass


    def plot_performance(self):
        """method that plots stock object performance as percent """
        base_price = self.data["Close"].iloc[0]
        self.data["percent_gain"] = ((self.data["Close"] - base_price) / base_price) * 100
    
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data["percent_gain"], label="Percent Gain/Loss", color="red", linewidth=2)
        plt.axhline(0, color="black", linestyle="--", linewidth=1)  
        plt.title(f"{self.symbol} Performance (% Gain/Loss)", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Percent Change (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.show()
        pass
                  



def main():
    # uncomment (remove pass) code below to test
    test = Stock(symbol='DELL') # optionally test custom data range
    print(test.data)
    display(test.data.head())
    test.plot_performance()
    test.plot_return_dist()
    pass

if __name__ == '__main__':
    main() 
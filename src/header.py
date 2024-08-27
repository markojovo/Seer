import datetime
import random
import tensorflow as tf
import numpy as np
from transformers import TFBertModel
import multiprocessing
from tensorflow.keras.layers import Layer


random.seed(123456789) #so it's same every time

TotalStockList =  sorted(['MMM','AOS','ABT','ABBV','ABMD','ACN','ATVI','ADM','ADBE','AAP','AMD','AES','AFL','A','APD','AKAM','ALK','ALB','ARE','ALGN','ALLE','LNT','ALL','GOOGL','GOOG','MO','AMZN','AMCR','AEE','AAL','AEP','AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','ADI','ANSS','ANTM','AON','APA','AAPL','AMAT','APTV','ANET','AJG','AIZ','T','ATO','ADSK','ADP','AZO','AVB','AVY','BKR','BLL','BAC','BBWI','BAX','BDX','BRK.B','BBY','BIO','TECH','BIIB','BLK','BK','BA','BKNG','BWA','BXP','BSX','BMY','AVGO','BR','BRO','BF.B','CHRW','CDNS','CZR','CPB','COF','CAH','KMX','CCL','CARR','CTLT','CAT','CBOE','CBRE','CDW','CE','CNC','CNP','CDAY','CERN','CF','CRL','SCHW','CHTR','CVX','CMG','CB','CHD','CI','CINF','CTAS','CSCO','C','CFG','CTXS','CLX','CME','CMS','KO','CTSH','CL','CMCSA','CMA','CAG','COP','ED','STZ','CPRT','GLW','CTVA','COST','CTRA','CCI','CSX','CMI','CVS','DHI','DHR','DRI','DVA','DE','DAL','XRAY','DVN','DXCM','FANG','DLR','DFS','DISCA','DISCK','DISH','DG','DLTR','D','DPZ','DOV','DOW','DTE','DUK','DRE','DD','DXC','EMN','ETN','EBAY','ECL','EIX','EW','EA','LLY','EMR','ENPH','ETR','EOG','EFX','EQIX','EQR','ESS','EL','ETSY','RE','EVRG','ES','EXC','EXPE','EXPD','EXR','XOM','FFIV','FB','FAST','FRT','FDX','FIS','FITB','FRC','FE','FISV','FLT','FMC','F','FTNT','FTV','FBHS','FOXA','FOX','BEN','FCX','GPS','GRMN','IT','GNRC','GD','GE','GIS','GM','GPC','GILD','GPN','GL','GS','HAL','HBI','HAS','HCA','PEAK','HSIC','HES','HPE','HLT','HOLX','HD','HON','HRL','HST','HWM','HPQ','HUM','HBAN','HII','IBM','IEX','IDXX','INFO','ITW','ILMN','INCY','IR','INTC','ICE','IFF','IP','IPG','INTU','ISRG','IVZ','IPGP','IQV','IRM','JBHT','JKHY','J','SJM','JNJ','JCI','JPM','JNPR','KSU','K','KEY','KEYS','KMB','KIM','KMI','KLAC','KHC','KR','LHX','LH','LRCX','LW','LVS','LEG','LDOS','LEN','LNC','LIN','LYV','LKQ','LMT','L','LOW','LUMN','LYB','MTB','MRO','MPC','MKTX','MAR','MMC','MLM','MAS','MA','MTCH','MKC','MCD','MCK','MDT','MRK','MET','MTD','MGM','MCHP','MU','MSFT','MAA','MRNA','MHK','TAP','MDLZ','MPWR','MNST','MCO','MS','MSI','MSCI','NDAQ','NTAP','NFLX','NWL','NEM','NWSA','NWS','NEE','NLSN','NKE','NI','NSC','NTRS','NOC','NLOK','NCLH','NRG','NUE','NVDA','NVR','NXPI','ORLY','OXY','ODFL','OMC','OKE','ORCL','OGN','OTIS','PCAR','PKG','PH','PAYX','PAYC','PYPL','PENN','PNR','PBCT','PEP','PKI','PFE','PM','PSX','PNW','PXD','PNC','POOL','PPG','PPL','PFG','PG','PGR','PLD','PRU','PTC','PEG','PSA','PHM','PVH','QRVO','QCOM','PWR','DGX','RL','RJF','RTX','O','REG','REGN','RF','RSG','RMD','RHI','ROK','ROL','ROP','ROST','RCL','SPGI','CRM','SBAC','SLB','STX','SEE','SRE','NOW','SHW','SPG','SWKS','SNA','SO','LUV','SWK','SBUX','STT','STE','SYK','SIVB','SYF','SNPS','SYY','TMUS','TROW','TTWO','TPR','TGT','TEL','TDY','TFX','TER','TSLA','TXN','TXT','COO','HIG','HSY','MOS','TRV','DIS','TMO','TJX','TSCO','TT','TDG','TRMB','TFC','TWTR','TYL','TSN','USB','UDR','ULTA','UAA','UA','UNP','UAL','UPS','URI','UNH','UHS','VLO','VTR','VRSN','VRSK','VZ','VRTX','VFC','VIAC','VTRS','V','VNO','VMC','WRB','GWW','WAB','WBA','WMT','WM','WAT','WEC','WFC','WELL','WST','WDC','WU','WRK','WY','WHR','WMB','WLTW','WYNN','XEL','XLNX','XYL','YUM','ZBRA','ZBH','ZION','ZTS'] + ['SPY','DJI'])
TotalTestList = random.sample(TotalStockList, 100)  # Randomly select 50 elements for testing
TotalTrainList = [x for x in TotalStockList if x not in TotalTestList]

TotalTradeList = TotalStockList
TotalTradeList = random.SystemRandom().sample(TotalStockList, 30)

TotalSectorList = ['Materials', 'Health Care', 'Information Technology', 'Consumer Discretionary', 'Financials', 'Communication Services', 'Energy', 'Real Estate', 'Consumer Staples', 'Utilities', 'Industrials']

#Numerical_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI', 'OBV', 'MACD', 'MACD_Signal', '%K', 'MA', 'Signal_Variance','Upper_Band','Middle_Band','Lower_Band','EMA','ATR']
#Numerical_cols = ['Open','High','Low','Close','Growth','Adj Close','Volume','M1_Money_Supply','M2_Money_Supply','CPI','PPI','RSI','OBV','MACD','MACD_Signal','%K','MA','Signal_Variance','Upper_Band','Middle_Band','Lower_Band','EMA','ATR']
Numerical_cols = ['Open','High','Low','Close','Adj Close','Volume','Growth','M1_Money_Supply','M2_Money_Supply','CPI','PPI','RSI','OBV','MACD','MACD_Signal','%K','MA','Signal_Variance','SMA','WMA','ROC','Bullish_Engulfing','Bearish_Engulfing','Upper_Band','Middle_Band','Lower_Band','EMA','ATR','TimeStep','YearPos']
# Make sure numerical cols is right or everything gets fucked up

num_cols = len(Numerical_cols)

#For retrieving numerical data, BE SURE THAT THE FILTERED TIME IS INCLUDED IN THIS
start_date = datetime.datetime(2018, 5, 13)  
end_date = datetime.datetime(2023, 7, 12)


#For retrieving news article data
news_start_date = datetime.datetime(2018, 6, 13)  
news_end_date = datetime.datetime(2023, 6, 28)


#For cutting our training data so that news articles fit with numerical data
filtered_start_date = news_start_date
filtered_end_date = news_end_date

window_size = 31
articles_per_day = 4
max_BERT_sequence_length = 64
max_num_cpu_threads = multiprocessing.cpu_count()

BERT_model_name = 'huawei-noah/TinyBERT_General_4L_312D'


# Your custom layer definition
class MultiTimeDistributed(tf.keras.layers.Layer):
    def __init__(self, layer, **kwargs):
        self.layer = layer
        super(MultiTimeDistributed, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]) + self.layer.compute_output_shape(input_shape[2:])

    def call(self, inputs):
        input_shape = inputs.shape
        timesteps = input_shape[1]
        articles = input_shape[2]

        inputs = tf.reshape(inputs, (-1,) + tuple(map(int, input_shape[3:])))
        outputs = self.layer(inputs)
        pooled_output = outputs[1]  # extract the pooled output

        # Reshape to (-1, timesteps, articles) + output_shape
        output_shape = pooled_output.shape
        outputs = tf.reshape(pooled_output, (-1, timesteps, articles) + tuple(map(int, output_shape[1:])))

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"layer": tf.keras.layers.serialize(self.layer)})
        return config

    @classmethod
    def from_config(cls, config):
        layer_config = config.pop("layer")  # Get and remove the layer config from the config
        layer = tf.keras.layers.deserialize(layer_config, custom_objects={'TFBertModel': TFBertModel})
        return cls(layer=layer, **config)
    

def gaussian_nll(y_true, y_pred):
    """Gaussian negative log likelihood

    Note that y_pred should have shape (batch_size, 2), where y_pred[:, 0]
    corresponds to the mean of the distribution (mu) and y_pred[:, 1] corresponds
    to the log of the standard deviation (log_sigma).

    Parameters
    ----------
    y_true : tf.Tensor
        Actual values (labels)
    y_pred : tf.Tensor
        Predicted values, must be of shape (batch_size, 2)

    Returns
    -------
    nll : tf.Tensor
        The negative log-likelihood
    """
    # Split the input tensor into mean and variance

    mu = y_pred[:, 0]
    log_sigma = tf.clip_by_value(y_pred[:, 1], -10, 10)
    sigma = tf.exp(log_sigma) + 1e-6  # prevent sigma from being too close to zero
    mse = tf.math.squared_difference(y_true, mu)
    tf.print("mu:", mu, "sigma:", sigma, "mse:", mse)  # print values during training
    nll = 0.5 * tf.math.log(2 * np.pi * tf.math.square(sigma)) + mse / (2 * tf.math.square(sigma))
    return tf.reduce_mean(nll)


class ScalingLayer(Layer):
    def __init__(self, scale, **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return self.scale * inputs

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


def custom_loss(y_true, y_pred):
    # Calculate the difference between the true and predicted values
    diff = y_true - y_pred
    
    # Determine if the prediction is in the wrong direction
    wrong_direction_penalty = tf.where(tf.sign(y_true) != tf.sign(y_pred), 0.8, 1.0)
    wrong_direction_penalty = tf.cast(wrong_direction_penalty, y_true.dtype)


    # Determine if the prediction is an undershoot
    undershoot_penalty = tf.where(tf.abs(y_true) > tf.abs(y_pred), 1.5, 1.0)
    undershoot_penalty = tf.cast(undershoot_penalty, y_true.dtype)
    
    return tf.reduce_mean(tf.square(diff)*undershoot_penalty*wrong_direction_penalty)


def custom_binary_loss(y_true, y_pred):
    # Map -1 to 0 and 1 to 1
    y_true = (y_true + 1) / 2

    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


    '''
    TODO FOR EVERYTHING:

    Phase 1:
    - Make label 0 to 1 instead of -1 to 1
    - Split training and validation data sample-wise rather than by stock?
    - Find optimal prediction window period (based on profit)
    - change "Growth" to be based on open price -> close price in a given day, rather than an overnight growth% (maybe not actually, as it affects our label)
    - Add SPY growth as input

    Phase 2:
    - pre-tokenize articles
    - Re-integrate news data
    
    Phase 3:
    - Optimal threshold adjustment (best threshold last 2 weeks?)
    

        
    Far Future:
    - Interactive Brokers
    - Switch to Pytorch
    - Find much smaller memory footprint LLMs (check this guy's channel https://www.youtube.com/@vrsen/videos)
    '''

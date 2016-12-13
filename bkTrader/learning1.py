import datetime

import backtrader as bt
from dateutil import parser as date_parser

class ColorMACD(bt.indicators.MACD):

    lines = ('histo_p', 'histo_n')


    plotlines = dict(histo_p=dict(_method='bar', alpha=0.50, width=1.0, color='r'),
                     histo_n=dict(_method='bar', alpha=0.50, width=1.0, color='g'))

    def __init__(self):
        super(ColorMACD, self).__init__()
        self.lines.histo_p = bt.If(self.lines.macd - self.lines.signal >= 0, self.lines.macd - self.lines.signal, 0)
        self.lines.histo_n = bt.If(self.lines.macd - self.lines.signal <= 0, self.lines.macd - self.lines.signal, 0)



class TestStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        '''Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print '%s, %s' % (dt.isoformat(), txt)

    def __init__(self, exitbars=None, maperiod=None):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.params.exitbars = exitbars
        self.params.maperiod = maperiod
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=self.params.maperiod)

        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25, plot=False)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25, plot=False)
        bt.indicators.StochasticSlow(self.datas[0], plot=False)
        rsi = bt.indicators.RSI(self.datas[0], plot=False)
        bt.indicators.SmoothedMovingAverage(rsi, period=20, plot=False)
        bt.indicators.ATR(self.datas[0], plot=False)
        ColorMACD(self.datas[0], plotname='MACD', subplot=True)

    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])

        if self.order:
            return

        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                self.order = self.buy()
                self.log('Buy order created at %.2f' % self.dataclose[0])
        else:
            if self.dataclose[0] < self.sma[0]:
                self.log('Sell order created at : %.2f' % self.dataclose[0])
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log('Buy order executed, price: %.2f, cost: %.2f, comm: %.2f' % (order.executed.price,
                                                                                      order.executed.value,
                                                                                      order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm

            elif order.issell():
                self.log('Sell order executed, price: %.2f, cost: %.2f, comm: %.2f' % (order.executed.price,
                                                                                       order.executed.value,
                                                                                       order.executed.comm))

            self.bar_executed = len(self)

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION profit: Gross: %.2f, Net: %.2f' % (trade.pnl, trade.pnlcomm))


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    data = bt.feeds.YahooFinanceData(dataname='600019.SS', fromdate=datetime.datetime(2000, 1, 1),
                                     todate=date_parser.parse('2002-12-31'), reverse=True)

    cerebro.adddata(data)
    cerebro.addsizer(bt.sizers.FixedSize, stake=5)
    cerebro.addstrategy(TestStrategy, maperiod=15)
    cerebro.broker.set_cash(100)
    cerebro.broker.setcommission(commission=0.001)

    print 'Starting Portfolio value: %.2f' % cerebro.broker.getvalue()

    cerebro.run()

    print 'Final Portfolio value: %.2f' % cerebro.broker.getvalue()
    cerebro.plot()

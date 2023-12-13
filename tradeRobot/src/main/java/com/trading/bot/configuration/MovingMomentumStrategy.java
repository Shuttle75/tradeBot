package com.trading.bot.configuration;

import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseStrategy;
import org.ta4j.core.Rule;
import org.ta4j.core.Strategy;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.MACDIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.volume.ChaikinMoneyFlowIndicator;
import org.ta4j.core.rules.*;
import software.amazon.ion.Decimal;

public class MovingMomentumStrategy {

    /**
     * @param series a time series
     * @return a moving momentum strategy
     */
    public static Strategy buildStrategy(BarSeries series) {
        if (series == null) {
            throw new IllegalArgumentException("Series cannot be null");
        }

        ClosePriceIndicator closePrice = new ClosePriceIndicator(series);

        // The bias is bullish when the shorter-moving average moves above the longer moving average.
        // The bias is bearish when the shorter-moving average moves below the longer moving average.
        EMAIndicator shortEma = new EMAIndicator(closePrice, 50);
        EMAIndicator longEma = new EMAIndicator(closePrice, 200);

        MACDIndicator macd = new MACDIndicator(closePrice);
        EMAIndicator signal = new EMAIndicator(macd, 9);

        ChaikinMoneyFlowIndicator moneyFlowIndicator = new ChaikinMoneyFlowIndicator(series, 20);

        // Entry rule
        Rule entryRule = new OverIndicatorRule(shortEma, longEma) // Trend
            .and(new CrossedDownIndicatorRule(signal, macd)) // Signal 1
            .and(new OverIndicatorRule(moneyFlowIndicator, Decimal.ZERO)); // Signal 2

        // Exit rule
        Rule exitRule = new CrossedUpIndicatorRule(signal, macd); // Signal 1

        return new BaseStrategy(entryRule, exitRule);
    }
}

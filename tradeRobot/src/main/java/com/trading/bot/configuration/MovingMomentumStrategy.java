package com.trading.bot.configuration;

import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseStrategy;
import org.ta4j.core.Rule;
import org.ta4j.core.Strategy;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.MACDIndicator;
import org.ta4j.core.indicators.SMAIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.helpers.CombineIndicator;
import org.ta4j.core.num.Num;
import org.ta4j.core.rules.*;


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
        SMAIndicator smaFast = new SMAIndicator(closePrice, 5);

        MACDIndicator macd = new MACDIndicator(smaFast, 48, 104);
        EMAIndicator signal = new EMAIndicator(macd, 36);

        CombineIndicator histogram = new CombineIndicator(macd, signal, Num::minus);

        // Entry rule
        Rule entryRule = new UnderIndicatorRule(macd, signal)
                .and(new IsRisingRule(histogram, 2));

        // Exit rule
        Rule exitRule = new OverIndicatorRule(macd, signal)
                .and(new IsFallingRule(histogram, 8));

        return new BaseStrategy(entryRule, exitRule);
    }
}

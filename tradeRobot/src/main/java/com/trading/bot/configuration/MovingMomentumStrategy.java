package com.trading.bot.configuration;

import org.ta4j.core.BarSeries;
import org.ta4j.core.BaseStrategy;
import org.ta4j.core.Rule;
import org.ta4j.core.Strategy;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.MACDIndicator;
import org.ta4j.core.indicators.StochasticOscillatorKIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.rules.CrossedDownIndicatorRule;
import org.ta4j.core.rules.CrossedUpIndicatorRule;
import org.ta4j.core.rules.OverIndicatorRule;
import org.ta4j.core.rules.UnderIndicatorRule;
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

        StochasticOscillatorKIndicator stochasticOscillK = new StochasticOscillatorKIndicator(series, 14);

        MACDIndicator macd = new MACDIndicator(closePrice);
        EMAIndicator emaMacd = new EMAIndicator(macd, 9);

        // Entry rule
        Rule entryRule = new OverIndicatorRule(shortEma, longEma) // Trend
            .and(new CrossedDownIndicatorRule(stochasticOscillK, Decimal.valueOf(20))); // Signal 1
//            .and(new OverIndicatorRule(macd, emaMacd)); // Signal 2

        // Exit rule
        Rule exitRule = new UnderIndicatorRule(shortEma, longEma) // Trend
            .and(new CrossedUpIndicatorRule(stochasticOscillK, Decimal.valueOf(80))); // Signal 1
//            .and(new UnderIndicatorRule(macd, emaMacd)); // Signal 2

        return new BaseStrategy(entryRule, exitRule);
    }
}

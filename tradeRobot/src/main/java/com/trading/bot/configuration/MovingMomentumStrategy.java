package com.trading.bot.configuration;

import org.ta4j.core.*;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.MACDIndicator;
import org.ta4j.core.indicators.SMAIndicator;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.helpers.CombineIndicator;
import org.ta4j.core.num.Num;
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
        EMAIndicator shortEma = new EMAIndicator(closePrice, 9);

        MACDIndicator macd = new MACDIndicator(shortEma, 12, 26);
        EMAIndicator signal = new EMAIndicator(macd, 9);

        CombineIndicator histogram = new CombineIndicator(macd, signal, Num::minus);

        // Entry rule
        Rule entryRule = new UnderIndicatorRule(macd, signal)
                .and(new IsRisingRule(histogram, 1))
                .and(new OverIndicatorRule(macd, Decimal.ZERO));

        // Exit rule
        Rule exitRule = new OverIndicatorRule(macd, signal)
                .and(new IsFallingRule(histogram, 2));

        return new BaseStrategy(entryRule, exitRule);
    }
}

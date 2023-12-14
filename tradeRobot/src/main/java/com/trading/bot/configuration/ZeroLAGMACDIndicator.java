package com.trading.bot.configuration;

import org.ta4j.core.Indicator;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.RecursiveCachedIndicator;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.num.Num;

public class ZeroLAGMACDIndicator extends RecursiveCachedIndicator<Num> {

    private final EMAIndicator shortTermEma;
    private final EMAIndicator shortTermEmaEma;
    private final EMAIndicator longTermEma;
    private final EMAIndicator longTermEmaEma;

    /**
     * Constructor with shortBarCount "12" and longBarCount "26".
     *
     * @param indicator the indicator
     */
    public ZeroLAGMACDIndicator(Indicator<Num> indicator) {
        this(indicator, 12, 26);
    }

    /**
     * Constructor.
     *
     * @param indicator     the indicator
     * @param shortBarCount the short time frame (normally 12)
     * @param longBarCount  the long time frame (normally 26)
     */
    public ZeroLAGMACDIndicator(Indicator<Num> indicator, int shortBarCount, int longBarCount) {
        super(indicator);
        if (shortBarCount > longBarCount) {
            throw new IllegalArgumentException("Long term period count must be greater than short term period count");
        }
        shortTermEma = new EMAIndicator(indicator, shortBarCount);
        shortTermEmaEma = new EMAIndicator(shortTermEma, shortBarCount);
        longTermEma = new EMAIndicator(indicator, longBarCount);
        longTermEmaEma = new EMAIndicator(longTermEma, longBarCount);
    }

    @Override
    protected Num calculate(int index) {
        Num shortPart = DecimalNum.valueOf(2)
                .multipliedBy(shortTermEma.getValue(index))
                .minus(shortTermEmaEma.getValue(index));
        Num longPart = DecimalNum.valueOf(2)
                .multipliedBy(longTermEma.getValue(index))
                .minus(longTermEmaEma.getValue(index));

        return shortPart.minus(longPart);
    }
}

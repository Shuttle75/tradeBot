package com.trading.bot.configuration;

import org.ta4j.core.Indicator;
import org.ta4j.core.indicators.EMAIndicator;
import org.ta4j.core.indicators.RecursiveCachedIndicator;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.num.Num;

public class ZeroLAGSignalIndicator extends RecursiveCachedIndicator<Num> {

    private final int barCount;
    private final EMAIndicator ema;
    private final EMAIndicator emaEma;

    /**
     * Constructor.
     *
     * @param indicator the indicator
     * @param barCount  the time frame
     */
    public ZeroLAGSignalIndicator(Indicator<Num> indicator, int barCount) {
        super(indicator);
        this.barCount = barCount;
        this.ema = new EMAIndicator(indicator, barCount);
        this.emaEma = new EMAIndicator(ema, barCount);
    }

    @Override
    protected Num calculate(int index) {
        return DecimalNum.valueOf(2)
                .multipliedBy(ema.getValue(index))
                .minus(emaEma.getValue(index));
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + " barCount: " + barCount;
    }
}
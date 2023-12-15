package com.trading.bot.configuration;

import org.ta4j.core.*;
import org.ta4j.core.Bar;
import org.ta4j.core.rules.AbstractRule;

/**
 * A {@link org.ta4j.core.Rule} which waits for a number of {@link Bar} after a
 * trade.
 *
 * Satisfied after a fixed number of bars since the last trade.
 */
public class WaitAfterGoodTradeRule extends AbstractRule {

    /**
     * The number of bars to wait for
     */
    private final int numberOfBars;

    /**
     * Constructor.
     *
     * @param numberOfBars the number of bars to wait for
     */
    public WaitAfterGoodTradeRule(int numberOfBars) {
        this.numberOfBars = numberOfBars;
    }

    /** This rule uses the {@code tradingRecord}. */
    @Override
    public boolean isSatisfied(int index, TradingRecord tradingRecord) {
        boolean satisfied = false;
        // No trading history, no need to wait
        if (tradingRecord != null) {
            Position position = tradingRecord.getLastPosition();
            if (position != null) {
                if (position.hasProfit()) {
                    int currentNumberOfBars = index - position.getExit().getIndex();
                    satisfied = currentNumberOfBars >= numberOfBars;
                } else {
                    satisfied = true;
                }
            }
        }
        traceIsSatisfied(index, satisfied);
        return satisfied;
    }
}

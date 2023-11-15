package com.trading.bot.util;

import org.knowm.xchange.Exchange;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;

import javax.ws.rs.NotSupportedException;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collector;

import static com.trading.bot.configuration.BotConfig.*;
import static java.lang.Math.round;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min1;

public class TradeUtil {

    private TradeUtil() {
    }

    public static Ticker getTicker(Exchange exchange) throws IOException {
        return ((KucoinMarketDataService) exchange.getMarketDataService())
                .getTicker(CURRENCY_PAIR);
    }

    public static List<KucoinKline> getKucoinKlines(Exchange exchange, long startDate, long endDate) throws IOException {
        return ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(CURRENCY_PAIR, startDate, endDate, min1);
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int pos) {
        BigDecimal data = kucoinKlines.get(pos + PREDICT_DEEP).getClose()
                .subtract(kucoinKlines.get(pos).getClose());

        int delta = round((data.floatValue() + CURRENCY_DELTA * 3.5F) / CURRENCY_DELTA);

        delta = Math.max(delta, 0);
        delta = Math.min(delta, OUTPUT_SIZE - 1);
        return delta;
    }
}

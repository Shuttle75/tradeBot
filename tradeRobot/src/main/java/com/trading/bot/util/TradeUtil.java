package com.trading.bot.util;

import org.knowm.xchange.Exchange;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;

import java.io.IOException;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;

import static com.trading.bot.configuration.BotConfig.*;
import static java.lang.Math.round;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min1;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min30;

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

    public static boolean isGoodTrend(Exchange exchange) throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
                .minusMinutes(30)
                .toEpochSecond(ZoneOffset.UTC);

        KucoinKline kucoinKline = ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(CURRENCY_PAIR, startDate, 0L, min30)
                .get(0);

        return kucoinKline.getOpen().compareTo(kucoinKline.getClose()) < 0;
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int pos) {
        BigDecimal data = kucoinKlines.get(pos + PREDICT_DEEP).getClose()
                .subtract(kucoinKlines.get(pos).getClose());

        int delta = round((data.floatValue() + CURRENCY_DELTA * 2) / CURRENCY_DELTA);

        delta = Math.max(delta, 0);
        delta = Math.min(delta, OUTPUT_SIZE - 1);
        return delta;
    }

    public static String printRates(float[] floatResult) {
        return String.format("%.2f", floatResult[0]) + " " +
                String.format("%.2f", floatResult[1]) + "  " +
                String.format("%.2f", floatResult[2]) + "  " +
                String.format("%.2f", floatResult[3]) + " " +
                String.format("%.2f", floatResult[4]);
    }
}

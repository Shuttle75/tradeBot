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

    public static float calcData(List<KucoinKline> kucoinKlines, int i, int y, int predict) {
        return kucoinKlines.get(i + y + predict).getClose()
                .subtract(kucoinKlines.get(i + y + predict).getOpen())
                .multiply(kucoinKlines.get(i + y + predict).getVolume())
                .floatValue();
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int i) {
        BigDecimal data0 = kucoinKlines.get(i).getClose()
                .subtract(kucoinKlines.get(i).getOpen())
                .multiply(kucoinKlines.get(i).getVolume());

        BigDecimal data1 = kucoinKlines.get(i + 1).getClose()
                .subtract(kucoinKlines.get(i + 1).getOpen())
                .multiply(kucoinKlines.get(i + 1).getVolume());

        int delta = (data0.add(data1).intValue() + (OUTPUT_SIZE * CURRENCY_DELTA) / 2) / CURRENCY_DELTA;

        delta = Math.max(delta, 0);
        delta = Math.min(delta, 7);
        return delta;
    }

    public static String printRates(float[] floatResult) {
        return String.format("%.2f", floatResult[0]) + " " +
                String.format("%.2f", floatResult[1]) + " " +
                String.format("%.2f", floatResult[2]) + " " +
                String.format("%.2f", floatResult[3]) + " | " +
                String.format("%.2f", floatResult[4]) + " " +
                String.format("%.2f", floatResult[5]) + " " +
                String.format("%.2f", floatResult[6]) + " " +
                String.format("%.2f", floatResult[7]);
    }
}

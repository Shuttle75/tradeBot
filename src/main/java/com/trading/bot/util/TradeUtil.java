package com.trading.bot.util;

import org.knowm.xchange.Exchange;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;

import java.io.IOException;
import java.math.BigDecimal;
import java.util.List;

import static com.trading.bot.configuration.BotConfig.*;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min1;

public class TradeUtil {

    private TradeUtil() {
    }

    public static List<KucoinKline> getKucoinKlines(Exchange exchange, long startDate, long endDate) throws IOException {
        return ((KucoinMarketDataService) exchange.getMarketDataService())
                .getKucoinKlines(CURRENCY_PAIR, startDate, endDate, min1)
                .stream()
                .collect(new KucoinKlineCollector());
    }

    public static float calcData(List<KucoinKline> kucoinKlines, int i, int y, int predict) {
        return kucoinKlines.get(i + y + predict).getClose()
                .subtract(kucoinKlines.get(i + y + predict).getOpen())
                .multiply(kucoinKlines.get(i + y + predict).getVolume())
                .floatValue() * TRAIN_DEEP / (TRAIN_DEEP + y * 2);
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int i) {
        BigDecimal data0 = kucoinKlines.get(i).getClose().subtract(kucoinKlines.get(i).getOpen());
        BigDecimal data1 = kucoinKlines.get(i + 1).getClose().subtract(kucoinKlines.get(i + 1).getOpen());
        BigDecimal data2 = kucoinKlines.get(i + 2).getOpen().subtract(kucoinKlines.get(i + 2).getClose());
        BigDecimal data3 = kucoinKlines.get(i + 3).getOpen().subtract(kucoinKlines.get(i + 3).getClose());
        int delta = (data0.add(data1).add(data2).add(data3).intValue() + OUTPUT_SIZE * CURRENCY_DELTA / 2) / CURRENCY_DELTA;

        delta = Math.max(delta, 0);
        delta = Math.min(delta, 7);
        return delta;
    }

    public static String printRates(float[][] floatResult) {
        return String.format("%.2f", floatResult[0][0]) + " " +
                String.format("%.2f", floatResult[0][1]) + " " +
                String.format("%.2f", floatResult[0][2]) + " " +
                String.format("%.2f", floatResult[0][3]) + " | " +
                String.format("%.2f", floatResult[0][4]) + " " +
                String.format("%.2f", floatResult[0][5]) + " " +
                String.format("%.2f", floatResult[0][6]) + " " +
                String.format("%.2f", floatResult[0][7]);
    }
}

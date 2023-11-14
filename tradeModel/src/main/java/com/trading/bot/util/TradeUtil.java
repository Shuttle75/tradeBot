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

    public static int getDelta(List<KucoinKline> kucoinKlines, int i, int y) {
        BigDecimal data0 = kucoinKlines.get(i + y).getClose()
                .subtract(kucoinKlines.get(i + y).getOpen())
                .multiply(kucoinKlines.get(i + y).getVolume());

        BigDecimal data1 = kucoinKlines.get(i + y + 1).getClose()
                .subtract(kucoinKlines.get(i + y + 1).getOpen())
                .multiply(kucoinKlines.get(i + y + 1).getVolume());

        int delta = (data0.add(data1).intValue() + (OUTPUT_SIZE * CURRENCY_DELTA) / 2) / CURRENCY_DELTA;

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

    public static Collector<KucoinKline, List<KucoinKline>, List<KucoinKline>> reduceKucoinKlines() {
        return new KucoinKlineCollector();
    }

    static class KucoinKlineCollector implements Collector<KucoinKline, List<KucoinKline>, List<KucoinKline>> {
        @Override
        public Supplier<List<KucoinKline>> supplier() {
            return ArrayList::new;
        }

        @Override
        public BiConsumer<List<KucoinKline>, KucoinKline> accumulator() {
            return (kucoinKlines, newKucoinKline) -> {
                if (kucoinKlines.isEmpty()) {
                    kucoinKlines.add(newKucoinKline);
                    return;
                }

                KucoinKline prevKucoinKline = kucoinKlines.get(kucoinKlines.size() - 1);
                if (newKucoinKline.getVolume().floatValue() < 0.3) {
                    kucoinKlines.set(
                            kucoinKlines.size() - 1,
                            new KucoinKline(
                                    prevKucoinKline.getPair(),
                                    prevKucoinKline.getIntervalType(),
                                    new Object[]{
                                            (prevKucoinKline.getTime() + newKucoinKline.getTime()) / 2L,
                                            prevKucoinKline.getOpen().floatValue(),
                                            newKucoinKline.getClose().floatValue(),
                                            prevKucoinKline.getHigh().compareTo(newKucoinKline.getHigh()) > 0 ?
                                                    prevKucoinKline.getHigh().floatValue() :
                                                    newKucoinKline.getHigh().floatValue(),
                                            prevKucoinKline.getLow().compareTo(newKucoinKline.getHigh()) < 0 ?
                                                    prevKucoinKline.getLow().floatValue() :
                                                    newKucoinKline.getLow().floatValue(),
                                            prevKucoinKline.getVolume().floatValue() + newKucoinKline.getVolume().floatValue(),
                                            prevKucoinKline.getAmount().floatValue() + newKucoinKline.getAmount().floatValue()}
                            )
                    );
                } else {
                    kucoinKlines.add(newKucoinKline);
                }
            };
        }

        @Override
        public BinaryOperator<List<KucoinKline>> combiner() {
            return (kucoinKlines1, kucoinKlines2) -> {
                throw new NotSupportedException();
            };
        }

        @Override
        public Function<List<KucoinKline>, List<KucoinKline>> finisher() {
            return kucoinKlines -> kucoinKlines;
        }

        @Override
        public Set<Characteristics> characteristics() {
            return Collections.unmodifiableSet(EnumSet.of(Collector.Characteristics.IDENTITY_FINISH));
        }
    }
}

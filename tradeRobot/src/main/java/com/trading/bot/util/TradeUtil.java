package com.trading.bot.util;

import org.knowm.xchange.Exchange;
import org.knowm.xchange.dto.marketdata.Ticker;
import org.knowm.xchange.kucoin.KucoinMarketDataService;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;

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
                .getKucoinKlines(CURRENCY_PAIR, startDate, endDate, min1)
                .stream()
                .collect(reduceKucoinKlines());
    }

    public static int getDelta(List<KucoinKline> kucoinKlines, int pos) {
        BigDecimal data = kucoinKlines.get(pos + PREDICT_DEEP).getClose()
                .subtract(kucoinKlines.get(pos).getClose());

        int delta = round((data.floatValue() + CURRENCY_DELTA * 2) / CURRENCY_DELTA);

        delta = Math.max(delta, 0);
        delta = Math.min(delta, OUTPUT_SIZE - 1);
        return delta;
    }

    public static void calcData(List<KucoinKline> kucoinKlines, int i, int y, INDArray indData, INDArray indLabels) {
        indData.putScalar(new int[]{i, 0, y},
                kucoinKlines.get(y).getClose()
                        .subtract(kucoinKlines.get(y).getOpen()).movePointLeft(NORMAL).floatValue());
        indData.putScalar(new int[]{i, 1, y},
                kucoinKlines.get(y).getClose()
                        .compareTo(kucoinKlines.get(y).getOpen()) > 0 ?
                        kucoinKlines.get(y).getHigh()
                                .subtract(kucoinKlines.get(y).getClose()).movePointLeft(NORMAL).floatValue() :
                        kucoinKlines.get(y).getHigh()
                                .subtract(kucoinKlines.get(y).getOpen()).movePointLeft(NORMAL).floatValue());
        indData.putScalar(new int[]{i, 2, y},
                kucoinKlines.get(y).getClose()
                        .compareTo(kucoinKlines.get(y).getOpen()) > 0 ?
                        kucoinKlines.get(y).getOpen()
                                .subtract(kucoinKlines.get(y).getLow()).movePointLeft(NORMAL).floatValue() :
                        kucoinKlines.get(y).getClose()
                                .subtract(kucoinKlines.get(y).getLow()).movePointLeft(NORMAL).floatValue());
        indData.putScalar(new int[]{i, 1, y},
                kucoinKlines.get(y).getVolume().movePointLeft(NORMAL).floatValue());
        indLabels.putScalar(new int[]{i, getDelta(kucoinKlines, y), y}, 1);
    }

    public static String printRates(float[] floatResult) {
        return String.format("%.2f", floatResult[0]) + " " +
                String.format("%.2f", floatResult[1]) + "  " +
                String.format("%.2f", floatResult[2]) + "  " +
                String.format("%.2f", floatResult[3]) + " " +
                String.format("%.2f", floatResult[4]);
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

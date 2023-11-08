package com.trading.bot.util;

import org.knowm.xchange.kucoin.dto.response.KucoinKline;

import javax.ws.rs.NotSupportedException;
import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collector;

public class KucoinKlineCollector implements Collector<KucoinKline, List<KucoinKline>, List<KucoinKline>> {
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

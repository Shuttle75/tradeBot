package com.trading.bot.configuration;

import com.amazonaws.AmazonServiceException;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.bybit.api.client.domain.CategoryType;
import com.bybit.api.client.domain.market.IntervalTime;
import com.bybit.api.client.domain.market.MarketInterval;
import com.bybit.api.client.domain.market.request.MarketDataRequest;
import com.bybit.api.client.domain.market.response.kline.MarketKlineEntry;
import com.bybit.api.client.domain.market.response.kline.MarketKlineResult;
import com.bybit.api.client.domain.market.response.openInterests.OpenInterestEntry;
import com.bybit.api.client.domain.market.response.openInterests.OpenInterestResult;
import com.bybit.api.client.restApi.BybitApiAsyncMarketDataRestClient;
import com.bybit.api.client.restApi.BybitApiMarketRestClient;
import com.bybit.api.client.service.BybitApiClientFactory;;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static com.trading.bot.util.TradeUtil.calcData;
import static com.trading.bot.util.TradeUtil.getDelta;
import static java.util.Objects.nonNull;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    public static final int INPUT_SIZE = 6;
    public static final int LAYER_SIZE = 24;
    public static final int OUTPUT_SIZE = 3;
    public static final int TRAIN_EXAMPLES = 8;
    public static final int TRAIN_KLINES = 672;
    public static final int PREDICT_DEEP = 8;
    public static final float PRICE_PERCENT = 1.6F;
    final static String API_KEY = "tAOkQmuG2EUz1stdlUDpDyJRXp8oAY59W1A9Hv4HeR349jfPJbl6Hxuz4gHWKN4I";
    final static String API_SECRET = "QzJmDQtT02xhxLxrN5irmCtPWxQgJjyMIOQb5AqXXWPQRb9jQQtjrGy36tpkmvJF";
    public static final String SYMBOL = "SOLUSDT";
    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Value("${model.bucket:trade-crypto-models}")
    public String bucketName;

    @Bean
    public MultiLayerConfiguration getConfig() {
        logger.info("Build model....");
        return new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .list()
                .layer(new LSTM.Builder().activation(Activation.TANH).nIn(INPUT_SIZE).nOut(LAYER_SIZE).build())
                .layer(new LSTM.Builder().activation(Activation.TANH).nOut(LAYER_SIZE).dropOut(0.9).build())
                .layer(new LSTM.Builder().activation(Activation.TANH).nOut(LAYER_SIZE).dropOut(0.9).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nOut(OUTPUT_SIZE).build())
                .build();
    }

    @Bean
    public StatsStorage getStatsStorage() {
        return new InMemoryStatsStorage();
    }

    @Bean
    public UIServer getUiServer(StatsStorage statsStorage) {
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);
        return uiServer;
    }

    @Bean
    public MultiLayerNetwork getModel(MultiLayerConfiguration config, StatsStorage statsStorage)
            throws IOException {
        final String keyName = SYMBOL + ".zip";
        final String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), keyName);
        final LocalDateTime now = LocalDateTime.of(2023, 12,1, 0, 0);
        final AmazonS3 s3 = AmazonS3ClientBuilder.standard().withRegion(Regions.EU_CENTRAL_1).build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new StatsListener(statsStorage));

        BybitApiMarketRestClient client = BybitApiClientFactory.newInstance().newMarketDataRestClient();

        MarketDataRequest marketDataRequest = MarketDataRequest.builder()
            .category(CategoryType.LINEAR)
            .symbol(SYMBOL)
            .marketInterval(MarketInterval.FIFTEEN_MINUTES)
            .marketIntervalTime(IntervalTime.FIFTEEN_MINUTES)
            .build();
        MarketKlineResult marketKlineResult = new MarketKlineResult();
        OpenInterestResult openInterestResult = new OpenInterestResult();
        Object response = new Object();

        try (INDArray indData = Nd4j.zeros(TRAIN_EXAMPLES, INPUT_SIZE, TRAIN_KLINES);
             INDArray indLabels = Nd4j.zeros(TRAIN_EXAMPLES, OUTPUT_SIZE, TRAIN_KLINES)) {

            int i = TRAIN_EXAMPLES - 1;
            while (i >= 0) {
                LocalDateTime startDate = now.minusSeconds(i * (long) TRAIN_KLINES * 15 * 60 + (TRAIN_KLINES + PREDICT_DEEP) * 15 * 60);

                List<MarketKlineEntry> klines = new ArrayList<>();
                marketDataRequest.setStart(startDate.toEpochSecond(ZoneOffset.UTC) * 1000);
                marketDataRequest.setLimit(PREDICT_DEEP + 96 * 7 + PREDICT_DEEP);
                response = client.getMarketLinesData(marketDataRequest);
                if (nonNull(response) && response instanceof Map) {
                    marketKlineResult = MAPPER.convertValue(((Map) response).get("result"), MarketKlineResult.class);
                    klines.addAll(marketKlineResult.getMarketKlineEntries());
                }

                List<OpenInterestEntry> openInterests = new ArrayList<>();
                marketDataRequest.setStartTime(startDate.toEpochSecond(ZoneOffset.UTC) * 1000);
                marketDataRequest.setLimit(100);
                for (int j = 0; j < 7; j++) {
                    response = client.getOpenInterest(marketDataRequest);
                    if (nonNull(response) && response instanceof Map) {
                        openInterestResult = MAPPER.convertValue(((Map) response).get("result"), OpenInterestResult.class);
                        openInterests.addAll(openInterestResult.getOpenInterestEntries());
                        marketDataRequest.setCursor(openInterestResult.getNextPageCursor());
                    }
                }

                Collections.reverse(klines);
                Collections.reverse(openInterests);

                for (int y = 0; y < TRAIN_KLINES; y++) {
                    calcData(indData, i, klines, openInterests, y);
                    indLabels.putScalar(new int[]{i, getDelta(klines, y), y}, 1);
                }

                i--;
            }

            net.fit(indData, indLabels);
            while (net.score() > 1D) {
                net.fit(indData, indLabels);
            }
        }

        try {
            net.save(new File(path));

            s3.putObject(bucketName, keyName, new File(path));
        } catch (AmazonServiceException e) {
            logger.error(e.getErrorMessage());
        }

        return net;
    }
}

package com.trading.bot.configuration;

import com.amazonaws.AmazonServiceException;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
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
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.ExchangeFactory;
import org.knowm.xchange.ExchangeSpecification;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.kucoin.KucoinExchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
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
import java.time.temporal.ChronoUnit;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.util.TradeUtil.*;
import static org.knowm.xchange.kucoin.dto.KlineIntervalType.min15;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    public static final int INPUT_SIZE = 4;
    public static final int LAYER_SIZE = 48;
    public static final int OUTPUT_SIZE = 3;
    public static final int TRAIN_EXAMPLES = 8;
    public static final int TRAIN_KLINES = 672;
    public static final int PREDICT_DEEP = 8;
    public static final float UP_PERCENT = 2F;
    public static final float DOWN_PERCENT = 1F;

    @Value("${model.bucket}")
    public String bucketName;
    @Value("${exchange.username}")
    public String exchangeUserName;
    @Value("${exchange.apikey}")
    public String exchangeApiKey;
    @Value("${exchange.secretkey}")
    public String exchangeSecretKey;
    @Value("${exchange.passphrase}")
    public String exchangePassphrase;
    public static final CurrencyPair CURRENCY_PAIR = new CurrencyPair("SOL", "USDT");

    @Bean
    public Exchange getXChangeExchange() {
        ExchangeSpecification exchangeSpecification = new ExchangeSpecification(KucoinExchange.class);

        exchangeSpecification.setUserName(exchangeUserName);
        exchangeSpecification.setApiKey(exchangeApiKey);
        exchangeSpecification.setSecretKey(exchangeSecretKey);
        exchangeSpecification.setExchangeSpecificParametersItem("passphrase", exchangePassphrase);

        return ExchangeFactory.INSTANCE.createExchange(exchangeSpecification);
    }

    @Bean
    public MultiLayerConfiguration getConfig() {
        logger.info("Build model....");
        return new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .list()
                .layer(new LSTM.Builder().activation(Activation.TANH).nIn(INPUT_SIZE).nOut(LAYER_SIZE).build())
                .layer(new LSTM.Builder().activation(Activation.TANH).nOut(LAYER_SIZE / 4).build())
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
    public MultiLayerNetwork getModel(Exchange exchange, MultiLayerConfiguration config, StatsStorage statsStorage)
            throws IOException {
        final String keyName = CURRENCY_PAIR.base + ".zip";
        final String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), keyName);
        final LocalDateTime now = LocalDateTime.now(ZoneOffset.UTC).minusDays(15).truncatedTo(ChronoUnit.DAYS);
        final AmazonS3 s3 = AmazonS3ClientBuilder.standard().withRegion(Regions.EU_CENTRAL_1).build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();
        net.setListeners(new StatsListener(statsStorage));

        try (INDArray indData = Nd4j.zeros(TRAIN_EXAMPLES, INPUT_SIZE, TRAIN_KLINES);
             INDArray indLabels = Nd4j.zeros(TRAIN_EXAMPLES, OUTPUT_SIZE, TRAIN_KLINES)) {

            int i = TRAIN_EXAMPLES - 1;
            while (i >= 0) {
                LocalDateTime startDate = now.minusSeconds(
                        i * (long) TRAIN_KLINES * min15.getSeconds()
                                + TRAIN_KLINES * min15.getSeconds());
                LocalDateTime endDate = now.minusSeconds(
                        i * (long) TRAIN_KLINES * min15.getSeconds()
                                - PREDICT_DEEP * min15.getSeconds());

                List<KucoinKline> kucoinKlines =
                        getKucoinKlines(
                                exchange,
                                startDate.toEpochSecond(ZoneOffset.UTC),
                                endDate.toEpochSecond(ZoneOffset.UTC));
                Collections.reverse(kucoinKlines);

                logger.info("startDate {} endDate {}", startDate, endDate);

                for (int y = 0; y < TRAIN_KLINES; y++) {
                    calcData(indData, kucoinKlines.get(y), i, y);
                    indLabels.putScalar(new int[]{i, getDelta(kucoinKlines, y), y}, 1);
                }

                i--;
            }

            net.fit(indData, indLabels);
            while (net.score() > 1D) {
                net.fit(indData, indLabels);
            }
        }

        try {
            String fileName = CURRENCY_PAIR.base + ".zip";
            net.save(new File(path));

            s3.putObject(bucketName, fileName, new File(path));
        } catch (AmazonServiceException e) {
            logger.error(e.getErrorMessage());
        }

        return net;
    }
}

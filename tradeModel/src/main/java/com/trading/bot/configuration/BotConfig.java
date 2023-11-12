package com.trading.bot.configuration;

import com.amazonaws.AmazonServiceException;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.ExchangeFactory;
import org.knowm.xchange.ExchangeSpecification;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.kucoin.KucoinExchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
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
import java.util.ArrayList;
import java.util.List;

import static com.trading.bot.util.TradeUtil.*;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    public static final int OUTPUT_SIZE = 8;
    public static final int TRAIN_DEEP = 24;
    public static final int PREDICT_DEEP = 4;
    public static final int CURRENCY_DELTA = 10;

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
    public static final CurrencyPair CURRENCY_PAIR = new CurrencyPair("BTC", "USDT");

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
                .seed(6)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(TRAIN_DEEP).nOut(4096).dropOut(0.5)
                        .build())
                .layer(new DenseLayer.Builder().nIn(4096).nOut(512)
                        .build())
                .layer(new DenseLayer.Builder().nIn(512).nOut(64)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX) //Override the global TANH activation with softmax for this layer
                        .nIn(64).nOut(OUTPUT_SIZE).build())
                .build();
    }

    @Bean
    public MultiLayerNetwork getModel(Exchange exchange, MultiLayerConfiguration config) throws IOException {
        LocalDateTime startDate = LocalDateTime.now(ZoneOffset.UTC).truncatedTo(ChronoUnit.DAYS);
        LocalDateTime endDate = LocalDateTime.now(ZoneOffset.UTC).truncatedTo(ChronoUnit.DAYS).plusDays(1);
        final List<KucoinKline> kucoinKlines = new ArrayList<>();

        for (long i = 0; i < 3; i++) {
            kucoinKlines.addAll(
                    getKucoinKlines(
                            exchange,
                            startDate.minusDays(i).toEpochSecond(ZoneOffset.UTC),
                            endDate.minusDays(i).toEpochSecond(ZoneOffset.UTC)));
        }

        float[][] floatData = new float[kucoinKlines.size() - TRAIN_DEEP - PREDICT_DEEP][TRAIN_DEEP];
        int[][] intLabels = new int[kucoinKlines.size() - TRAIN_DEEP - PREDICT_DEEP][OUTPUT_SIZE];
        for (int i = 0; i < kucoinKlines.size() - TRAIN_DEEP - PREDICT_DEEP; i++) {
            for (int y = 0; y < TRAIN_DEEP; y++) {
                floatData[i][y] = calcData(kucoinKlines, i, y, PREDICT_DEEP);
            }

            intLabels[i][getDelta(kucoinKlines, i)] = 1;
        }

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        try (INDArray indData = Nd4j.create(floatData);
             INDArray indLabels = Nd4j.create(intLabels)) {
            for (int i = 0; i < 1000; i++) {
                model.fit(indData, indLabels);
            }
        }

        final AmazonS3 s3 = AmazonS3ClientBuilder.standard()
                .withRegion(Regions.EU_CENTRAL_1)
                .build();

        try {
            String fileName = CURRENCY_PAIR.base + ".zip";
            String path = FilenameUtils.concat(
                    System.getProperty("java.io.tmpdir"), fileName);
            model.save(new File(path));

            s3.putObject(bucketName, fileName, new File(path));
        } catch (AmazonServiceException e) {
            logger.error(e.getErrorMessage());
        }

        return model;
    }
}

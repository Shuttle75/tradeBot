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
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.List;

import static com.trading.bot.util.TradeUtil.*;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    public static final long MINUS_DAYS = 0;
    public static final int OUTPUT_SIZE = 8;
    public static final int TRAIN_DEEP = 16;
    public static final int PREDICT_DEEP = 1;
    public static final int CURRENCY_DELTA = 10;
    public static final String BUCKET_NAME = "trade-crypto-models";

    public static final CurrencyPair CURRENCY_PAIR = new CurrencyPair("BTC", "USDT");

    @Bean
    public Exchange getXChangeExchange() {
        ExchangeSpecification exchangeSpecification = new ExchangeSpecification(KucoinExchange.class);

        exchangeSpecification.setUserName("alexey.osadchiy@gmail.com");
        exchangeSpecification.setApiKey("65380094ab6f110001a5a383");
        exchangeSpecification.setSecretKey("58da9817-38b5-4f89-85cd-718e54edd8fe");
        exchangeSpecification.setExchangeSpecificParametersItem("Use_Sandbox", false);
        exchangeSpecification.setExchangeSpecificParametersItem("passphrase", "cassandre");

        return ExchangeFactory.INSTANCE.createExchange(exchangeSpecification);
    }

    @Bean
    public MultiLayerConfiguration getConfig() {
        logger.info("Build model....");
        return new NeuralNetConfiguration.Builder()
                .seed(6)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.05))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(TRAIN_DEEP).nOut(4096)
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

        final long startDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.DAYS)
                .minusDays(MINUS_DAYS + 1)
                .toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC)
                .truncatedTo(ChronoUnit.DAYS)
                .minusDays(MINUS_DAYS)
                .toEpochSecond(ZoneOffset.UTC);

        final List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, endDate);
        kucoinKlines.addAll(getKucoinKlines(exchange, startDate - 86400, endDate - 86400));
        kucoinKlines.addAll(getKucoinKlines(exchange, startDate - 2 * 86400, endDate - 2 * 86400));

        float[][] floatData = new float[kucoinKlines.size() - TRAIN_DEEP][TRAIN_DEEP];
        int[][] intLabels = new int[kucoinKlines.size() - TRAIN_DEEP][OUTPUT_SIZE];
        for (int i = 0; i < kucoinKlines.size() - TRAIN_DEEP; i++) {
            for (int y = 0; y < TRAIN_DEEP; y++) {
                floatData[i][y] = calcData(kucoinKlines, i, y, PREDICT_DEEP);
            }

            intLabels[i][getDelta(kucoinKlines, i)] = 1;
        }

        INDArray indData = Nd4j.create(floatData);
        INDArray indLabels = Nd4j.create(intLabels);

        //run the model
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        for (int i = 0; i < 1000; i++) {
            model.fit(indData, indLabels);
        }

        final AmazonS3 s3 = AmazonS3ClientBuilder.standard()
                .withRegion(Regions.EU_CENTRAL_1)
                .build();

        try {
            String fileName = CURRENCY_PAIR.toString().replace("/", "-") + ".zip";
            String path = FilenameUtils.concat(
                    System.getProperty("java.io.tmpdir"), fileName);
            model.save(new File(path));

            s3.putObject(BUCKET_NAME, fileName, new File(path));
        } catch (AmazonServiceException e) {
            logger.error(e.getErrorMessage());
        }

        return model;
    }
}

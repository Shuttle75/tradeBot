package com.trading.bot.configuration;

import com.amazonaws.AmazonServiceException;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.amazonaws.services.s3.model.S3Object;
import com.amazonaws.services.s3.model.S3ObjectInputStream;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.knowm.xchange.Exchange;
import org.knowm.xchange.ExchangeFactory;
import org.knowm.xchange.ExchangeSpecification;
import org.knowm.xchange.currency.CurrencyPair;
import org.knowm.xchange.kucoin.KucoinExchange;
import org.knowm.xchange.kucoin.dto.response.KucoinKline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Collections;
import java.util.List;

import static com.trading.bot.util.TradeUtil.getKucoinKlines;

@Configuration
public class BotConfig {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    public static final int INPUT_SIZE = 2;
    public static final int OUTPUT_SIZE = 5;
    public static final int TRAIN_MINUTES = 480;
    public static final int PREDICT_DEEP = 1;
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
    public MultiLayerNetwork getModel(Exchange exchange) throws IOException {
        final String keyName = CURRENCY_PAIR.base + ".zip";
        final String path = FilenameUtils.concat(
                System.getProperty("java.io.tmpdir"), keyName);
        final AmazonS3 s3 = AmazonS3ClientBuilder.standard()
                .withRegion(Regions.EU_CENTRAL_1)
                .build();

        try {
            S3Object o = s3.getObject(bucketName, keyName);
            S3ObjectInputStream s3is = o.getObjectContent();
            FileOutputStream fos = new FileOutputStream(path);
            byte[] readBuf = new byte[1024];
            int readLen;
            while ((readLen = s3is.read(readBuf)) > 0) {
                fos.write(readBuf, 0, readLen);
            }
            s3is.close();
            fos.close();
        } catch (AmazonServiceException e) {
            logger.error(e.getErrorMessage());
            System.exit(1);
        } catch (IOException e) {
            logger.error(e.getMessage());
            System.exit(1);
        }

        MultiLayerNetwork net = null;
        try {
            net = MultiLayerNetwork.load(new File(path), true);
            logger.info("MultiLayerNetwork loaded from S3");
        } catch (IOException e) {
            logger.error(e.getMessage());
            System.exit(1);
        }

        reloadFirstHour(exchange, net);

        return net;
    }

    private void reloadFirstHour(Exchange exchange, MultiLayerNetwork net) throws IOException {
        final long startDate = LocalDateTime.now(ZoneOffset.UTC).minusMinutes(TRAIN_MINUTES).toEpochSecond(ZoneOffset.UTC);
        final long endDate = LocalDateTime.now(ZoneOffset.UTC).toEpochSecond(ZoneOffset.UTC);

        List<KucoinKline> kucoinKlines = getKucoinKlines(exchange, startDate, endDate);
        Collections.reverse(kucoinKlines);

        INDArray nextInput = Nd4j.zeros(1, INPUT_SIZE, TRAIN_MINUTES);
        for (int y = 0; y < TRAIN_MINUTES; y++) {
            nextInput.putScalar(new int[]{0, 0, y},
                    kucoinKlines.get(y).getClose()
                            .subtract(kucoinKlines.get(y).getOpen()).movePointLeft(2).floatValue());
            nextInput.putScalar(new int[]{0, 1, y},
                    kucoinKlines.get(y).getVolume().movePointLeft(2).floatValue());
        }
        net.rnnClearPreviousState();
        net.rnnTimeStep(nextInput);
        logger.info("First hour loaded to MultiLayerNetwork");
    }
}

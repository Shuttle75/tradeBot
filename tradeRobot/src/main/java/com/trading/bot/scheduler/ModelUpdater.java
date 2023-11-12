package com.trading.bot.scheduler;

import com.amazonaws.AmazonServiceException;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;

import static com.trading.bot.configuration.BotConfig.CURRENCY_PAIR;

@Service
public class ModelUpdater {
    protected final Logger logger = LoggerFactory.getLogger(getClass().getName());
    private MultiLayerNetwork model;

    @Value("${model.bucket}")
    public String bucketName;

    public ModelUpdater(MultiLayerNetwork model) {
        this.model = model;
    }

//    @Scheduled(cron = "0 0 * * * *")
    public void save() {

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
        }catch (IOException e) {
            logger.error(e.getMessage());
        }

        logger.info("Model saved to S3, IterationCount {}", model.getIterationCount());
    }
}

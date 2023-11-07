package com.trading.bot.dto;

import lombok.Getter;
import lombok.Setter;

import static com.trading.bot.configuration.BotConfig.PREDICT_DEEP;


public class Purchase {
    @Getter @Setter
    private boolean active;
    @Getter @Setter
    private float minPrice;
    @Getter @Setter
    private float curPrice;
    @Getter
    private float maxPrice;

    public void setMaxPrice(float maxPrice) {
        if (maxPrice > this.maxPrice) {
            this.maxPrice = maxPrice;
        }
    }
}

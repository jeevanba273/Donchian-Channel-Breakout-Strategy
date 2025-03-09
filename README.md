# Donchian Channel Breakout Strategy with Detailed Drawdown & Advanced Indicator Suite

*Donchian Channel Breakout: Detailed Drawdown & Advanced Indicator Suite* is an advanced quantitative trading dashboard built with Streamlit. It implements a classic Donchian Channel breakout strategy enriched with detailed drawdown analytics and a suite of advanced technical indicators. Users can dynamically enable and configure extra trade filters (RSI, Moving Averages, MACD, ADX, and Stochastic Oscillator) via the sidebar to tailor the strategy to their needs.

> *Live Demo:*
> Check out the live app on [Streamlit Community Cloud](https://donchian-channel-breakout-strategy.streamlit.app/).

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [Core Strategy – Donchian Channel Breakout](#core-strategy--donchian-channel-breakout)
  - [Advanced Indicator Filters](#advanced-indicator-filters)
    - [RSI Filter](#rsi-filter)
    - [Moving Average Filter](#moving-average-filter)
    - [MACD Filter](#macd-filter)
    - [ADX Filter](#adx-filter)
    - [Stochastic Oscillator Filter](#stochastic-oscillator-filter)
- [Data Source &amp; Processing](#data-source--processing)
- [Installation &amp; Setup](#installation--setup)

---

## Overview

This interactive dashboard provides a comprehensive toolkit for quantitative traders. At its core is a *Donchian Channel breakout* strategy. The app also integrates detailed drawdown analysis and advanced technical indicator filters, allowing you to fine-tune trade signals based on RSI, Moving Averages, MACD, ADX, and the Stochastic Oscillator.

---

## Features

- *Donchian Channel Breakout Strategy:*Automatically generates trade signals based on historical highs and lows.
- *Detailed Drawdown Analysis:*Visualizes maximum drawdowns, recovery periods, and other risk metrics.
- *Advanced Indicator Filters:*Optionally filter signals using:

  - *RSI Filter:* Confirm momentum via a 14-day RSI.
  - *Moving Average Filter:* Ensure price alignment with the trend.
  - *MACD Filter:* Validate momentum shifts using MACD crossovers.
  - *ADX Filter:* Take trades only in strong trending conditions.
  - *Stochastic Oscillator Filter:* Confirm overbought/oversold momentum.
- *Dynamic User Inputs:*All extra filters have configurable thresholds and periods directly via the sidebar.
- *Interactive Dashboard:*
  Multiple tabs display Overview, Ticker Data, Trades Data, Returns Analysis, Drawdown Analysis, and Performance Metrics.

---

## How It Works

### Core Strategy – Donchian Channel Breakout

- *Entry Conditions:*

  - *Long Entry:* A long signal is generated when today's closing price exceeds the highest price of the previous n days (the "upper channel"), while yesterday’s close is at or below that channel.
  - *Short Entry:* A short signal is generated when today's closing price falls below the lowest price of the previous n days (the "lower channel"), while yesterday’s close is at or above that channel.
- *Exit Conditions:*
  Once a trade is active, it remains open until a reversal occurs or an extra filter cancels the signal.

### Advanced Indicator Filters

Each extra indicator acts as a confirmation condition for the basic Donchian signal:

#### RSI Filter

- *Long Trades:* RSI must be *above* a user-specified threshold.
- *Short Trades:* RSI must be *below* that threshold.
- *Effect:* Cancels the trade signal if the RSI condition isn’t met.

#### Moving Average Filter

- *Long Trades:* Yesterday’s closing price must be *above* a designated long MA (e.g., 50-day).
- *Short Trades:* Yesterday’s closing price must be *below* a designated short MA (e.g., 200-day).
- *Effect:* Filters out trades not aligned with the trend.

#### MACD Filter

- *Long Trades:* The MACD line must be *above* its signal line.
- *Short Trades:* The MACD line must be *below* its signal line.
- *User-Defined Parameters:* Fast, slow, and signal periods.
- *Effect:* Confirms momentum shifts.

#### ADX Filter

- *Condition:* ADX must be *above* a user-specified threshold (e.g., >25).
- *Effect:* Ensures trades occur only in strong trending markets.

#### Stochastic Oscillator Filter

- *Long Trades:* The %K line must be *above* the %D line.
- *Short Trades:* The %K line must be *below* the %D line.
- *User-Defined Parameters:* %K period and smoothing (%D period).
- *Effect:* Validates trade signals based on momentum conditions.

*Combination:*
All enabled filters must be met for the signal to remain active. If any filter condition fails, the signal is cancelled.

---

## Data Source & Processing

The dataset was created from scratch using raw data sourced directly from the official NSE website:

- *Sources:*
  - Bhavcopy data (official NSE website)
  - MTO data (official NSE website)
- *Processing:*The data was cleaned and normalized to account for corporate actions such as bonuses, splits, and rights.
- *Time Period:*
  The dataset covers from *January 4, 2010* to *February 7, 2025*.

---

## Installation & Setup

1. *Clone the Repository:*

   ```bash
   git clone https://github.com/jeevanba273/Donchian-Channel-Breakout-Strategy.git
   cd Donchian-Channel-Breakout-Strategy
   ```

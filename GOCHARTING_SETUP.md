# GoCharting → Delta Scanner

The Streamlit scanner can use genuine GoCharting footprint/order-flow data instead of the OHLCV Delta proxy.

## 1. Add the Lipi script

Open `gocharting_orderflow_export.lipi` in this repository and copy its contents into the GoCharting Lipi editor.

The script exposes:
- Delta
- BuyVolume
- SellVolume
- MaxDelta
- MinDelta
- CVD
- Buy
- Sell
- Trades

GoCharting documents these as footprint/order-flow series.

## 2. Export chart data

Run the indicator on the required Nifty 200 stock and export the chart data to CSV.

The scanner accepts:
- one combined CSV containing Symbol, Time/DateTime and the order-flow fields, or
- multiple CSV files, one per stock.

If a CSV does not contain a Symbol column, name the file with the stock symbol, for example `RELIANCE.csv`. The app will infer RELIANCE.

## 3. Upload to Streamlit

Upload the CSV file(s) in the **GoCharting chart-data CSV export(s)** uploader.

The scanner automatically normalizes common column-name variants.

## 4. Use genuine order flow only

Enable **Require genuine GoCharting order flow** if you do not want the scanner to fall back to its OHLCV Delta proxy.

## Important

GoCharting's Lipi scripting environment exposes order-flow fields on the chart, but it is not an external multi-symbol data API. The workflow is therefore:

GoCharting footprint data → chart-data CSV export → Streamlit Delta scanner.

The scanner continues to label OHLCV-based calculations as a proxy and genuine footprint data as GOCHARTING_REAL.

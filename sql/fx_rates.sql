-- Exchange rates to US Dollars, implied by the data itself.
--
-- When a transfer is paid in one currency and received in another, the two amounts reveal
-- the rate used. We take the MEDIAN over many such transfers so a few odd rows don't matter.
-- Run on the TRAINING window only, so nothing from the test period shapes the features.

WITH train_txn AS (
    SELECT *
    FROM read_parquet($parquet)
    WHERE ts >= $start AND ts < $end
),

implied AS (
    -- Paid in X, received in USD: 1 unit of X = received / paid dollars
    SELECT payment_currency            AS currency,
           amount_received / amount_paid AS usd_per_unit
    FROM train_txn
    WHERE receiving_currency = 'US Dollar'
      AND payment_currency <> 'US Dollar'
      AND amount_paid > 0

    UNION ALL

    -- Paid in USD, received in X: 1 unit of X = paid / received dollars
    SELECT receiving_currency,
           amount_paid / amount_received
    FROM train_txn
    WHERE payment_currency = 'US Dollar'
      AND receiving_currency <> 'US Dollar'
      AND amount_received > 0
)

SELECT currency,
       median(usd_per_unit) AS usd_per_unit,
       count(*)             AS n_observations
FROM implied
GROUP BY currency

UNION ALL

SELECT 'US Dollar', 1.0, NULL

ORDER BY currency

-- Behavioural features: one row per account, built ONLY from transfers inside [$start, $end).
--
-- Expects a table `fx` (currency, usd_per_unit) built from the training window by sql/fx_rates.sql.
-- Counts and totals are divided by $window_days, because train (6 days) and test (4 days) have
-- different lengths: the same behaviour must give the same number in both.

WITH window_txn AS (
    SELECT *
    FROM read_parquet($parquet)
    WHERE ts >= $start AND ts < $end
),

all_accounts AS (
    -- Every account active in the window, including ones with only self-transfers,
    -- so this table lines up one-to-one with the labels table
    SELECT src AS account_id FROM window_txn
    UNION
    SELECT dst FROM window_txn
),

moves AS (
    -- Self-transfers are dropped: money that never leaves its owner would look like an
    -- instant in-and-out, i.e. a perfect fake mule signal
    SELECT * FROM window_txn WHERE src <> dst
),

sides AS (
    -- Each transfer appears twice: once from the sender's point of view, once from the receiver's.
    -- Amounts are taken in the currency that side actually saw, then converted to USD.
    SELECT m.src                                      AS account_id,
           'out'                                      AS direction,
           m.dst                                      AS counterparty,
           m.ts,
           m.amount_paid                              AS amount,
           m.amount_paid * fx.usd_per_unit            AS amount_usd,
           (m.from_bank <> m.to_bank)::INT            AS cross_bank,
           m.payment_format,
           (m.payment_currency <> m.receiving_currency)::INT AS ccy_mismatch
    FROM moves m
    JOIN fx ON fx.currency = m.payment_currency

    UNION ALL

    SELECT m.dst,
           'in',
           m.src,
           m.ts,
           m.amount_received,
           m.amount_received * fx.usd_per_unit,
           (m.from_bank <> m.to_bank)::INT,
           m.payment_format,
           (m.payment_currency <> m.receiving_currency)::INT
    FROM moves m
    JOIN fx ON fx.currency = m.receiving_currency
),

flows AS (
    SELECT account_id,
           count(*) FILTER (WHERE direction = 'in')                     AS in_count,
           count(*) FILTER (WHERE direction = 'out')                    AS out_count,
           coalesce(sum(amount_usd) FILTER (WHERE direction = 'in'), 0)  AS in_usd,
           coalesce(sum(amount_usd) FILTER (WHERE direction = 'out'), 0) AS out_usd,
           count(DISTINCT counterparty) FILTER (WHERE direction = 'in')  AS fan_in,
           count(DISTINCT counterparty) FILTER (WHERE direction = 'out') AS fan_out,
           -- moving money to other banks makes the trail harder to follow
           avg(cross_bank) FILTER (WHERE direction = 'out')              AS cross_bank_out_share,
           avg((payment_format = 'ACH')::INT)                            AS fmt_ach_share,
           avg((payment_format = 'Bitcoin')::INT)                        AS fmt_bitcoin_share,
           avg((payment_format = 'Cash')::INT)                           AS fmt_cash_share,
           avg((payment_format = 'Cheque')::INT)                         AS fmt_cheque_share,
           avg((payment_format = 'Credit Card')::INT)                    AS fmt_credit_card_share,
           avg((payment_format = 'Wire')::INT)                           AS fmt_wire_share,
           avg(ccy_mismatch)                                             AS ccy_mismatch_share,
           -- round in the transfer's own currency (people think "send 5,000", not "send $59.81");
           -- compared in whole cents to avoid floating-point noise
           avg((round(amount * 100)::BIGINT % ($round_multiple * 100) = 0)::INT) AS round_amount_share,
           -- spread of amounts relative to their size; low = similar-sized chunks
           stddev_pop(amount_usd) / nullif(avg(amount_usd), 0)           AS amount_cv,
           min(ts)                                                       AS first_ts
    FROM sides
    GROUP BY account_id
),

dwell AS (
    -- For every incoming transfer, find the FIRST outgoing transfer at or after it (ASOF join),
    -- then take the median waiting time per account. Median, so one odd gap doesn't dominate.
    SELECT i.account_id,
           median(date_diff('minute', i.ts, o.ts) / 60.0) AS dwell_median_hours
    FROM (SELECT account_id, ts FROM sides WHERE direction = 'in')  AS i
    ASOF JOIN
         (SELECT account_id, ts FROM sides WHERE direction = 'out') AS o
      ON i.account_id = o.account_id AND i.ts <= o.ts
    GROUP BY i.account_id
),

busiest_24h AS (
    -- Rolling window: for each transfer, count the account's transfers in the next 24 hours
    -- (timestamps are to the minute, so 23h59m after the current one). The maximum is the
    -- busiest 24-hour stretch, wherever it falls; a burst crossing midnight is not split.
    SELECT account_id, max(n_next_24h) AS busiest_24h_count
    FROM (
        SELECT account_id,
               count(*) OVER (PARTITION BY account_id ORDER BY ts
                              RANGE BETWEEN CURRENT ROW AND INTERVAL '23 hours 59 minutes' FOLLOWING) AS n_next_24h
        FROM sides
    )
    GROUP BY account_id
)

SELECT a.account_id,
       coalesce(f.in_count, 0)  / $window_days AS in_count_per_day,
       coalesce(f.out_count, 0) / $window_days AS out_count_per_day,
       coalesce(f.in_usd, 0)    / $window_days AS in_usd_per_day,
       coalesce(f.out_usd, 0)   / $window_days AS out_usd_per_day,
       coalesce(f.fan_in, 0)    / $window_days AS fan_in_per_day,
       coalesce(f.fan_out, 0)   / $window_days AS fan_out_per_day,
       -- no incoming money means the ratio is undefined: use 0 (in_count = 0 tells the model why)
       CASE WHEN f.in_usd > 0 THEN least(f.out_usd / f.in_usd, $ratio_cap) ELSE 0 END AS pass_through_ratio,
       -- no in->out pair: fill with one week (slow = not mule-like) and flag that it was filled
       coalesce(d.dwell_median_hours, $dwell_fill_hours)         AS dwell_median_hours,
       (d.account_id IS NOT NULL)::INT                           AS has_dwell,
       coalesce(b.busiest_24h_count / (f.in_count + f.out_count), 0) AS burst_share,
       -- 0 = active from the start of the window, near 1 = first appears at the very end;
       -- accounts with no money moving at all are treated as "not seen" (1.0)
       coalesce(date_diff('minute', $start::TIMESTAMP, f.first_ts) / ($window_days * 24 * 60.0), 1.0) AS first_seen_frac,
       coalesce(f.cross_bank_out_share, 0)  AS cross_bank_out_share,
       coalesce(f.fmt_ach_share, 0)         AS fmt_ach_share,
       coalesce(f.fmt_bitcoin_share, 0)     AS fmt_bitcoin_share,
       coalesce(f.fmt_cash_share, 0)        AS fmt_cash_share,
       coalesce(f.fmt_cheque_share, 0)      AS fmt_cheque_share,
       coalesce(f.fmt_credit_card_share, 0) AS fmt_credit_card_share,
       coalesce(f.fmt_wire_share, 0)        AS fmt_wire_share,
       coalesce(f.ccy_mismatch_share, 0)    AS ccy_mismatch_share,
       coalesce(f.round_amount_share, 0)    AS round_amount_share,
       coalesce(f.amount_cv, 0)             AS amount_cv
FROM all_accounts a
LEFT JOIN flows       f USING (account_id)
LEFT JOIN dwell       d USING (account_id)
LEFT JOIN busiest_24h b USING (account_id)
ORDER BY a.account_id

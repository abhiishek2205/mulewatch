-- Label B "pass-through mule", one row per account active in the window.
--
-- An account is a mule (is_mule = 1) if, inside [$start, $end), it both RECEIVED and SENT
-- at least one laundering transfer. Self-transfers are ignored: money that never leaves
-- its owner is not being passed on.
--
-- is_grey marks accounts that touched laundering money but are not mules by this rule
-- (only received, or only sent). It is for evaluation only and must never be a feature.

WITH window_txn AS (
    -- Only transfers inside this split's window: train never sees test days, and vice versa
    SELECT *
    FROM read_parquet($parquet)
    WHERE ts >= $start AND ts < $end
),

account_sides AS (
    -- Each transfer counts once for its sender (outgoing side) and once for its receiver (incoming side)
    SELECT src AS account_id,
           (is_laundering = 1 AND src <> dst)::INT AS laundering_out,
           0                                       AS laundering_in
    FROM window_txn
    UNION ALL
    SELECT dst,
           0,
           (is_laundering = 1 AND src <> dst)::INT
    FROM window_txn
),

per_account AS (
    SELECT account_id,
           sum(laundering_in)  AS laundering_in,
           sum(laundering_out) AS laundering_out
    FROM account_sides
    GROUP BY account_id
)

SELECT account_id,
       laundering_in,
       laundering_out,
       (laundering_in > 0 AND laundering_out > 0)::INT                     AS is_mule,
       (laundering_in + laundering_out > 0
        AND NOT (laundering_in > 0 AND laundering_out > 0))::INT           AS is_grey
FROM per_account
ORDER BY account_id

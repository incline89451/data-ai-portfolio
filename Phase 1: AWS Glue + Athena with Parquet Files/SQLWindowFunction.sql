WITH ranked_homers AS (
    SELECT 
        p.nameFirst,
        p.nameLast,
        t.name AS team_name,
        b.HR,
        RANK() OVER (PARTITION BY t.lgID ORDER BY b.HR DESC) AS hr_rank
    FROM batting b
    JOIN people p ON b.playerID = p.playerID
    JOIN teams t  ON b.teamID = t.teamID AND b.yearID = t.yearID
    WHERE b.yearID = 2010
      AND t.lgID = 'NL'
)
SELECT *
FROM ranked_homers
WHERE hr_rank <= 5;

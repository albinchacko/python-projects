# This source code is the implementation of the coursework tasks using the sqlite package.

import csv
import sqlite3

# SQIite database called census_income created
conn = sqlite3.connect(':memory:')
conn = sqlite3.connect('census_income.db')
c = conn.cursor()



# Task 1
# Income table created
c.execute("""CREATE TABLE Income (
    AAGE INTEGER,
    ACLSWKR TEXT,
    ADTIND TEXT,
    ADTOCC TEXT,
    AHGA TEXT,
    AHRSPAY REAL,
    AHSCOL TEXT,
    AMARITL TEXT,
    AMJIND TEXT,
    AMJOCC TEXT,
    ARACE TEXT,
    AREORGN TEXT,
    ASEX TEXT,
    AUNMEM TEXT,
    AUNTYPE TEXT,
    AWKSTAT TEXT,
    CAPGAIN REAL,
    CAPLOSS REAL,
    DIVVAL REAL,
    FILESTAT TEXT,
    GRINREG TEXT,
    GRINST TEXT,
    HDFMX TEXT,
    HHDREL TEXT,
    MARSUPWT REAL,
    MIGMTR1 TEXT,
    MIGMTR3 TEXT,
    MIGMTR4 TEXT,
    MIGSAME TEXT,
    MIGSUN TEXT,
    NOEMP REAL,
    PARENT TEXT,
    PEFNTVTY TEXT,
    PEMNTVTY TEXT,
    PENATVTY TEXT,
    PRCITSHP TEXT,
    SEOTR TEXT,
    VETQVA TEXT,
    VETYN TEXT,
    WKSWORK REAL,
    YEAR TEXT,
    TRGT TEXT
)""")

# Dataframe created for the given database
data = open("census-income.data")
row = csv.reader(data)

# Values inserted into Income table
c.executemany("INSERT INTO Income VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)




# Task 2
c.execute("ALTER TABLE Income RENAME TO Income_old")

# Column with the name SS_ID added to the Income table
c.execute("""CREATE TABLE Income (
    SS_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    AAGE INTEGER,
    ACLSWKR TEXT,
    ADTIND TEXT,
    ADTOCC TEXT,
    AHGA TEXT,
    AHRSPAY REAL,
    AHSCOL TEXT,
    AMARITL TEXT,
    AMJIND TEXT,
    AMJOCC TEXT,
    ARACE TEXT,
    AREORGN TEXT,
    ASEX TEXT,
    AUNMEM TEXT,
    AUNTYPE TEXT,
    AWKSTAT TEXT,
    CAPGAIN REAL,
    CAPLOSS REAL,
    DIVVAL REAL,
    FILESTAT TEXT,
    GRINREG TEXT,
    GRINST TEXT,
    HDFMX TEXT,
    HHDREL TEXT,
    MARSUPWT REAL,
    MIGMTR1 TEXT,
    MIGMTR3 TEXT,
    MIGMTR4 TEXT,
    MIGSAME TEXT,
    MIGSUN TEXT,
    NOEMP REAL,
    PARENT TEXT,
    PEFNTVTY TEXT,
    PEMNTVTY TEXT,
    PENATVTY TEXT,
    PRCITSHP TEXT,
    SEOTR TEXT,
    VETQVA TEXT,
    VETYN TEXT,
    WKSWORK REAL,
    YEAR TEXT,
    TRGT TEXT
)""")

# Values added from the table after renaming
c.execute("""INSERT INTO Income (
    AAGE,
    ACLSWKR,
    ADTIND,
    ADTOCC,
    AHGA,
    AHRSPAY,
    AHSCOL,
    AMARITL,
    AMJIND,
    AMJOCC,
    ARACE,
    AREORGN,
    ASEX,
    AUNMEM,
    AUNTYPE,
    AWKSTAT,
    CAPGAIN,
    CAPLOSS,
    DIVVAL,
    FILESTAT,
    GRINREG,
    GRINST,
    HDFMX,
    HHDREL,
    MARSUPWT,
    MIGMTR1,
    MIGMTR3,
    MIGMTR4,
    MIGSAME,
    MIGSUN,
    NOEMP,
    PARENT,
    PEFNTVTY,
    PEMNTVTY,
    PENATVTY,
    PRCITSHP,
    SEOTR,
    VETQVA,
    VETYN,
    WKSWORK,
    YEAR,
    TRGT)

    SELECT
    AAGE,
    ACLSWKR,
    ADTIND,
    ADTOCC,
    AHGA,
    AHRSPAY,
    AHSCOL,
    AMARITL,
    AMJIND,
    AMJOCC,
    ARACE,
    AREORGN,
    ASEX,
    AUNMEM,
    AUNTYPE,
    AWKSTAT,
    CAPGAIN,
    CAPLOSS,
    DIVVAL,
    FILESTAT,
    GRINREG,
    GRINST,
    HDFMX,
    HHDREL,
    MARSUPWT,
    MIGMTR1,
    MIGMTR3,
    MIGMTR4,
    MIGSAME,
    MIGSUN,
    NOEMP,
    PARENT,
    PEFNTVTY,
    PEMNTVTY,
    PENATVTY,
    PRCITSHP,
    SEOTR,
    VETQVA,
    VETYN,
    WKSWORK,
    YEAR,
    TRGT

    FROM Income_old
""")



# Task 3
# Query returns total number of males and females for each race group
c.execute("""SELECT DISTINCT ASEX, ARACE, COUNT(SS_ID) AS Gender_count_for_each_race 
    FROM Income GROUP BY ASEX, ARACE""")

print(c.fetchall())



# Task 4
# Query calculates and returns the average annual income for each race groups
c.execute("""SELECT DISTINCT ARACE, ASEX, ROUND(AVG( WKSWORK * (AHRSPAY * 40))) AS AVERAGE_INCOME 
    FROM Income 
    WHERE AHRSPAY > 0
    GROUP BY ARACE, ASEX""")

print(c.fetchall())



# Task 5
c.execute("DROP TABLE IF EXISTS Person")

# Table Person created from Income table with appropriate values of corresponding attributes
c.execute("""CREATE TABLE Person (
    SS_ID INTEGER PRIMARY KEY AUTOINCREMENT,    
    AGE INTEGER,
    EDU TEXT,
    SEX TEXT,
    CTZNSHP TEXT,
    FAM_18 TEXT,
    PRE_STAT TEXT,
    PRE_REG TEXT,
    HIS_ORGN TEXT,
    EMP_STAT TEXT) """)

c.execute("""INSERT INTO Person (
    AGE,
    EDU,
    SEX,
    CTZNSHP,
    FAM_18,
    PRE_STAT,
    PRE_REG,
    HIS_ORGN,
    EMP_STAT)

    SELECT 
    AAGE,
    AHGA,
    ASEX ,
    PRCITSHP,
    PARENT,
    GRINST,
    GRINREG,
    AREORGN,
    AWKSTAT

    FROM Income""")

c.execute("DROP TABLE IF EXISTS Job")

# Table Job created from Income table with appropriate values of corresponding attributes
c.execute("""CREATE TABLE Job (
    SS_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    DET_IND_COD TEXT,
    DET_OCC_COD TEXT,
    MAJ_IND_COD TEXT,
    MAJ_OCC_COD TEXT) """)

c.execute("""INSERT INTO Job (
    DET_IND_COD,
    DET_OCC_COD,
    MAJ_IND_COD,
    MAJ_OCC_COD)

    SELECT
    ADTIND,
    ADTOCC,
    AMJIND,
    AMJOCC

    FROM Income""")

c.execute("DROP TABLE IF EXISTS Pay")

# Table Pay created from Income table with appropriate values of corresponding attributes
c.execute("""CREATE TABLE Pay (
    SS_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    HR_WAGE REAL,
    WKSWRK REAL) """)

c.execute("""INSERT INTO Pay (
    HR_WAGE,
    WKSWRK)

    SELECT
    AHRSPAY,
    WKSWORK

    FROM Income""")



# Task 6.1
# Highest hourly wage selected and its columns retrieved
c.execute("""SELECT 
    Person.PRE_STAT AS State, 
    Job.MAJ_IND_COD AS Major_Industry, 
    Job.MAJ_OCC_COD AS Job_Type, 
    MAX(Pay.HR_WAGE) AS Highest_hourly_wage
	FROM Pay
	JOIN Job ON Job.SS_ID = Pay.SS_ID
	JOIN Person ON Person.SS_ID = Pay.SS_ID""")

# Number of people in each state with the highest paid job
c.execute("""SELECT 
    Person.PRE_STAT AS State, 
    COUNT(*) AS Number_of_people_in_each_state, 
    Job.MAJ_OCC_COD
    FROM Person
    JOIN Job ON Job.SS_ID = Person.SS_ID
    WHERE Job.MAJ_OCC_COD = ' Professional specialty'
    GROUP BY Person.PRE_STAT""")

# Number of people in the highest paid state
c.execute("""SELECT 
    COUNT(*) AS Number_of_people_in_highest_paid_state, 
    PRE_STAT
    FROM Person                
    WHERE PRE_STAT = ' Not in universe'""")

# Number of people in the highest paid job type
c.execute("""SELECT 
    COUNT(*) AS Number_of_people_in_highest_paid_job_type, 
    MAJ_OCC_COD
    FROM Job                
    WHERE MAJ_OCC_COD = ' Professional specialty'""")

# Number of people in the highest paid industry
c.execute("""SELECT 
    COUNT(*) AS Number_of_people_in_highest_paid_industry, 
    MAJ_IND_COD
    FROM Job                
    WHERE MAJ_IND_COD = ' Other professional services'""")

print(c.fetchall())



# Task 6.2
c.execute("DROP VIEW IF EXISTS WITH_HISP")

# Table view of people with hispanic origin and with education as Bachelors, Masters and PhD degrees
c.execute("""CREATE VIEW WITH_HISP (
    EDUCATION, HISPANIC_ORIGIN, INDUSTRY_TYPE, HOURLY_WAGE, WKS_WRKED_YEAR)
	AS SELECT EDU, HIS_ORGN, MAJ_IND_COD, HR_WAGE, WKSWRK 
	FROM Person
	JOIN Job ON Job.SS_ID = Person.SS_ID
	JOIN Pay ON Pay.SS_ID = Person.SS_ID
	WHERE (
	 Person.EDU = ' Bachelors degree(BA AB BS)'
     OR
     Person.EDU = ' Masters degree(MA MS MEng MEd MSW MBA)'
     OR
     Person.EDU = ' Doctorate degree(PhD EdD)')
     AND
     (Person.HIS_ORGN NOT IN (' All other', ' Do not know', ' NA')) """)

# Query returns the average hourly wage and average number of weeks worked per year for each industry
c.execute("""SELECT INDUSTRY_TYPE,
	avg(HOURLY_WAGE) AS AVERAGE_HR_WAGE, 
	avg(WKS_WRKED_YEAR) AS AVERAGE_WEEKS_WORKED
	FROM WITH_HISP
	GROUP BY INDUSTRY_TYPE""")

print(c.fetchall())

conn.commit()
c.close()
conn.close()
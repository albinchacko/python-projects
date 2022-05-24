# This source code is the implementation of coursework tasks using the dplython package.

import pandas as pd
from dplython import *

# Dataframe created using the given .data file
Income_df = pd.DataFrame(pd.read_csv("./census-income.data", header=None))

# Constructed a row to insert as a data frame header
Income_df.columns = ['AAGE',
                     'ACLSWKR',
                     'ADTIND',
                     'ADTOCC',
                     'AHGA',
                     'AHRSPAY',
                     'AHSCOL',
                     'AMARITL',
                     'AMJIND',
                     'AMJOCC',
                     'ARACE',
                     'AREORGN',
                     'ASEX',
                     'AUNMEM',
                     'AUNTYPE',
                     'AWKSTAT',
                     'CAPGAIN',
                     'CAPLOSS',
                     'DIVVAL',
                     'FILESTAT',
                     'GRINREG',
                     'GRINST',
                     'HDFMX',
                     'HHDREL',
                     'MARSUPWT',
                     'MIGMTR1',
                     'MIGMTR3',
                     'MIGMTR4',
                     'MIGSAME',
                     'MIGSUN',
                     'NOEMP',
                     'PARENT',
                     'PEFNTVTY',
                     'PEMNTVTY',
                     'PENATVTY',
                     'PRCITSHP',
                     'SEOTR',
                     'VETQVA',
                     'VETYN',
                     'WKSWORK',
                     'YEAR',
                     'TRGT']



# Task:1 - Dataframe(Income table) for dplython object created
Income_old = DplyFrame(Income_df)



# Task:2 - Column with the name SS_ID added to the Income table
Income = Income_old >> \
         mutate(SS_ID=range(1, nrow(Income_old) + 1))



# Task:3 - Query returns total number of males and females for each race group
race_gender = Income >> \
              group_by(X.ASEX, X.ARACE) >> \
              summarize(Gender_count_for_each_race = X.SS_ID.count())



# Task:4 - Query calculates and returns the average annual income for each race groups
average_income = Income >> \
                 sift(X.AHRSPAY > 0) >> \
                 group_by(X.ASEX, X.ARACE) >> \
                 mutate(Annual_income=X.WKSWORK * (X.AHRSPAY * 40)) >> \
                 summarize(average_annual_income=X.Annual_income.mean())



# Task:5 - Tables Person, Job and Pay created from Income table with appropriate values of corresponding attributes
Person = Income >> \
         select(X.SS_ID, X.AAGE, X.AHGA, X.ASEX, X.PRCITSHP, X.PARENT, X.GRINST, X.GRINREG, X.AREORGN, X.AWKSTAT)
Job = Income >> \
      select(X.SS_ID, X.ADTIND, X.ADTOCC, X.AMJOCC, X.AMJIND)
Pay = Income >> \
      select(X.SS_ID, X.AHRSPAY, X.WKSWORK)



# Task:6.1 - Highest hourly wage selected
highest_wage = Pay >> \
               inner_join(Person, by=[('SS_ID','SS_ID')]) >> \
               inner_join(Job, by=[('SS_ID','SS_ID')]) >> \
               sift(X.AHRSPAY == X.AHRSPAY.max()) >> \
               select(X.AHRSPAY, X.AMJOCC, X.GRINST)

# Task:6.1 - Number of people in each state with the highest paid job
highest_job_states = Job >> \
                     inner_join(Person, by=[('SS_ID','SS_ID')]) >> \
                     sift(X.AMJOCC == ' Professional specialty') >> \
                     group_by(X.GRINST,X.AMJOCC) >> \
                     select(X.SS_ID, X.AMJIND, X.AMJOCC, X.GRINST) >> \
                     summarize(Number_of_people_in_each_state=X.SS_ID.count())

# Task:6.1 - Number of people in the highest paid state
highest_paid_state = Person >> \
                     sift(X.GRINST == ' Not in universe') >> \
                     select(X.SS_ID, X.GRINST) >> \
                     summarize(Number_of_people_in_highest_paid_state=X.SS_ID.count())

# Task:6.1 - Number of people in the highest paid job type
highest_paid_jobtype = Job >> sift(X.AMJOCC == ' Professional specialty') >> \
                       select(X.SS_ID, X.AMJOCC) >> \
                       summarize(Number_of_people_in_highest_paid_job_type=X.SS_ID.count())

# Task:6.1 - Number of people in the highest paid industry
highest_paid_industry = Job >> sift(X.AMJIND == ' Other professional services') >> \
                        select(X.SS_ID, X.AMJIND) >> \
                        summarize(Number_of_people_in_highest_paid_industry=X.SS_ID.count())



# Task:6.2 - Dataframe of people with hispanic origin and with education as Bachelors, Masters and PhD degrees
with_hisp = Person >> \
            sift(((X.AHGA == ' Bachelors degree(BA AB BS)') |
                           (X.AHGA == ' Masters degree(MA MS MEng MEd MSW MBA)') |
                           (X.AHGA == ' Doctorate degree(PhD EdD)')) &
                           ((X.AREORGN != ' All other') &
                           (X.AREORGN != ' Do not know') &
                           (X.AREORGN != ' NA'))) >> \
            inner_join(Job, by=[('SS_ID','SS_ID')]) >> \
            inner_join(Pay, by=[('SS_ID','SS_ID')]) >> \
            select(X.AHGA, X.AREORGN, X.AMJIND, X.AHRSPAY, X.WKSWORK)

# Task:6.2 - Query returns the average hourly wage and average number of weeks worked per year for each industry
employment_average = with_hisp >> \
                     group_by(X.AMJIND) >> \
                     summarize(Average_hourly_wage=X.AHRSPAY.mean(), Average_weeks_worked=X.WKSWORK.mean())



